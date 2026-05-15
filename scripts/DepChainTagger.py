import estnltk
import json
from typing import (
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Iterable,
    Self,
    Any,
    Callable,
    TypeAlias,
    Literal,
)
from dataclasses import dataclass, field
from enum import Enum
from estnltk.taggers import Tagger
from estnltk import Layer, Text

NodePredicate: TypeAlias = Callable[[Any], bool]


class ConditionMode(str, Enum):
    """
    Supported matching modes for scalar attributes.
    """

    EXACT = "exact"  # Match when actual value is exactly equal to expected value
    NEGATION = "negation"  # Match when actual value is not equal to expected value
    WILDCARD = "wildcard"  # Match any value (expected value is ignored, must be None)
    MEMBERSHIP = "membership"  # Match when actual value is in the expected iterable (list, tuple, set, etc.)


class DirectionMode(str, Enum):
    """
    Supported edge direction modes for iterating edges in the syntax graph.
    """

    UP = "up"  # Move from id to head (up the tree)
    DOWN = "down"  # Move from head to id (down the tree)
    BOTH = "both"  # Include both up and down edges (default)


class EdgeContext:
    """
    Context for an edge in the dependency graph, used for matching edges during feature extraction.
    """

    direction: DirectionMode
    deprel: Optional[str]
    hops: int
    crosses_sentence: bool


class SyntaxGraphIndex:
    """
    A class to represent the dependency syntax graph of a sentence, indexed by token IDs.

    The graph is built from an estnltk Layer containing the sentence annotations, and provides methods to access nodes, parents, children, and edges in the dependency graph.

    ## Attributes:
    - **sentences_layer** (`estnltk.Layer`): The layer containing the stanza syntax annotations for the sentence from which the graph index is built.
    - **nodes_by_id** (`Dict[int, estnltk.Span]`): A mapping from token IDs to their corresponding estnltk Span annotations.
    - **parent_by_id** (`Dict[int, Optional[int]]`): A mapping from token IDs to their parent token IDs in the dependency graph.
    - **children_by_id** (`Dict[int, List[int]]`): A mapping from token IDs to a list of their child token IDs in the dependency graph.
    - **token_order** (`List[int]`): A list of token IDs in the order they appear in the sentence.
    - **sent_id** (`Optional[int]`): The ID of the sentence being indexed.
    - **sentence_span** (`Optional[Tuple[int, int]]`): The character span of the sentence in the original text.

    ## Methods:
    - :func:`~SyntaxGraphIndex.__init__`: Initializes the graph index from the given sentences layer.
    - :func:`~SyntaxGraphIndex.build_from_layer`: Builds the graph index from the provided sentences layer.
    - :func:`~SyntaxGraphIndex.get_node`: Retrieves the estnltk Span annotation for a given token ID.
    - :func:`~SyntaxGraphIndex.get_parent`: Retrieves the parent node of a given token ID in the dependency graph.
    - :func:`~SyntaxGraphIndex.get_children`: Retrieves the child nodes of a given token ID in the dependency graph.
    - :func:`~SyntaxGraphIndex.iter_nodes`: Iterates over all nodes in the graph in the order they appear in the sentence.
    - :func:`~SyntaxGraphIndex.iter_edges`: Iterates over all edges in the graph, optionally filtering by direction (up, down, or both).
    - :func:`~SyntaxGraphIndex.get_root_nodes`: Retrieves the root nodes of the dependency graph (nodes with no parent).
    - :func:`~SyntaxGraphIndex.has_node`: Checks if a given token ID exists in the graph.
    """

    stanza_syntax: estnltk.Layer
    nodes_by_id: Dict[int, estnltk.Span] = {}
    parent_by_id: Dict[int, Optional[int]] = {}
    children_by_id: Dict[int, List[int]] = {}
    token_order: List[int] = []
    sent_id: Optional[int] = None
    sentence_span: Optional[Tuple[int, int]] = None
    lookup_cache: Dict[Tuple, Any] = {}

    def __init__(
        self: Self,
        stanza_syntax_layer: estnltk.Layer,
        sentence_id: Optional[int] = None,
        sentence_span: Optional[Tuple[int, int]] = None,
    ) -> None:
        """
        Initializes the SyntaxGraphIndex from the given sentences layer.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being initialized.
            stanza_syntax_layer (estnltk.Layer): The layer containing the stanza syntax annotations for the sentence from which to build the graph index.
            sentence_id (Optional[int], optional): The ID of the sentence being indexed. Defaults to None.
            sentence_span (Optional[Tuple[int, int]], optional): The character span of the sentence in the original text. Defaults to None.
        """

        # Initialize the graph index from the given sentences layer
        self.stanza_syntax: estnltk.Layer = stanza_syntax_layer
        self.nodes_by_id: Dict[int, estnltk.Span] = {}
        self.parent_by_id: Dict[int, Optional[int]] = {}
        self.children_by_id: Dict[int, List[int]] = {}
        self.token_order: List[int] = []
        self.sent_id: Optional[int] = sentence_id
        self.sentence_span: Optional[Tuple[int, int]] = sentence_span
        self.lookup_cache = {}

        # Build the graph index from the sentences layer
        self.build_from_layer(self.stanza_syntax)

        # Validate the graph structure (optional, can be commented out if not needed)
        if not self._validate_tree():
            raise ValueError(
                "The provided stanza syntax layer does not form a valid tree structure."
            )

    def build_from_layer(self: Self, stanza_syntax: estnltk.Layer) -> None:
        """
        Builds the graph index from the provided syntax layer.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being built.
            stanza_syntax (estnltk.Layer): The layer containing the stanza syntax annotations for the sentence from which to build the graph index.
        """
        # Build the graph index from the provided sentences layer
        for ann in stanza_syntax:
            if ann.id in self.nodes_by_id:
                raise ValueError(f"Duplicate token id encountered: {ann.id}")
            self.nodes_by_id[ann.id] = ann
            self.parent_by_id[ann.id] = ann.head
            self.children_by_id[ann.id] = []
            self.token_order.append(ann.id)

        # Populate the children_by_id mapping based on the parent_by_id mapping
        for ann in stanza_syntax:
            if ann.head == 0:
                continue
            if ann.head not in self.children_by_id:
                raise ValueError(
                    f"Invalid head reference: token {ann.id} points to missing head {ann.head}."
                )
            self.children_by_id[ann.head].append(ann.id)

    def get_node(self: Self, token_id: int) -> Optional[estnltk.Span]:
        """
        Gets the estnltk Span annotation for a given token ID.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being queried.
            token_id (int): The ID of the token for which to retrieve the annotation.

        Returns:
            Optional[estnltk.Span]: The estnltk Span annotation corresponding to the given token ID, or None if the token ID does not exist in the graph index.
        """
        return self.nodes_by_id.get(token_id)

    def get_parent(self: Self, token_id: int) -> Optional[estnltk.Span]:
        """
        Gets the parent node of a given token ID in the dependency graph.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being queried.
            token_id (int): The ID of the token for which to retrieve the parent node.

        Returns:
            Optional[estnltk.Span]: The estnltk Span annotation corresponding to the parent node of the given token ID in the dependency graph, or None if the token ID does not exist in the graph index or if it is a root node (with no parent).
        """
        parent_id = self.parent_by_id.get(token_id)
        if parent_id is not None:
            return self.nodes_by_id.get(parent_id)
        return None

    def get_children(self: Self, token_id: int) -> List[estnltk.Span]:
        """
        Gets the child nodes of a given token ID in the dependency graph.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being queried.
            token_id (int): The ID of the token for which to retrieve the child nodes.

        Returns:
            List[estnltk.Span]: A list of estnltk Span annotations corresponding to the child nodes of the given token ID in the dependency graph. If the token ID does not exist in the graph index or has no children, an empty list is returned.
        """
        child_ids = self.children_by_id.get(token_id, [])
        return [self.nodes_by_id[child_id] for child_id in child_ids]

    def iter_nodes(self: Self) -> Iterable[estnltk.Span]:
        """
        Iterates over all nodes in the graph in the order they appear in the sentence.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being iterated over.

        Returns:
            Iterable[estnltk.Span]: An iterator that yields estnltk Span annotations for each node in the graph, in the order they appear in the sentence.

        Yields:
            estnltk.Span: The estnltk Span annotation for each node in the graph, yielded in the order they appear in the sentence.
        """
        for token_id in self.token_order:
            yield self.nodes_by_id[token_id]

    def iter_edges(
        self: Self, direction: str = "both"
    ) -> Iterable[Tuple[Optional[estnltk.Span], Optional[estnltk.Span], str]]:
        """
        Iterates over all edges in the graph, optionally filtering by direction (up, down, or both).

        Args:
            self (Self): The instance of the SyntaxGraphIndex being iterated over.
            direction (str, optional): The direction of edges to iterate over. Can be "up" for parent-child edges, "down" for child-parent edges, or "both" for all edges. Defaults to "both".

        Returns:
            Iterable[Tuple[Optional[estnltk.Span], Optional[estnltk.Span], str]]: _description_

        Yields:
            Iterator[Iterable[Tuple[Optional[estnltk.Span], Optional[estnltk.Span], str]]]: _description_
        """
        for token_id in self.token_order:
            node = self.nodes_by_id[token_id]
            parent_id = self.parent_by_id.get(token_id)
            if parent_id is not None and parent_id != 0:
                parent_node = self.nodes_by_id.get(parent_id)
                if direction in ["both", "up"]:  # Move from id to head (up the tree)
                    yield (node, parent_node, "up")
                if direction in [
                    "both",
                    "down",
                ]:  # Move from head to id (down the tree)
                    if (
                        parent_id != 0
                    ):  # Skip the root node which has no parent (head = 0), so we don't yield a down edge for it
                        yield (parent_node, node, "down")

    def get_root_nodes(self: Self) -> List[estnltk.Span]:
        """
        Gets the root nodes of the dependency graph (nodes with no parent).

        Args:
            self (Self): The instance of the SyntaxGraphIndex being queried.

        Returns:
            List[estnltk.Span]: A list of estnltk Span annotations corresponding to the root nodes of the dependency graph (nodes with no parent). If there are no root nodes, an empty list is returned.
        """
        root_nodes = []
        for token_id in self.token_order:
            if self.parent_by_id.get(token_id) == 0:  # Root nodes have head = 0
                root_nodes.append(self.nodes_by_id[token_id])
        return root_nodes

    def has_node(self: Self, token_id: int) -> bool:
        """
        Checks if a given token ID exists in the graph.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being queried.
            token_id (int): The ID of the token to check for existence in the graph.

        Returns:
            bool: True if the given token ID exists in the graph index, False otherwise.
        """
        return token_id in self.nodes_by_id

    def _validate_tree(self: Self) -> bool:
        """
        Validates that the graph forms a proper tree structure. Checks for cycles, missing heads, and orphan references.

        Args:
            self (Self): The instance of the SyntaxGraphIndex being validated.

        Returns:
            bool: True if the graph forms a valid tree structure, False otherwise. A valid tree structure means that there are no cycles in the parent-child relationships, all nodes have a valid head (except for root nodes), and there are no orphan references (nodes that reference a non-existent head).
        """
        # Check for cycles using a depth-first search
        visited = set()

        def _dfs(node_id: int, parent_id: Optional[int]) -> bool:
            """
            Performs a depth-first search to detect cycles in the graph.

            Args:
                node_id (int): The ID of the current node being visited in the depth-first search.
                parent_id (Optional[int]): The ID of the parent node of the current node in the depth-first search. This is used to avoid false positive cycle detection when traversing back to the parent node.

            Returns:
                bool: True if no cycles are detected in the graph, False if a cycle is detected. A cycle is detected if a node is visited more than once during the depth-first search, indicating that there is a circular reference in the parent-child relationships of the graph.
            """
            if node_id in visited:
                return False  # Cycle detected
            visited.add(node_id)
            for child_id in self.children_by_id.get(node_id, []):
                if child_id == parent_id:
                    continue  # Skip the parent node to avoid false positive cycle detection
                if not _dfs(child_id, node_id):
                    return False
            return True

        # Check for valid heads and orphan references
        for token_id in self.token_order:
            parent_id = self.parent_by_id.get(token_id)
            if (
                parent_id is not None  # A head is specified
                and parent_id != 0  # Root nodes have head = 0, so we allow that
                and parent_id
                not in self.nodes_by_id  # The specified head does not exist in the graph
            ):
                return False  # Orphan reference detected (node references a non-existent head)

        # Check for cycles starting from root nodes
        root_nodes = self.get_root_nodes()
        if not root_nodes:
            return False

        for root_node in root_nodes:
            root_id = int(getattr(root_node, "id"))
            if not _dfs(root_id, None):
                return False  # Cycle detected

        # Every node must be reachable from some root.
        if len(visited) != len(self.token_order):
            return False

        return True


@dataclass(frozen=True, slots=True)
class ValueCondition:
    """
    Match one scalar value using exact, negation, or wildcard logic.

    ## Attributes:
    - **mode** (`ConditionMode`): The matching mode to use (EXACT, NEGATION, or WILDCARD).
    - **value** (`Any`, optional): The value to match against for EXACT and NEGATION modes. Must be None for WILDCARD mode. Defaults to None.
    - **allow_missing** (`bool`, optional): Whether to allow missing values (e.g., None, empty string, or other specified missing markers) as a match. Defaults to False.
    - **normalizer** (`Optional[Callable[[Any], Any]]`, optional): An optional function to normalize both the expected value and the actual value before comparison. This can be used to implement case-insensitive matching, for example. Defaults to None (no normalization).
    - **missing_markers** (`Tuple[Any, ...]`, optional): A tuple of values that should be treated as missing when `allow_missing` is True. Defaults to (None, "", "_").
    ## Methods:
    - :func:`~ValueCondition.matches`: Checks whether a given actual value satisfies this condition.
    - :func:`~ValueCondition.describe`: Returns a human-readable explanation of the condition.
    """

    mode: ConditionMode
    value: Any = None
    allow_missing: bool = False
    normalizer: Optional[Callable[[Any], Any]] = None
    missing_markers: Tuple[Any, ...] = (None, "", "_")

    def __post_init__(self: Self) -> None:
        """
        Validate config and pre-normalise expected value once.
        """
        self._validate_or_raise()

        if (
            self.normalizer is not None
            and self.value is not None
            and self.mode is not ConditionMode.WILDCARD
        ):
            # dataclass is frozen, so we use object.__setattr__
            object.__setattr__(self, "value", self.normalizer(self.value))

    def matches(self: Self, actual_value: Any) -> bool:
        """
        Check whether `actual_value` satisfies this condition.

        Args:
            actual_value (Any): The value to check against this condition.

        Returns:
            bool: True if the actual value satisfies the condition, False otherwise.
        """
        if self.mode is ConditionMode.WILDCARD:
            return True

        if self._is_missing(actual_value):
            return self.allow_missing

        if self.normalizer is not None:
            actual_value = self.normalizer(actual_value)

        if self.mode is ConditionMode.EXACT:
            return actual_value == self.value
        if self.mode is ConditionMode.NEGATION:
            return actual_value != self.value
        if self.mode is ConditionMode.MEMBERSHIP:
            return actual_value in self.value

        # Defensive fallback; should be unreachable due to validation.
        raise ValueError(f"Unsupported mode: {self.mode}")

    def describe(self: Self) -> str:
        """
        Return a human-readable explanation of the condition.
        """
        if self.mode is ConditionMode.EXACT:
            return f"Value must be exactly {self.value!r}"
        if self.mode is ConditionMode.NEGATION:
            return f"Value must not be {self.value!r}"
        if self.mode is ConditionMode.WILDCARD:
            return "Value can be any value"
        if self.mode is ConditionMode.MEMBERSHIP:
            return f"Value must be in {self.value!r}"
        raise ValueError(f"Unsupported mode: {self.mode}")

    def _is_missing(self: Self, value: Any) -> bool:
        """
        Return True when value should be treated as missing.

        Args:
            value (Any): The value to check for missingness.
        Returns:
            bool: True if the value should be treated as missing, False otherwise.
        """
        return value in self.missing_markers

    def _validate_or_raise(self: Self) -> None:
        """
        Validate constructor arguments and raise explicit errors.
        """
        if not isinstance(self.mode, ConditionMode):
            raise TypeError(
                "mode must be ConditionMode (EXACT, NEGATION, WILDCARD, or MEMBERSHIP)."
            )

        if self.mode in (ConditionMode.EXACT, ConditionMode.NEGATION):
            if self.value is None:
                raise ValueError("value is required for EXACT and NEGATION modes.")

        if self.mode is ConditionMode.WILDCARD and self.value is not None:
            raise ValueError("value must be None when mode is WILDCARD.")

        if self.mode is ConditionMode.MEMBERSHIP:
            if self.value is None:
                raise ValueError("value is required for MEMBERSHIP mode.")
            # Check if value is iterable (but not string)
            if isinstance(self.value, str):
                raise TypeError(
                    "value for MEMBERSHIP mode must be an iterable (list, tuple, set) but not a string."
                )
            try:
                iter(self.value)
            except TypeError:
                raise TypeError(
                    f"value for MEMBERSHIP mode must be iterable, got {type(self.value).__name__}."
                )

        if self.normalizer is not None and not callable(self.normalizer):
            raise TypeError("normalizer must be callable or None.")


@dataclass(frozen=True, slots=True)
class FeatureCondition:
    """
    Match a dictionary of features using exact, negation, or wildcard logic.
    ## Attributes:
    - **mode** (`ConditionMode`): The matching mode to use (EXACT, NEGATION, or WILDCARD).
    - **required** (`Optional[Dict[str, Any]]`): A dictionary of feature keys and their expected values that must be present for the condition to match. When `mode` is EXACT, all `required` pairs must be present and equal. When `mode` is NEGATION, reject if all `required` pairs match simultaneously.
    Defaults to None.
    - **forbidden** (`Optional[Dict[str, Any]]`): A dictionary of feature keys and their values that must not be present for the condition to match. When `mode` is EXACT, all `forbidden` pairs must not be present with equal value. When `mode` is NEGATION, reject if any `forbidden` pair appears with equal value.
    Defaults to None.
    - **allow_extra_keys** (`bool`, optional): Whether to allow extra keys in the actual features that are not specified in either `required` or `forbidden`. When `mode` is EXACT and `allow_extra_keys` is False, no keys outside union of `required` and `forbidden` are allowed. When `mode` is NEGATION, `allow_extra_keys` has no effect since we only check the specified keys.
    Defaults to False.
    - **allow_missing** (`bool`, optional): Whether to allow missing keys (i.e., keys that are specified in `required` but not present in the actual features) as a match.
    Defaults to False.
    - **normalizer** (`Optional[Callable[[Any], Any]]`, optional): An optional function to normalize both the expected values and the actual values before comparison. This can be used to implement case-insensitive matching, for example.
    Defaults to None (no normalization).
    ## Methods:
    - :func:`~FeatureCondition.matches`: Checks whether a given actual features dictionary satisfies this condition.
    - :func:`~FeatureCondition.describe`: Returns a human-readable explanation of the condition.
    """

    mode: ConditionMode
    required: Optional[Dict[str, Any]] = None
    forbidden: Optional[Dict[str, Any]] = None
    allow_extra_keys: Optional[bool] = False
    allow_missing: Optional[bool] = False
    normalizer: Optional[Callable[[Any], Any]] = None

    def __post_init__(self: Self) -> None:
        """
        Validate config and pre-normalise expected values once.
        """
        self._validate_or_raise()

        if self.normalizer is not None:
            if self.required is not None:
                # dataclass is frozen, so we use object.__setattr__
                object.__setattr__(
                    self,
                    "required",
                    {k: self.normalizer(v) for k, v in self.required.items()},
                )
            if self.forbidden is not None:
                # dataclass is frozen, so we use object.__setattr__
                object.__setattr__(
                    self,
                    "forbidden",
                    {k: self.normalizer(v) for k, v in self.forbidden.items()},
                )

    def matches(self: Self, actual_value: Dict[str, Any] | None) -> bool:
        """
        Check whether `actual_value` satisfies this condition.

        Args:
            actual_value (Dict[str, Any] | None): The value to check against this condition.

        Returns:
            bool: True if the actual value satisfies the condition, False otherwise.
        """
        if self.mode is ConditionMode.WILDCARD:
            return True

        if not isinstance(actual_value, dict):
            return False

        def norm(v: Any) -> Any:
            """
            Apply normalizer if defined, otherwise return value as is.

            Args:
                v (Any): The value to normalize.

            Returns:
                Any: The normalized value if normalizer is defined, otherwise the original value.
            """
            return self.normalizer(v) if self.normalizer is not None else v

        required = self.required or {}
        forbidden = self.forbidden or {}

        if self.mode is ConditionMode.EXACT:
            # Required checks
            for key, expected in required.items():
                if key not in actual_value:
                    if not self.allow_missing:
                        return False
                    continue
                if norm(actual_value[key]) != expected:
                    return False

            # Forbidden checks
            for key, forbidden_value in forbidden.items():
                if key in actual_value and norm(actual_value[key]) == forbidden_value:
                    return False

            # Extra-key policy check
            if not self.allow_extra_keys:
                allowed_keys = set(required.keys()) | set(forbidden.keys())
                if any(key not in allowed_keys for key in actual_value.keys()):
                    return False

            return True

        if self.mode is ConditionMode.NEGATION:
            # Negate required pattern: if full required pattern matches, reject
            if required:
                full_required_match = True
                for key, expected in required.items():
                    if key not in actual_value or norm(actual_value[key]) != expected:
                        full_required_match = False
                        break
                if full_required_match:
                    return False

            # Forbidden still rejects if any forbidden pair matches
            for key, forbidden_value in forbidden.items():
                if key in actual_value and norm(actual_value[key]) == forbidden_value:
                    return False

            return True

        raise ValueError(f"Unsupported mode: {self.mode}")

    def describe(self: Self) -> str:
        """
        Return a human-readable explanation of the condition.
        """
        if self.mode is ConditionMode.EXACT:
            return f"Features must include {self.required!r} and exclude {self.forbidden!r}"
        if self.mode is ConditionMode.NEGATION:
            return f"Features must not include {self.forbidden!r} and must not exclude {self.required!r}"
        if self.mode is ConditionMode.WILDCARD:
            return "Features can be any value"
        raise ValueError(f"Unsupported mode: {self.mode}")

    def _validate_or_raise(self) -> None:
        """
        Validate constructor arguments with explicit, actionable errors.
        """
        if not isinstance(self.mode, ConditionMode):
            raise TypeError("mode must be ConditionMode.")

        if self.required is not None and not isinstance(self.required, dict):
            raise TypeError("required must be dict or None.")

        if self.forbidden is not None and not isinstance(self.forbidden, dict):
            raise TypeError("forbidden must be dict or None.")

        if self.normalizer is not None and not callable(self.normalizer):
            raise TypeError("normalizer must be callable or None.")

        if self.mode in (ConditionMode.EXACT, ConditionMode.NEGATION):
            if self.required is None and self.forbidden is None:
                raise ValueError(
                    "Provide required and/or forbidden for EXACT/NEGATION."
                )

        if self.mode is ConditionMode.WILDCARD:
            if self.required is not None or self.forbidden is not None:
                raise ValueError("required/forbidden must be None for WILDCARD mode.")


@dataclass(frozen=True, slots=True)
class NodeConstraint:
    """
    Constraint for a single node in the dependency graph, used for matching nodes during feature extraction.

    ## Attributes:
    - **role** (`str`): The role of the node in the dependency chain (e.g., "self", "parent", "child", "sibling", etc.).
    - **upostag_condition** (`Optional[ValueCondition]`): An optional ValueCondition to match the UPOS tag of the node.
    - **xpostag_condition** (`Optional[ValueCondition]`): An optional ValueCondition to match the XPOS tag of the node.
    - **lemma_condition** (`Optional[ValueCondition]`): An optional ValueCondition to match the lemma of the node.
    - **deprel_at_node_condition** (`Optional[ValueCondition]`): An optional ValueCondition to match the dependency relation (deprel) at the node itself.
    - **feats_condition** (`Optional[FeatureCondition]`): An optional FeatureCondition to match the morphological features (feats) of the node.
    - **extra_predicates** (`Optional[Tuple[NodePredicate, ...]]`): An optional tuple of additional callables that take the node annotation as input and return a boolean indicating whether the node satisfies some custom condition. These can be used for more complex checks that are not easily expressed with the other conditions.

    ## Methods:
    - :func:`~NodeConstraint.matches`: Checks whether a given node annotation satisfies all the specified conditions in this constraint.
    - :func:`~NodeConstraint.score_selectivity`: Calculates a heuristic selectivity score for this constraint, which can be used to prioritize more selective constraints during matching.
    - :func:`~NodeConstraint.describe`: Returns a human-readable explanation of this node constraint, including the role and the specified conditions.
    """

    role: str
    upostag_condition: Optional[ValueCondition] = None
    xpostag_condition: Optional[ValueCondition] = None
    lemma_condition: Optional[ValueCondition] = None
    deprel_at_node_condition: Optional[ValueCondition] = None
    feats_condition: Optional[FeatureCondition] = None
    extra_predicates: Optional[Tuple[NodePredicate, ...]] = None

    def __post_init__(self: Self) -> None:
        """
        Validate config and pre-normalise expected values once.
        """
        self._validate_or_raise()

    def matches(self: Self, node_annotation: estnltk.Span) -> bool:
        """
        Check whether the given node annotation satisfies this constraint.

        Args:
            node_annotation (estnltk.Span): The estnltk Span annotation of the node to check against this constraint.

        Returns:
            bool: True if the node annotation satisfies all specified conditions,
            otherwise False. Conditions that are None are ignored.
        """
        if self.upostag_condition and not self.upostag_condition.matches(
            node_annotation.upostag
        ):
            return False
        if self.xpostag_condition and not self.xpostag_condition.matches(
            node_annotation.xpostag
        ):
            return False
        if self.lemma_condition and not self.lemma_condition.matches(
            node_annotation.lemma
        ):
            return False
        if self.deprel_at_node_condition:
            deprel = getattr(node_annotation, "deprel", None)
            if not self.deprel_at_node_condition.matches(deprel):
                return False
        if self.feats_condition:
            feats = getattr(node_annotation, "feats", None)
            if not self.feats_condition.matches(feats):
                return False
        if self.extra_predicates:
            for pred in self.extra_predicates:
                if not pred(node_annotation):
                    return False
        return True

    def score_selectivity(self: Self) -> float:
        """
        Calculate a heuristic selectivity score for this constraint, which can be used to prioritize more selective constraints during matching.

        Returns:
            float: A selectivity score where higher values indicate more selective constraints. The score is calculated based on the number and restrictiveness of the specified conditions. For example, an EXACT ValueCondition is more selective than a NEGATION, and both are more selective than a WILDCARD. Similarly, having multiple conditions (e.g., UPOS, lemma, feats) increases selectivity compared to having only one or none.
        """
        score = 0.0
        # Exact > Negation > Wildcard(0.0) in terms of selectivity
        for cond in [
            self.upostag_condition,
            self.xpostag_condition,
            self.lemma_condition,
            self.deprel_at_node_condition,
        ]:
            if cond is not None:
                if cond.mode == ConditionMode.EXACT:  # Exact match is most selective
                    score += 1.0
                elif (
                    cond.mode == ConditionMode.NEGATION
                ):  # Negation is less selective than exact but more than wildcard
                    score += 0.5
        if self.feats_condition is not None:
            if (
                self.feats_condition.mode == ConditionMode.EXACT
            ):  # Exact match on features is very selective due to combinatorial nature of features
                score += 1.0
            elif (
                self.feats_condition.mode == ConditionMode.NEGATION
            ):  # Negation on features is less selective than exact but still adds some selectivity
                score += 0.5
        if self.extra_predicates:
            score += 0.5 * len(
                self.extra_predicates
            )  # Each extra predicate adds to selectivity

        return score

    def describe(self: Self) -> str:
        """
        Return a human-readable explanation of this node constraint, including the role and the specified conditions.

        Returns:
            str: A human-readable string describing this node constraint, including the role and the details of each specified condition. This can be used for debugging or for explaining why a particular node did or did not match this constraint.
        """
        parts = [f"Role: {self.role}"]
        if self.upostag_condition:
            parts.append(f"UPOS: {self.upostag_condition.describe()}")
        if self.xpostag_condition:
            parts.append(f"XPOS: {self.xpostag_condition.describe()}")
        if self.lemma_condition:
            parts.append(f"Lemma: {self.lemma_condition.describe()}")
        if self.deprel_at_node_condition:
            parts.append(f"Deprel at node: {self.deprel_at_node_condition.describe()}")
        if self.feats_condition:
            parts.append(f"Feats: {self.feats_condition.describe()}")
        if self.extra_predicates:
            parts.append(
                f"Extra predicates: {len(self.extra_predicates)} predicates defined"
            )
        return "; ".join(parts)

    def _validate_or_raise(self: Self) -> None:
        """
        Validate constructor arguments with explicit, actionable errors.
        """
        if not isinstance(self.role, str) or self.role.strip() == "":
            raise TypeError("role must be a non-empty string.")

        if self.upostag_condition is not None and not isinstance(
            self.upostag_condition, ValueCondition
        ):
            raise TypeError("upostag_condition must be ValueCondition or None.")

        if self.xpostag_condition is not None and not isinstance(
            self.xpostag_condition, ValueCondition
        ):
            raise TypeError("xpostag_condition must be ValueCondition or None.")

        if self.lemma_condition is not None and not isinstance(
            self.lemma_condition, ValueCondition
        ):
            raise TypeError("lemma_condition must be ValueCondition or None.")

        if self.deprel_at_node_condition is not None and not isinstance(
            self.deprel_at_node_condition, ValueCondition
        ):
            raise TypeError("deprel_at_node_condition must be ValueCondition or None.")

        if self.feats_condition is not None and not isinstance(
            self.feats_condition, FeatureCondition
        ):
            raise TypeError("feats_condition must be FeatureCondition or None.")

        if self.extra_predicates is not None:
            if not isinstance(self.extra_predicates, tuple):
                raise TypeError(
                    "extra_predicates must be a tuple of callables or None."
                )
            for pred in self.extra_predicates:
                if not callable(pred):
                    raise TypeError("Each item in extra_predicates must be callable.")


@dataclass(frozen=True, slots=True)
class EdgeConstraint:
    """
    A constraint for filtering edges in the syntax graph based on their properties.

    ## Attributes:
    - **direction** (`DirectionMode`): The direction of the edge to consider (up, down, or both).
    - **deprel_condition** (`Optional[ValueCondition]`): An optional ValueCondition to match the dependency relation (deprel) of the edge.
    - **min_hops** (`Optional[int]`): The minimum number of hops (edges) to traverse in the specified direction for this constraint to apply. Defaults to 1.
    - **max_hops** (`Optional[int]`): The maximum number of hops (edges) to traverse in the specified direction for this constraint to apply. Defaults to 1.
    - **allow_crossing_sentence** (`bool`): Whether to allow traversing edges that cross sentence boundaries (i.e., edges that connect nodes from different sentences). Defaults to False.

    ## Methods:
    - :func:`~EdgeConstraint.matches`: Checks whether a given edge context satisfies this constraint.
    - :func:`~EdgeConstraint.describe`: Returns a human-readable explanation of this edge constraint, including the direction, deprel condition, hop range, and other settings.
    """

    direction: DirectionMode
    deprel_condition: Optional[ValueCondition] = None
    min_hops: Optional[int] = 1
    max_hops: Optional[int] = 1
    allow_crossing_sentence: bool = False

    def __post_init__(self: Self) -> None:
        """
        Validate config once.
        """
        self._validate_or_raise()

    def matches(self: Self, edge_context: EdgeContext) -> bool:
        """
        Check whether the given edge context satisfies this constraint.

        Args:
            edge_context (EdgeContext): The context of the edge to check against this constraint, including its direction, deprel, hop count, and whether it crosses sentence boundaries.

        Returns:
            bool: True if the edge context satisfies this constraint, False otherwise. If `deprel_condition` is specified, the edge's deprel must match it. The edge's direction must match the specified direction. The number of hops must be within the specified min and max range. If `allow_crossing_sentence` is False, edges that cross sentence boundaries will not satisfy the constraint.
        """
        # Check deprel condition
        if self.deprel_condition and not self.deprel_condition.matches(
            edge_context.deprel
        ):
            return False
        # Check direction
        # If BOTH, we allow any direction, so no check needed. Otherwise, the edge's direction must match the specified direction.
        if (
            self.direction != DirectionMode.BOTH
            and edge_context.direction != self.direction
        ):
            return False
        # Check hop bounds
        if self.min_hops is not None and edge_context.hops < self.min_hops:
            return False
        if self.max_hops is not None and edge_context.hops > self.max_hops:
            return False
        # Check sentence boundary crossing
        if not self.allow_crossing_sentence and edge_context.crosses_sentence:
            return False
        return True

    def describe(self: Self) -> str:
        """
        Return a human-readable explanation of this edge constraint, including the direction, deprel condition, hop range, and other settings.

        Returns:
            str: A human-readable string describing this edge constraint, including the direction, deprel condition, hop range, and other settings. This can be used for debugging or for explaining why a particular edge did or did not match this constraint.
        """
        parts = [f"Direction: {self.direction.value}"]
        if self.deprel_condition:
            parts.append(f"Deprel: {self.deprel_condition.describe()}")
        if self.min_hops is not None or self.max_hops is not None:
            parts.append(
                f"Hops: {self.min_hops or 0} to {self.max_hops or '∞'}"
            )  # Display 0 when min_hops is None for clarity
        parts.append(f"Allow crossing sentence: {self.allow_crossing_sentence}")
        return "; ".join(parts)

    def _validate_or_raise(self: Self) -> None:
        """
        Validate constructor arguments with explicit, actionable errors.
        """
        if not isinstance(self.direction, DirectionMode):
            raise TypeError("direction must be an instance of DirectionMode.")

        if self.deprel_condition is not None and not isinstance(
            self.deprel_condition, ValueCondition
        ):
            raise TypeError("deprel_condition must be a ValueCondition or None.")

        if self.min_hops is not None:
            if not isinstance(self.min_hops, int) or self.min_hops < 0:
                raise ValueError("min_hops must be a non-negative integer or None.")

        if self.max_hops is not None:
            if not isinstance(self.max_hops, int) or self.max_hops < 0:
                raise ValueError("max_hops must be a non-negative integer or None.")

        if (
            self.min_hops is not None
            and self.max_hops is not None
            and self.min_hops > self.max_hops
        ):
            raise ValueError("min_hops cannot be greater than max_hops.")

        if not isinstance(self.allow_crossing_sentence, bool):
            raise TypeError("allow_crossing_sentence must be a boolean value.")


@dataclass(frozen=True, slots=True)
class PathPattern:
    """
    Represents a pattern for matching paths in the dependency tree.

    ## Attributes:
    - **name** (`str`): A unique name for this path pattern, used for identification and debugging purposes.
    - **node_steps** (`Tuple[NodeConstraint, ...]`): A tuple of NodeConstraint objects that specify the constraints for each node along the path. The first NodeConstraint corresponds to the starting node (e.g., the "self" node), and subsequent NodeConstraints correspond to nodes reached by traversing edges according to the specified EdgeConstraints.
    - **edge_steps** (`Tuple[EdgeConstraint, ...]`): A tuple of EdgeConstraint objects that specify the constraints for the edges to traverse between the nodes specified in `node_steps`. The length of `edge_steps` should be one less than the length of `node_steps`, as each edge connects two nodes.
    - **anchor_role** (`str`): The role of the anchor node in this path pattern, which serves as the reference point for the path. This is typically the role of the first NodeConstraint in `node_steps`, but it can be specified separately for clarity.
    - **emit_roles** (`Tuple[str, ...]`): A tuple of roles corresponding to the nodes in `node_steps` that should be included in the emitted features when this pattern matches. This allows for selective feature extraction based on the roles of the nodes in the matched path.

    ## Methods:
    - :func:`~PathPattern.get_node_constraint`: Retrieves the NodeConstraint associated with a given role in this path pattern.
    - :func:`~PathPattern.get_edge_constraint`: Retrieves the EdgeConstraint associated with the edge connecting two specified roles in this path pattern.
    - :func:`~PathPattern.describe`: Returns a human-readable explanation of this path pattern, including the name, node constraints, edge constraints, anchor role, and emit roles.
    """

    name: str
    node_steps: Tuple[NodeConstraint, ...]
    edge_steps: Tuple[EdgeConstraint, ...]
    anchor_role: str
    emit_roles: Tuple[str, ...]

    def __post_init__(self: Self) -> None:
        """
        Validate config once.
        """
        self._validate_or_raise()

    def get_node_constraint(self: Self, role: str) -> Optional[NodeConstraint]:
        """
        Retrieve the NodeConstraint associated with a given role in this path pattern.

        Args:
            role (str): The role of the node constraint to retrieve (e.g., "self", "parent", "child", etc.).

        Returns:
            Optional[NodeConstraint]: The NodeConstraint object associated with the specified role, or None if no such role exists in this pattern. This allows for easy access to the constraints for specific roles when processing matched paths.
        """
        for nc in self.node_steps:
            if nc.role == role:
                return nc
        return None

    def get_edge_constraint(
        self: Self, from_role: str, to_role: str
    ) -> Optional[EdgeConstraint]:
        """
        Retrieve the EdgeConstraint associated with the edge connecting two specified roles in this path pattern.

        Args:
            from_role (str): The role of the starting node of the edge (e.g., "self", "parent", "child", etc.).
            to_role (str): The role of the ending node of the edge (e.g., "self", "parent", "child", etc.).

        Returns:
            Optional[EdgeConstraint]: The EdgeConstraint object associated with the edge connecting the specified roles, or None if no such edge exists in this pattern. This allows for easy access to the constraints for specific edges when processing matched paths.
        """
        for i in range(len(self.edge_steps)):
            if (
                self.node_steps[i].role == from_role
                and self.node_steps[i + 1].role == to_role
            ):
                return self.edge_steps[i]
        return None

    def describe(self: Self) -> str:
        """
        Return a human-readable explanation of this path pattern, including the name, node constraints, edge constraints, anchor role, and emit roles.

        Returns:
            str: A human-readable string describing this path pattern, including the name, node constraints, edge constraints, anchor role, and emit roles. This can be used for debugging or for explaining what this pattern is designed to match.
        """
        parts = [f"Pattern name: {self.name}"]
        for i in range(len(self.node_steps)):
            parts.append(f"Node step {i}: {self.node_steps[i].describe()}")
            if i < len(self.edge_steps):
                parts.append(f"Edge step {i}: {self.edge_steps[i].describe()}")
        parts.append(f"Anchor role: {self.anchor_role}")
        parts.append(f"Emit roles: {', '.join(self.emit_roles)}")
        return "\n".join(parts)

    def _validate_or_raise(self: Self) -> None:
        """
        Validate constructor arguments with explicit, actionable errors.
        """
        if not isinstance(self.name, str) or self.name.strip() == "":
            raise TypeError("name must be a non-empty string.")

        if not isinstance(self.node_steps, tuple) or not all(
            isinstance(nc, NodeConstraint) for nc in self.node_steps
        ):
            raise TypeError("node_steps must be a tuple of NodeConstraint objects.")

        if not isinstance(self.edge_steps, tuple) or not all(
            isinstance(ec, EdgeConstraint) for ec in self.edge_steps
        ):
            raise TypeError("edge_steps must be a tuple of EdgeConstraint objects.")

        if len(self.edge_steps) != len(self.node_steps) - 1:
            raise ValueError(
                "Length of edge_steps must be one less than length of node_steps."
            )

        if not isinstance(self.anchor_role, str) or self.anchor_role.strip() == "":
            raise TypeError("anchor_role must be a non-empty string.")

        roles = [nc.role for nc in self.node_steps]
        valid_roles = {nc.role for nc in self.node_steps}
        if self.anchor_role not in valid_roles:
            raise ValueError(
                f"anchor_role '{self.anchor_role}' must match one of the roles defined in node_steps: {valid_roles}"
            )

        if len(valid_roles) != len(roles):
            raise ValueError("Roles defined in node_steps must be unique.")

        if not isinstance(self.emit_roles, tuple) or not all(
            isinstance(role, str) and role.strip() != "" for role in self.emit_roles
        ):
            raise TypeError("emit_roles must be a tuple of non-empty strings.")

        emit_roles_list = list(self.emit_roles)
        emit_roles_set = set(self.emit_roles)
        if not emit_roles_set.issubset(valid_roles):
            raise ValueError(
                f"All emit_roles must match roles defined in node_steps. Valid roles: {valid_roles}, emit_roles: {emit_roles_set}"
            )

        if len(emit_roles_list) != len(emit_roles_set):
            raise ValueError("emit_roles must not contain duplicate roles.")


@dataclass(frozen=True, slots=True)
class ChainMatch:
    """
    Represents a successful match of a PathPattern against a specific instance in the data, including the matched nodes, traversed edges, and any extracted features.
    ## Attributes:
    - **pattern_name** (`str`): The name of the PathPattern that was matched.
    - **sentence_index** (`int`): The index of the sentence in which the match was found.
    - **sentence_span** (`Tuple[int, int]`): The character span (start, end) of the entire sentence in the original text.
    - **role_to_token_id** (`Dict[str, int]`): A dictionary mapping each role defined in the PathPattern to the token ID of the node that was matched for that role in this instance.
    - **role_to_node** (`Dict[str, estnltk.Span]`): A dictionary mapping each role defined in the PathPattern to the actual estnltk Span annotation of the node that was matched for that role in this instance. This allows for access to all the properties of the matched nodes, not just their token IDs.
    - **traversed_edges** (`Tuple[Tuple[str, str, EdgeContext], ...]`): A tuple of tuples representing the edges that were traversed to match this pattern. Each inner tuple contains the from_role, to_role, and the EdgeContext of the traversed edge. This provides a detailed record of how the pattern was matched in terms of the dependency graph traversal.
    - **matched_text** (`str`): The text of the entire matched path, which can be useful for debugging, analysis, or as part of the emitted features.
    - **metadata** (`Dict[str, Any]`): An optional dictionary for storing any additional metadata about this match that may be useful for downstream processing, debugging, or analysis. This can include things like confidence scores, timestamps, or any other relevant information that does not fit into the other fields.
    """

    pattern_name: str
    sentence_index: int
    sentence_span: Tuple[int, int]
    role_to_token_id: Dict[str, int]
    role_to_node: Dict[str, estnltk.Span]
    traversed_edges: Tuple[
        Tuple[str, str, EdgeContext], ...
    ]  # (from_role, to_role, edge_context) for each traversed edge
    matched_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self: Self) -> None:
        """
        Validate config once.
        """
        self._validate_or_raise()

    def get_node(self: Self, role: str) -> estnltk.Span:
        """
        Get the estnltk Span annotation of the node matched for the given role.

        Args:
            role (str): The role of the node to retrieve (e.g., "self", "parent", "child", etc.).

        Returns:
            estnltk.Span: The estnltk Span annotation of the node matched for the given role. This allows for easy access to the properties of the matched nodes when processing the results of pattern matching.
        """
        if role not in self.role_to_node:
            raise ValueError(f"Role '{role}' not found in this match.")
        return self.role_to_node[role]

    def get_token_id(self: Self, role: str) -> int:
        """
        Get the token ID of the node matched for the given role.

        Args:
            role (str): The role of the node to retrieve (e.g., "self", "parent", "child", etc.).

        Returns:
            int: The token ID of the node matched for the given role. This allows for easy access to the token IDs of the matched nodes when processing the results of pattern matching.
        """
        if role not in self.role_to_token_id:
            raise ValueError(f"Role '{role}' not found in this match.")
        return self.role_to_token_id[role]

    def get_roles(self: Self) -> Set[str]:
        """
        Get the set of roles that were matched in this ChainMatch.

        Returns:
            Set[str]: A set of roles that were matched in this ChainMatch. This can be useful for quickly checking which roles are present in the match without having to look at the individual node annotations.
        """
        return set(self.role_to_node.keys())

    def build_matched_text(self: Self, ordered_roles: Tuple[str, ...]) -> str:
        """
        Build the matched text for this ChainMatch based on the specified order of roles. This concatenates the text of the nodes corresponding to the given roles in the specified order.

        Args:
            ordered_roles (Tuple[str, ...]): A tuple specifying the order of roles to concatenate for building the matched text. The roles should correspond to those defined in the PathPattern and present in this ChainMatch.

        Returns:
            str: The concatenated text of the nodes corresponding to the specified roles in the given order. This can be used for generating a human-readable representation of the matched path or for use in emitted features.
        """
        parts: List[str] = []
        for role in ordered_roles:
            node = self.get_node(role)
            parts.append(
                getattr(node, "text", "")
            )  # Use empty string if 'text' attribute is missing
        return " ".join(parts).strip()

    def to_output_row(self: Self) -> Dict[str, Any]:
        """
        Convert this ChainMatch to a dictionary format suitable for output (e.g., for saving to JSON or CSV).

        Returns:
            Dict[str, Any]: A dictionary representation of this ChainMatch, including all relevant information such as pattern name, sentence index, sentence span, role-to-token ID mapping, role-to-node mapping (with node properties), traversed edges, matched text, and metadata. This can be used for exporting the results of pattern matching in a structured format.
        """
        return {
            "pattern_name": self.pattern_name,
            "sentence_index": self.sentence_index,
            "sentence_span": self.sentence_span,
            "role_to_token_id": self.role_to_token_id,
            "role_to_node": {
                role: {
                    "start": node.start,
                    "end": node.end,
                    "upostag": node.upostag,
                    "xpostag": node.xpostag,
                    "lemma": node.lemma,
                    "feats": node.feats,
                    # Include any other relevant properties of the node as needed
                }
                for role, node in self.role_to_node.items()
            },
            "traversed_edges": [
                {
                    "from_role": from_role,
                    "to_role": to_role,
                    "edge_context": {
                        "direction": edge_context.direction.value,
                        "deprel": edge_context.deprel,
                        "hops": edge_context.hops,
                        "crosses_sentence": edge_context.crosses_sentence,
                    },
                }
                for from_role, to_role, edge_context in self.traversed_edges
            ],
            "matched_text": self.matched_text,
            "metadata": self.metadata,
        }

    def describe(self: Self) -> str:
        """
        Return a human-readable explanation of this ChainMatch, including the pattern name, sentence index, sentence span, matched roles and their corresponding token IDs and node properties, traversed edges, matched text, and any metadata.

        Returns:
            str: A human-readable string describing this ChainMatch in detail. This can be used for debugging or for explaining what this match represents in terms of the pattern that was matched and the specific instance in the data.
        """
        parts = [f"Pattern name: {self.pattern_name}"]
        parts.append(f"Sentence index: {self.sentence_index}")
        parts.append(f"Sentence span: {self.sentence_span}")
        parts.append("Matched roles:")
        for role in self.get_roles():
            token_id = self.get_token_id(role)
            node = self.get_node(role)
            parts.append(
                f"  Role: {role}, Token ID: {token_id}, Node properties: {{start: {node.start}, end: {node.end}, upostag: {node.upostag}, xpostag: {node.xpostag}, lemma: {node.lemma}, feats: {node.feats}}}"
            )
        parts.append("Traversed edges:")
        for from_role, to_role, edge_context in self.traversed_edges:
            parts.append(
                f"  From role: {from_role} to role: {to_role}, Edge context: {{direction: {edge_context.direction.value}, deprel: {edge_context.deprel}, hops: {edge_context.hops}, crosses_sentence: {edge_context.crosses_sentence}}}"
            )
        parts.append(f"Matched text: '{self.matched_text}'")
        if self.metadata:
            parts.append(f"Metadata: {self.metadata}")
        return "\n".join(parts)

    def _validate_or_raise(self: Self) -> None:
        """
        Validate constructor arguments with explicit, actionable errors.
        """
        if not isinstance(self.pattern_name, str) or self.pattern_name.strip() == "":
            raise TypeError("pattern_name must be a non-empty string.")

        if not isinstance(self.sentence_index, int) or self.sentence_index < 0:
            raise ValueError("sentence_index must be a non-negative integer.")

        if (
            not isinstance(self.sentence_span, tuple)
            or len(self.sentence_span) != 2
            or not all(isinstance(i, int) and i >= 0 for i in self.sentence_span)
            or self.sentence_span[0] > self.sentence_span[1]
        ):
            raise ValueError(
                "sentence_span must be a tuple of two non-negative integers (start, end) with start <= end."
            )

        if not isinstance(self.role_to_token_id, dict) or not all(
            isinstance(k, str) and k.strip() != "" and isinstance(v, int) and v >= 0
            for k, v in self.role_to_token_id.items()
        ):
            raise TypeError(
                "role_to_token_id must be a dictionary mapping non-empty strings to non-negative integers."
            )

        if not isinstance(self.role_to_node, dict) or not all(
            isinstance(k, str) and k.strip() != "" and isinstance(v, estnltk.Span)
            for k, v in self.role_to_node.items()
        ):
            raise TypeError(
                "role_to_node must be a dictionary mapping non-empty strings to estnltk.Span objects."
            )

        if not isinstance(self.traversed_edges, tuple) or not all(
            isinstance(edge_info, tuple)
            and len(edge_info) == 3
            and isinstance(edge_info[0], str)
            and edge_info[0].strip() != ""
            and isinstance(edge_info[1], str)
            and edge_info[1].strip() != ""
            and isinstance(edge_info[2], EdgeContext)
            for edge_info in self.traversed_edges
        ):
            raise TypeError(
                "traversed_edges must be a tuple of tuples (from_role, to_role, edge_context) where from_role and to_role are non-empty strings and edge_context is an EdgeContext object."
            )

        if not isinstance(self.matched_text, str):
            raise TypeError("matched_text must be a string.")

        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

        token_roles = set(self.role_to_token_id.keys())
        node_roles = set(self.role_to_node.keys())
        if token_roles != node_roles:
            raise ValueError(
                f"Roles in role_to_token_id and role_to_node must match. token_roles: {token_roles}, node_roles: {node_roles}"
            )


@dataclass(slots=True)
class MatchCollector:
    """
    Collect and manage `ChainMatch` objects with optional deduplication.

    Attributes:
        matches (List[ChainMatch]): Insertion-ordered collection of accepted matches.
        dedup_mode (str): Deduplication strategy. Allowed values are
            "none", "exact", and "role_based".
        max_matches (int): Hard upper bound on how many matches can be stored.
    """

    matches: List[ChainMatch] = field(default_factory=list)
    dedup_mode: str = "none"
    max_matches: int = 100000

    def __post_init__(self: Self) -> None:
        """
        Validate collector configuration after initialisation.
        """
        self._validate_or_raise()

    def make_dedup_key(self: Self, match: ChainMatch) -> Tuple[Any, ...]:
        """
        Build a stable deduplication key for role-based deduplication.

        Args:
            match (ChainMatch): The match for which to build a key.

        Returns:
            Tuple[Any, ...]: A deterministic key representing the semantic identity
            of the match under role-based deduplication.
        """
        role_items = tuple(sorted(match.role_to_token_id.items()))
        return (match.pattern_name, match.sentence_index, role_items)

    def is_duplicate(self: Self, match: ChainMatch) -> bool:
        """
        Check whether `match` is already present according to current strategy.

        Args:
            match (ChainMatch): Candidate match to test.

        Returns:
            bool: True if duplicate, False otherwise.
        """
        if self.dedup_mode == "none":
            return False

        if self.dedup_mode == "exact":
            return match in self.matches

        if self.dedup_mode == "role_based":
            target_key = self.make_dedup_key(match)
            for existing_match in self.matches:
                if self.make_dedup_key(existing_match) == target_key:
                    return True
            return False

        # Defensive fallback; should be unreachable due to validation.
        raise ValueError(f"Unsupported dedup_mode: {self.dedup_mode}")

    def add(self: Self, match: ChainMatch) -> bool:
        """
        Add a match if capacity allows and deduplication does not reject it.

        Args:
            match (ChainMatch): Match candidate to add.

        Returns:
            bool: True if added, False if rejected.
        """
        if len(self.matches) >= self.max_matches:
            return False

        if self.is_duplicate(match):
            return False

        self.matches.append(match)
        return True

    def extend(self: Self, new_matches: Iterable[ChainMatch]) -> int:
        """
        Add multiple matches in sequence.

        Args:
            new_matches (Iterable[ChainMatch]): Iterable of match candidates.

        Returns:
            int: Number of matches successfully added.
        """
        added_count = 0
        for match in new_matches:
            if self.add(match):
                added_count += 1
        return added_count

    def count(self: Self) -> int:
        """
        Return the number of stored matches.

        Returns:
            int: Number of accepted matches currently stored.
        """
        return len(self.matches)

    def clear(self: Self) -> None:
        """
        Remove all collected matches.
        """
        self.matches.clear()

    def all(self: Self) -> List[ChainMatch]:
        """
        Return all stored matches in insertion order.

        Returns:
            List[ChainMatch]: Shallow copy of stored matches.
        """
        return list(self.matches)

    def summary(self: Self) -> Dict[str, int]:
        """
        Build a compact summary of collected matches.

        Returns:
            Dict[str, int]: Summary map containing total count and per-pattern counts.
        """
        summary_data: Dict[str, int] = {"total": len(self.matches)}
        for match in self.matches:
            key = f"pattern::{match.pattern_name}"
            summary_data[key] = summary_data.get(key, 0) + 1
        return summary_data

    def to_output_rows(self: Self) -> List[Dict[str, Any]]:
        """
        Convert all matches to output dictionaries.

        Returns:
            List[Dict[str, Any]]: Output rows suitable for saving/reporting.
        """
        return [match.to_output_row() for match in self.matches]

    def _validate_or_raise(self: Self) -> None:
        """
        Validate constructor arguments with explicit, actionable errors.
        """
        if self.dedup_mode not in {"none", "exact", "role_based"}:
            raise ValueError(
                "dedup_mode must be one of 'none', 'exact', or 'role_based'."
            )

        if not isinstance(self.max_matches, int) or self.max_matches <= 0:
            raise ValueError("max_matches must be a positive integer.")

        if not isinstance(self.matches, list) or not all(
            isinstance(match, ChainMatch) for match in self.matches
        ):
            raise TypeError("matches must be a list of ChainMatch objects.")


@dataclass(slots=True)
class DepChainMatcher:
    """
    Match dependency path patterns against one sentence-level syntax graph.

    This class performs the core graph-search phase of the pipeline:
    1. choose anchor-node candidates,
    2. expand the pattern step-by-step along dependency edges,
    3. materialise successful paths as `ChainMatch` objects,
    4. deduplicate/limit matches via `MatchCollector`.

    Attributes:
        patterns (Tuple[PathPattern, ...]): Patterns to evaluate.
        dedup_mode (str): Collector deduplication mode for sentence results.
            Allowed values are "none", "exact", and "role_based".
        max_matches_per_sentence (int): Upper bound for collected matches per
            sentence.
        allow_role_node_overlap (bool): If False, different roles must map to
            different token IDs within one match.
    """

    patterns: Tuple[PathPattern, ...]
    dedup_mode: str = "role_based"
    max_matches_per_sentence: int = 100000
    allow_role_node_overlap: bool = False

    def __post_init__(self: Self) -> None:
        """
        Validate matcher configuration after initialisation.
        """
        self._validate_or_raise()

    def match_sentence(
        self: Self,
        graph_index: SyntaxGraphIndex,
        sentence_index: int,
        sentence_span: Optional[Tuple[int, int]] = None,
    ) -> List[ChainMatch]:
        """
        Match all configured patterns against one sentence graph.

        Args:
            graph_index (SyntaxGraphIndex): Sentence-level dependency graph.
            sentence_index (int): Zero-based sentence index in the source text.
            sentence_span (Optional[Tuple[int, int]], optional): Sentence
                character span override. If None, uses graph metadata when
                available.

        Returns:
            List[ChainMatch]: Accepted matches in insertion order.
        """
        collector = MatchCollector(
            dedup_mode=self.dedup_mode,
            max_matches=self.max_matches_per_sentence,
        )

        for pattern in self.patterns:
            pattern_matches = self.match_pattern_in_sentence(
                pattern=pattern,
                graph_index=graph_index,
                sentence_index=sentence_index,
                sentence_span=sentence_span,
            )
            collector.extend(pattern_matches)

        return collector.all()

    def match_pattern_in_sentence(
        self: Self,
        pattern: PathPattern,
        graph_index: SyntaxGraphIndex,
        sentence_index: int,
        sentence_span: Optional[Tuple[int, int]] = None,
    ) -> List[ChainMatch]:
        """
        Match one path pattern against one sentence graph.

        Args:
            pattern (PathPattern): Pattern to match.
            graph_index (SyntaxGraphIndex): Sentence-level dependency graph.
            sentence_index (int): Zero-based sentence index in the source text.
            sentence_span (Optional[Tuple[int, int]], optional): Sentence
                character span override.

        Returns:
            List[ChainMatch]: All successful matches for this pattern.
        """
        anchor_index = self._get_anchor_index(pattern)
        anchor_constraint = pattern.node_steps[anchor_index]

        resolved_sentence_span = self._resolve_sentence_span(
            graph_index=graph_index,
            sentence_span=sentence_span,
        )

        matches: List[ChainMatch] = []
        for anchor_node in graph_index.iter_nodes():
            if not anchor_constraint.matches(anchor_node):
                continue

            initial_nodes: Dict[int, estnltk.Span] = {anchor_index: anchor_node}
            initial_edges: Dict[int, EdgeContext] = {}
            assignments = self._expand_assignments(
                pattern=pattern,
                graph_index=graph_index,
                assigned_nodes_by_index=initial_nodes,
                assigned_edge_by_index=initial_edges,
            )

            for assigned_nodes, assigned_edges in assignments:
                matches.append(
                    self._build_chain_match(
                        pattern=pattern,
                        sentence_index=sentence_index,
                        sentence_span=resolved_sentence_span,
                        assigned_nodes_by_index=assigned_nodes,
                        assigned_edge_by_index=assigned_edges,
                    )
                )

        return matches

    def _expand_assignments(
        self: Self,
        pattern: PathPattern,
        graph_index: SyntaxGraphIndex,
        assigned_nodes_by_index: Dict[int, estnltk.Span],
        assigned_edge_by_index: Dict[int, EdgeContext],
    ) -> List[Tuple[Dict[int, estnltk.Span], Dict[int, EdgeContext]]]:
        """
        Recursively expand a partial assignment until all pattern steps are set.

        Args:
            pattern (PathPattern): Pattern being matched.
            graph_index (SyntaxGraphIndex): Sentence-level dependency graph.
            assigned_nodes_by_index (Dict[int, estnltk.Span]): Current partial
                node assignment keyed by `node_steps` index.
            assigned_edge_by_index (Dict[int, EdgeContext]): Current partial
                edge assignment keyed by `edge_steps` index.

        Returns:
            List[Tuple[Dict[int, estnltk.Span], Dict[int, EdgeContext]]]:
            Completed assignments.
        """
        if len(assigned_nodes_by_index) == len(pattern.node_steps):
            return [(dict(assigned_nodes_by_index), dict(assigned_edge_by_index))]

        options = self._get_frontier_options(
            pattern=pattern,
            assigned_nodes_by_index=assigned_nodes_by_index,
        )
        if not options:
            return []

        # Picking the smallest index first keeps search deterministic.
        options.sort(key=lambda item: item[0])
        target_index, known_index, edge_index = options[0]

        node_constraint = pattern.node_steps[target_index]
        known_node = assigned_nodes_by_index[known_index]

        if target_index == known_index + 1:
            candidate_pairs = self._enumerate_from_node(
                graph_index=graph_index,
                source_node=known_node,
                edge_constraint=pattern.edge_steps[edge_index],
            )
        else:
            candidate_pairs = self._enumerate_sources_to_target(
                graph_index=graph_index,
                target_node=known_node,
                edge_constraint=pattern.edge_steps[edge_index],
            )

        completed: List[Tuple[Dict[int, estnltk.Span], Dict[int, EdgeContext]]] = []
        used_token_ids = {
            self._get_node_id(node) for node in assigned_nodes_by_index.values()
        }

        for candidate_node, edge_context in candidate_pairs:
            if not node_constraint.matches(candidate_node):
                continue

            # Enforce one token per role unless overlap is explicitly allowed.
            if (
                not self.allow_role_node_overlap
                and self._get_node_id(candidate_node) in used_token_ids
            ):
                continue

            next_nodes = dict(assigned_nodes_by_index)
            next_nodes[target_index] = candidate_node

            next_edges = dict(assigned_edge_by_index)
            next_edges[edge_index] = edge_context

            completed.extend(
                self._expand_assignments(
                    pattern=pattern,
                    graph_index=graph_index,
                    assigned_nodes_by_index=next_nodes,
                    assigned_edge_by_index=next_edges,
                )
            )

        return completed

    def _get_frontier_options(
        self: Self,
        pattern: PathPattern,
        assigned_nodes_by_index: Dict[int, estnltk.Span],
    ) -> List[Tuple[int, int, int]]:
        """
        Find one-step expansion options around currently assigned indices.

        Returns tuples in the form `(target_index, known_index, edge_index)`.
        """
        options: List[Tuple[int, int, int]] = []
        last_node_index = len(pattern.node_steps) - 1

        for known_index in assigned_nodes_by_index:
            right_index = known_index + 1
            if (
                right_index <= last_node_index
                and right_index not in assigned_nodes_by_index
            ):
                options.append((right_index, known_index, known_index))

            left_index = known_index - 1
            if left_index >= 0 and left_index not in assigned_nodes_by_index:
                options.append((left_index, known_index, left_index))

        return options

    def _enumerate_from_node(
        self: Self,
        graph_index: SyntaxGraphIndex,
        source_node: estnltk.Span,
        edge_constraint: EdgeConstraint,
    ) -> List[Tuple[estnltk.Span, EdgeContext]]:
        """
        Enumerate candidate target nodes reachable from `source_node`.

        Each candidate is returned together with the concrete `EdgeContext` that
        satisfied `edge_constraint`.
        """
        candidates: List[Tuple[estnltk.Span, EdgeContext]] = []
        min_hops, max_hops = self._resolve_hop_bounds(
            graph_index=graph_index,
            min_hops=edge_constraint.min_hops,
            max_hops=edge_constraint.max_hops,
        )

        for direction in self._directions_to_try(edge_constraint.direction):
            for hops in range(min_hops, max_hops + 1):
                for node, deprel in self._nodes_at_exact_hops(
                    graph_index=graph_index,
                    start_node=source_node,
                    direction=direction,
                    hops=hops,
                ):
                    edge_context = self._build_edge_context(
                        direction=direction,
                        deprel=deprel,
                        hops=hops,
                        crosses_sentence=False,
                    )
                    if edge_constraint.matches(edge_context):
                        candidates.append((node, edge_context))
        return candidates

    def _enumerate_sources_to_target(
        self: Self,
        graph_index: SyntaxGraphIndex,
        target_node: estnltk.Span,
        edge_constraint: EdgeConstraint,
    ) -> List[Tuple[estnltk.Span, EdgeContext]]:
        """
        Enumerate candidate source nodes that can reach `target_node`.

        This is used when filling pattern steps to the left of the anchor index.
        """
        candidates: List[Tuple[estnltk.Span, EdgeContext]] = []
        for source_node in graph_index.iter_nodes():
            for candidate_node, edge_context in self._enumerate_from_node(
                graph_index=graph_index,
                source_node=source_node,
                edge_constraint=edge_constraint,
            ):
                if self._get_node_id(candidate_node) == self._get_node_id(target_node):
                    candidates.append((source_node, edge_context))
        return candidates

    def _nodes_at_exact_hops(
        self: Self,
        graph_index: SyntaxGraphIndex,
        start_node: estnltk.Span,
        direction: DirectionMode,
        hops: int,
    ) -> List[Tuple[estnltk.Span, Optional[str]]]:
        """
        Return all nodes reachable from `start_node` at exactly `hops`.

        Returns:
            List[Tuple[estnltk.Span, Optional[str]]]: Tuples of
            `(reachable_node, last_edge_deprel)`.
        """
        if hops == 0:
            return [(start_node, None)]

        if direction == DirectionMode.UP:
            current_node = start_node
            last_deprel: Optional[str] = None
            for _ in range(hops):
                parent_node = graph_index.get_parent(self._get_node_id(current_node))
                if parent_node is None:
                    return []
                last_deprel = getattr(current_node, "deprel", None)
                current_node = parent_node
            return [(current_node, last_deprel)]

        if direction == DirectionMode.DOWN:
            results: List[Tuple[estnltk.Span, Optional[str]]] = []

            def _dfs_down(
                node: estnltk.Span,
                remaining_hops: int,
                last_deprel: Optional[str],
            ) -> None:
                if remaining_hops == 0:
                    results.append((node, last_deprel))
                    return

                for child_node in graph_index.get_children(self._get_node_id(node)):
                    _dfs_down(
                        node=child_node,
                        remaining_hops=remaining_hops - 1,
                        last_deprel=getattr(child_node, "deprel", None),
                    )

            _dfs_down(start_node, hops, None)
            return results

        raise ValueError(f"Unsupported direction: {direction}")

    def _resolve_hop_bounds(
        self: Self,
        graph_index: SyntaxGraphIndex,
        min_hops: Optional[int],
        max_hops: Optional[int],
    ) -> Tuple[int, int]:
        """
        Resolve hop bounds into concrete finite integers for traversal.

        If max bound is unbounded, it is capped by sentence token count.
        """
        lower = 0 if min_hops is None else min_hops
        sentence_size = max(0, len(graph_index.token_order))
        upper = sentence_size if max_hops is None else max_hops

        if lower > upper:
            return (1, 0)
        return (lower, upper)

    def _directions_to_try(
        self: Self, direction: DirectionMode
    ) -> Tuple[DirectionMode, ...]:
        """
        Expand one configured direction into concrete search directions.
        """
        if direction == DirectionMode.BOTH:
            return (DirectionMode.UP, DirectionMode.DOWN)
        return (direction,)

    def _build_edge_context(
        self: Self,
        direction: DirectionMode,
        deprel: Optional[str],
        hops: int,
        crosses_sentence: bool,
    ) -> EdgeContext:
        """
        Create an `EdgeContext` instance for edge constraint checks.
        """
        edge_context = EdgeContext()
        edge_context.direction = direction
        edge_context.deprel = deprel
        edge_context.hops = hops
        edge_context.crosses_sentence = crosses_sentence
        return edge_context

    def _get_node_id(self: Self, node: estnltk.Span) -> int:
        """
        Read a node ID from estnltk span-like annotations as an integer.
        """
        return int(getattr(node, "id"))

    def _build_chain_match(
        self: Self,
        pattern: PathPattern,
        sentence_index: int,
        sentence_span: Tuple[int, int],
        assigned_nodes_by_index: Dict[int, estnltk.Span],
        assigned_edge_by_index: Dict[int, EdgeContext],
    ) -> ChainMatch:
        """
        Convert a completed assignment into one `ChainMatch` object.
        """
        role_to_node: Dict[str, estnltk.Span] = {}
        role_to_token_id: Dict[str, int] = {}

        for node_index, node_constraint in enumerate(pattern.node_steps):
            node = assigned_nodes_by_index[node_index]
            role_to_node[node_constraint.role] = node
            role_to_token_id[node_constraint.role] = self._get_node_id(node)

        traversed_edges: List[Tuple[str, str, EdgeContext]] = []
        for edge_index, edge_constraint in enumerate(pattern.edge_steps):
            from_role = pattern.node_steps[edge_index].role
            to_role = pattern.node_steps[edge_index + 1].role
            edge_context = assigned_edge_by_index[edge_index]
            if not edge_constraint.matches(edge_context):
                raise ValueError(
                    "Internal matcher error: assigned edge context no longer satisfies edge constraint."
                )
            traversed_edges.append((from_role, to_role, edge_context))

        matched_text = " ".join(
            getattr(role_to_node[role], "text", "") for role in pattern.emit_roles
        ).strip()

        return ChainMatch(
            pattern_name=pattern.name,
            sentence_index=sentence_index,
            sentence_span=sentence_span,
            role_to_token_id=role_to_token_id,
            role_to_node=role_to_node,
            traversed_edges=tuple(traversed_edges),
            matched_text=matched_text,
        )

    def _get_anchor_index(self: Self, pattern: PathPattern) -> int:
        """
        Find the index of the anchor role in `pattern.node_steps`.
        """
        for node_index, node_constraint in enumerate(pattern.node_steps):
            if node_constraint.role == pattern.anchor_role:
                return node_index
        raise ValueError(
            f"anchor_role '{pattern.anchor_role}' not found in pattern node_steps."
        )

    def _resolve_sentence_span(
        self: Self,
        graph_index: SyntaxGraphIndex,
        sentence_span: Optional[Tuple[int, int]],
    ) -> Tuple[int, int]:
        """
        Resolve sentence span from method input or graph metadata.
        """
        if sentence_span is not None:
            return sentence_span
        if graph_index.sentence_span is not None:
            return graph_index.sentence_span
        return (0, 0)

    def _validate_or_raise(self: Self) -> None:
        """
        Validate constructor arguments with explicit, actionable errors.
        """
        if not isinstance(self.patterns, tuple) or not all(
            isinstance(pattern, PathPattern) for pattern in self.patterns
        ):
            raise TypeError("patterns must be a tuple of PathPattern objects.")

        if self.dedup_mode not in {"none", "exact", "role_based"}:
            raise ValueError(
                "dedup_mode must be one of 'none', 'exact', or 'role_based'."
            )

        if (
            not isinstance(self.max_matches_per_sentence, int)
            or self.max_matches_per_sentence <= 0
        ):
            raise ValueError("max_matches_per_sentence must be a positive integer.")

        if not isinstance(self.allow_role_node_overlap, bool):
            raise TypeError("allow_role_node_overlap must be a boolean value.")


@dataclass(slots=True)
class PhraseDecorator:
    """
    Transform `ChainMatch` objects into output rows for downstream layers.

    The decorator is intentionally independent of traversal logic: it only
    handles output shaping/serialisation. This keeps the matcher focused on
    search and keeps output schema changes local to one class.

    Attributes:
        include_pattern_name (bool): Include `pattern_name` field.
        include_sentence_context (bool): Include sentence-level fields
            (`sentence_index`, `sentence_span`).
        include_role_token_ids (bool): Include `role_to_token_id` mapping.
        include_role_texts (bool): Include role-wise node text mapping.
        include_role_spans (bool): Include role-wise `(start, end)` spans.
        include_traversed_edges (bool): Include serialised traversed edges.
        include_metadata (bool): Include copied `metadata` dictionary.
        output_text_roles (Optional[Tuple[str, ...]]): If provided, rebuild
            `matched_text` using this role order. If None, keep existing text.
        output_field_prefix (str): Optional prefix for all top-level keys.
    """

    include_pattern_name: bool = True
    include_sentence_context: bool = True
    include_role_token_ids: bool = True
    include_role_texts: bool = True
    include_role_spans: bool = True
    include_traversed_edges: bool = True
    include_metadata: bool = True
    output_text_roles: Optional[Tuple[str, ...]] = None
    output_field_prefix: str = ""

    def __post_init__(self: Self) -> None:
        """
        Validate decorator configuration after initialisation.
        """
        self._validate_or_raise()

    def decorate_match(self: Self, match: ChainMatch) -> Dict[str, Any]:
        """
        Convert one `ChainMatch` into one output dictionary.

        Args:
            match (ChainMatch): Match object to decorate.

        Returns:
            Dict[str, Any]: Decorated output row.
        """
        row: Dict[str, Any] = {}

        if self.include_pattern_name:
            row[self._field_name("pattern_name")] = match.pattern_name

        if self.include_sentence_context:
            row[self._field_name("sentence_index")] = match.sentence_index
            row[self._field_name("sentence_span")] = match.sentence_span

        # Matched text is always useful as a human-readable surface form.
        if self.output_text_roles is None:
            matched_text = match.matched_text
        else:
            matched_text = match.build_matched_text(self.output_text_roles)
        row[self._field_name("matched_text")] = matched_text

        if self.include_role_token_ids:
            row[self._field_name("role_to_token_id")] = dict(match.role_to_token_id)

        if self.include_role_texts:
            row[self._field_name("role_to_text")] = self._build_role_to_text(match)

        if self.include_role_spans:
            row[self._field_name("role_to_span")] = self._build_role_to_span(match)

        if self.include_traversed_edges:
            row[self._field_name("traversed_edges")] = self._serialize_edges(match)

        if self.include_metadata:
            # Copy to avoid accidental mutation from downstream steps.
            row[self._field_name("metadata")] = dict(match.metadata)

        return row

    def decorate_matches(
        self: Self,
        matches: Iterable[ChainMatch],
    ) -> List[Dict[str, Any]]:
        """
        Decorate many matches preserving insertion order.

        Args:
            matches (Iterable[ChainMatch]): Matches to decorate.

        Returns:
            List[Dict[str, Any]]: List of decorated rows.
        """
        return [self.decorate_match(match) for match in matches]

    def decorate_collector(
        self: Self, collector: MatchCollector
    ) -> List[Dict[str, Any]]:
        """
        Decorate all matches currently stored in a `MatchCollector`.

        Args:
            collector (MatchCollector): Source collector.

        Returns:
            List[Dict[str, Any]]: Decorated rows from collector contents.
        """
        return self.decorate_matches(collector.all())

    def _field_name(self: Self, base_name: str) -> str:
        """
        Build one top-level output key from optional prefix and base name.
        """
        if self.output_field_prefix == "":
            return base_name
        return f"{self.output_field_prefix}{base_name}"

    def _build_role_to_text(self: Self, match: ChainMatch) -> Dict[str, str]:
        """
        Build role-to-text mapping for one match.
        """
        role_to_text: Dict[str, str] = {}
        for role, node in match.role_to_node.items():
            role_to_text[role] = str(getattr(node, "text", ""))
        return role_to_text

    def _build_role_to_span(
        self: Self,
        match: ChainMatch,
    ) -> Dict[str, Tuple[Optional[int], Optional[int]]]:
        """
        Build role-to-(start, end) mapping for one match.
        """
        role_to_span: Dict[str, Tuple[Optional[int], Optional[int]]] = {}
        for role, node in match.role_to_node.items():
            role_to_span[role] = (
                getattr(node, "start", None),
                getattr(node, "end", None),
            )
        return role_to_span

    def _serialize_edges(self: Self, match: ChainMatch) -> List[Dict[str, Any]]:
        """
        Serialize traversed edges into JSON-friendly dictionaries.
        """
        rows: List[Dict[str, Any]] = []
        for from_role, to_role, edge_context in match.traversed_edges:
            rows.append(
                {
                    "from_role": from_role,
                    "to_role": to_role,
                    "direction": edge_context.direction.value,
                    "deprel": edge_context.deprel,
                    "hops": edge_context.hops,
                    "crosses_sentence": edge_context.crosses_sentence,
                }
            )
        return rows

    def _validate_or_raise(self: Self) -> None:
        """
        Validate constructor arguments with explicit, actionable errors.
        """
        for field_name, field_value in [
            ("include_pattern_name", self.include_pattern_name),
            ("include_sentence_context", self.include_sentence_context),
            ("include_role_token_ids", self.include_role_token_ids),
            ("include_role_texts", self.include_role_texts),
            ("include_role_spans", self.include_role_spans),
            ("include_traversed_edges", self.include_traversed_edges),
            ("include_metadata", self.include_metadata),
        ]:
            if not isinstance(field_value, bool):
                raise TypeError(f"{field_name} must be a boolean value.")

        if self.output_text_roles is not None:
            if not isinstance(self.output_text_roles, tuple) or not all(
                isinstance(role, str) and role.strip() != ""
                for role in self.output_text_roles
            ):
                raise TypeError(
                    "output_text_roles must be a tuple of non-empty strings or None."
                )

        if not isinstance(self.output_field_prefix, str):
            raise TypeError("output_field_prefix must be a string.")


@dataclass(slots=True)
class DepChainTaggerOrchestrator:
    """
    End-to-end orchestrator for dependency-chain tagging on sentence layers.

    The class owns the high-level pipeline:
    1. build sentence-level `SyntaxGraphIndex` objects,
    2. run `DepChainMatcher` on each sentence,
    3. optionally deduplicate across all sentence matches,
    4. transform matches into relational output rows with `PhraseDecorator`.

    Attributes:
        patterns (Tuple[PathPattern, ...]): Pattern set used by matcher.
        matcher (Optional[DepChainMatcher]): Optional preconfigured matcher.
            If None, a matcher is built from this tagger's config.
        decorator (Optional[PhraseDecorator]): Optional output decorator.
            If None, a default `PhraseDecorator` is created.
        sentence_match_dedup_mode (str): Matcher-level deduplication mode
            applied within each sentence.
        max_matches_per_sentence (int): Per-sentence upper bound.
        allow_role_node_overlap (bool): Passed through to matcher.
        global_dedup_mode (str): Optional second-stage deduplication across
            all collected matches.
        max_total_matches (int): Global cap across all sentences.
    """

    patterns: Tuple[PathPattern, ...]
    matcher: Optional[DepChainMatcher] = None
    decorator: Optional[PhraseDecorator] = None
    sentence_match_dedup_mode: str = "role_based"
    max_matches_per_sentence: int = 100000
    allow_role_node_overlap: bool = False
    global_dedup_mode: str = "none"
    max_total_matches: int = 1000000

    def __post_init__(self: Self) -> None:
        """
        Validate configuration and construct default components when needed.
        """
        self._validate_or_raise()

        if self.matcher is None:
            self.matcher = DepChainMatcher(
                patterns=self.patterns,
                dedup_mode=self.sentence_match_dedup_mode,
                max_matches_per_sentence=self.max_matches_per_sentence,
                allow_role_node_overlap=self.allow_role_node_overlap,
            )

        if self.decorator is None:
            self.decorator = PhraseDecorator()

    def tag_sentence_layers(
        self: Self,
        sentence_syntax_layers: Iterable[estnltk.Layer],
        sentence_spans: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> List[ChainMatch]:
        """
        Run dependency-chain matching over sentence-level syntax layers.

        Args:
            sentence_syntax_layers (Iterable[estnltk.Layer]): Iterable of
                per-sentence syntax layers (e.g., split `stanza_syntax`).
            sentence_spans (Optional[Iterable[Tuple[int, int]]], optional):
                Optional sentence character spans aligned with input order.

        Returns:
            List[ChainMatch]: Accepted matches after optional global dedup.
        """
        layers = list(sentence_syntax_layers)
        spans = list(sentence_spans) if sentence_spans is not None else None

        if spans is not None and len(spans) != len(layers):
            raise ValueError(
                "sentence_spans length must match sentence_syntax_layers length."
            )

        global_collector = MatchCollector(
            dedup_mode=self.global_dedup_mode,
            max_matches=self.max_total_matches,
        )
        matcher = self._get_matcher()

        for sentence_index, sentence_layer in enumerate(layers):
            sentence_span = spans[sentence_index] if spans is not None else None

            # Build a sentence-local graph index and match patterns in isolation.
            graph_index = SyntaxGraphIndex(
                stanza_syntax_layer=sentence_layer,
                sentence_id=sentence_index,
                sentence_span=sentence_span,
            )
            sentence_matches = matcher.match_sentence(
                graph_index=graph_index,
                sentence_index=sentence_index,
                sentence_span=sentence_span,
            )

            global_collector.extend(sentence_matches)
            if global_collector.count() >= self.max_total_matches:
                break

        return global_collector.all()

    def tag_sentence_layer(
        self: Self,
        sentence_syntax_layer: estnltk.Layer,
        sentence_index: int = 0,
        sentence_span: Optional[Tuple[int, int]] = None,
    ) -> List[ChainMatch]:
        """
        Convenience wrapper for matching one sentence syntax layer.

        Args:
            sentence_syntax_layer (estnltk.Layer): One sentence syntax layer.
            sentence_index (int, optional): Sentence index to stamp into
                results. Defaults to 0.
            sentence_span (Optional[Tuple[int, int]], optional): Optional
                sentence span override.

        Returns:
            List[ChainMatch]: Matches found in the given sentence.
        """
        graph_index = SyntaxGraphIndex(
            stanza_syntax_layer=sentence_syntax_layer,
            sentence_id=sentence_index,
            sentence_span=sentence_span,
        )
        matcher = self._get_matcher()
        return matcher.match_sentence(
            graph_index=graph_index,
            sentence_index=sentence_index,
            sentence_span=sentence_span,
        )

    def decorate_matches(
        self: Self, matches: Iterable[ChainMatch]
    ) -> List[Dict[str, Any]]:
        """
        Decorate existing matches into relational output rows.

        Args:
            matches (Iterable[ChainMatch]): Match objects to decorate.

        Returns:
            List[Dict[str, Any]]: Decorated rows ready for output layers/files.
        """
        decorator = self._get_decorator()
        return decorator.decorate_matches(matches)

    def tag_and_decorate_sentence_layers(
        self: Self,
        sentence_syntax_layers: Iterable[estnltk.Layer],
        sentence_spans: Optional[Iterable[Tuple[int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run full pipeline and return decorated relational rows.

        Args:
            sentence_syntax_layers (Iterable[estnltk.Layer]): Iterable of
                per-sentence syntax layers.
            sentence_spans (Optional[Iterable[Tuple[int, int]]], optional):
                Optional aligned sentence spans.

        Returns:
            List[Dict[str, Any]]: Decorated output rows.
        """
        matches = self.tag_sentence_layers(
            sentence_syntax_layers=sentence_syntax_layers,
            sentence_spans=sentence_spans,
        )
        decorator = self._get_decorator()
        return decorator.decorate_matches(matches)

    def _get_matcher(self: Self) -> DepChainMatcher:
        """
        Return configured matcher as non-optional instance.
        """
        if self.matcher is None:
            raise RuntimeError("matcher is not initialised.")
        return self.matcher

    def _get_decorator(self: Self) -> PhraseDecorator:
        """
        Return configured decorator as non-optional instance.
        """
        if self.decorator is None:
            raise RuntimeError("decorator is not initialised.")
        return self.decorator

    def _validate_or_raise(self: Self) -> None:
        """
        Validate constructor arguments with explicit, actionable errors.
        """
        if not isinstance(self.patterns, tuple) or not all(
            isinstance(pattern, PathPattern) for pattern in self.patterns
        ):
            raise TypeError("patterns must be a tuple of PathPattern objects.")

        if self.matcher is not None and not isinstance(self.matcher, DepChainMatcher):
            raise TypeError("matcher must be a DepChainMatcher or None.")

        if self.decorator is not None and not isinstance(
            self.decorator, PhraseDecorator
        ):
            raise TypeError("decorator must be a PhraseDecorator or None.")

        if self.sentence_match_dedup_mode not in {"none", "exact", "role_based"}:
            raise ValueError(
                "sentence_match_dedup_mode must be one of 'none', 'exact', or 'role_based'."
            )

        if self.global_dedup_mode not in {"none", "exact", "role_based"}:
            raise ValueError(
                "global_dedup_mode must be one of 'none', 'exact', or 'role_based'."
            )

        if (
            not isinstance(self.max_matches_per_sentence, int)
            or self.max_matches_per_sentence <= 0
        ):
            raise ValueError("max_matches_per_sentence must be a positive integer.")

        if not isinstance(self.max_total_matches, int) or self.max_total_matches <= 0:
            raise ValueError("max_total_matches must be a positive integer.")

        if not isinstance(self.allow_role_node_overlap, bool):
            raise TypeError("allow_role_node_overlap must be a boolean value.")

        if self.matcher is not None and self.matcher.patterns != self.patterns:
            raise ValueError(
                "matcher.patterns must be the same as DepChainTagger.patterns."
            )


class DepChainTagger(Tagger):
    """
    EstNLTK-compatible wrapper for DepChainTagger.

    This class integrates the DepChainTagger into the EstNLTK pipeline by implementing
    the Tagger interface. It processes sentence syntax layers and produces a new layer
    containing dependency chain matches.

    Attributes:
        patterns (`Tuple[PathPattern, ...]`): Patterns to match.
        output_layer (`str`): Name of the output layer.
        output_attributes (`Tuple[str, ...]`): Attributes for the output layer.
        sentence_match_dedup_mode (`Literal["none", "exact", "role_based"]`): Deduplication strategy within sentences.
        max_matches_per_sentence (`int`): Maximum matches per sentence.
        allow_role_node_overlap (`bool`): Allow different roles to map to same token.
        global_dedup_mode (`Literal["none", "exact", "role_based"]`): Global deduplication across sentences.
        max_total_matches (`int`): Maximum total matches.
    """

    conf_param = [
        "patterns",
        "output_layer",
        "output_attributes",
        "sentence_match_dedup_mode",
        "max_matches_per_sentence",
        "allow_role_node_overlap",
        "global_dedup_mode",
        "max_total_matches",
        "_depchain_tagger",
        "_pattern_by_name",
    ]

    def __init__(
        self: Self,
        patterns: Tuple[PathPattern, ...],
        output_layer: str = "dep_chains",
        output_attributes: Optional[Tuple[str, ...]] = None,
        sentence_match_dedup_mode: Literal["none", "exact", "role_based"] = "none",
        max_matches_per_sentence: int = 100000,
        allow_role_node_overlap: bool = False,
        global_dedup_mode: Literal["none", "exact", "role_based"] = "none",
        max_total_matches: int = 1000000,
    ) -> None:
        """
        Initialize EstNLTK-compatible dependency chain tagger.

        Args:
            patterns (Tuple[PathPattern, ...]): Patterns to match in syntax graphs.
            output_layer (str, optional): Name of output layer. Defaults to "dep_chains".
            output_attributes (Optional[Tuple[str, ...]], optional): Output layer attributes.
                Defaults to ("pattern_name", "matched_text", "roles_info").
            sentence_match_dedup_mode (Literal["none", "exact", "role_based"], optional): Deduplication within sentences.
                Defaults to "none".
            max_matches_per_sentence (int, optional): Max matches per sentence.
                Defaults to 100000.
            allow_role_node_overlap (bool, optional): Allow role overlaps.
                Defaults to False.
            global_dedup_mode (Literal["none", "exact", "role_based"], optional): Global deduplication.
                Defaults to "none".
            max_total_matches (int, optional): Max total matches.
                Defaults to 1000000.

        Deduplication modes:
            - `"none"`: No deduplication, all matches are kept.
            - `"exact"`: Remove duplicate matches with identical role-to-token mappings.
            - `"role_based"`: Remove matches that share the same token for any role, allowing only one match per unique token assignment.
        """
        # EstNLTK Tagger interface setup
        self.input_layers = ("stanza_syntax", "sentences")
        self.output_layer = output_layer
        if output_attributes is None:
            self.output_attributes = (
                "pattern_name",
                "matched_text",
                "upostag",
                "xpostag",
                "feats",
                "lemma",
                "deprel",
                "role",
                "is_anchor",
                "match_id",
            )
        else:
            self.output_attributes = output_attributes
        # Initialize internal DepChainTaggerOrchestrator with provided configuration
        self._depchain_tagger = DepChainTaggerOrchestrator(
            patterns=patterns,
            sentence_match_dedup_mode=sentence_match_dedup_mode,
            max_matches_per_sentence=max_matches_per_sentence,
            allow_role_node_overlap=allow_role_node_overlap,
            global_dedup_mode=global_dedup_mode,
            max_total_matches=max_total_matches,
        )
        self._pattern_by_name: Dict[str, PathPattern] = {
            pattern.name: pattern for pattern in patterns
        }

    def _make_layer_template(self: Self) -> Layer:
        """
        Create empty layer template according to tagger configuration.

        Returns:
            Layer: Empty layer template with configured name and attributes.
        """
        return Layer(
            name=self.output_layer,
            attributes=self.output_attributes,
            text_object=None,
            ambiguous=False,
        )

    def _make_layer(
        self: Self,
        text: Text,
        layers: Any,
        status: Optional[Dict[str, Any]] = None,
    ) -> Layer:
        """
        Create and populate layer with dependency chain matches.

        This method runs the full DepChainTagger pipeline on the stanza_syntax layer
        and converts matches into Layer annotations using the anchor role strategy.

        Args:
            text (Text): Input text object.
            layers (Any): Dictionary of input layers. Must contain "stanza_syntax" layer.
            status (Optional[Dict[str, Any]], optional): Processing status tracker.
                Defaults to None.

        Returns:
            Layer: Populated layer with dependency chain match annotations.
        """
        # Create layer template
        layer = self._make_layer_template()
        layer.text_object = text

        # Validate input layer availability
        if layers is None or "stanza_syntax" not in layers or "sentences" not in layers:
            # Return empty layer if no syntax layer or sentences layer is available
            return layer

        sentences_layer = layers["sentences"]

        # Run matching on the sentence syntax layer
        try:
            for i, sentence in enumerate(sentences_layer):
                sentence_span = (sentence.start, sentence.end)
                sentence_syntax = sentence.stanza_syntax
                sentence_matches = self._run_depchain_matcher(
                    i, sentence_span, sentence_syntax
                )
                # Add matches to layer
                try:
                    for match in sentence_matches:
                        self._add_match_to_layer(layer, match, text)
                except Exception as exc:
                    if status is not None:
                        status.setdefault("errors", []).append(
                            f"Error adding matches to layer for sentence {i}: {str(exc)}"
                        )

        except Exception as exc:
            # Log error and return empty layer on matching failure
            if status is not None:
                status.setdefault("errors", []).append(
                    f"DepChainTagger matching failed: {str(exc)}"
                )
            return layer

        return layer

    def _run_depchain_matcher(
        self: Self,
        sentence_index: int,
        sentence_span: Tuple[int, int],
        stanza_syntax_layer: Layer,
    ) -> List[ChainMatch]:
        """
        Run dependency chain matching on syntax layer.

        Args:
            sentence_index (int): Index of the sentence being processed.
            sentence_span (Tuple[int, int]): Character span of the sentence.
            stanza_syntax_layer (Layer): The stanza syntax layer from estNLTK.

        Returns:
            List[ChainMatch]: List of matched dependency chains.

        Raises:
            ValueError: If stanza_syntax_layer is invalid or empty.
        """
        if stanza_syntax_layer is None or len(stanza_syntax_layer) == 0:
            return []

        # Build syntax graph index from stanza_syntax layer
        graph_index = SyntaxGraphIndex(
            stanza_syntax_layer=stanza_syntax_layer,
            sentence_id=sentence_index,
            sentence_span=sentence_span,
        )

        # Run matcher
        matcher = self._depchain_tagger._get_matcher()
        matches = matcher.match_sentence(
            graph_index=graph_index,
            sentence_index=sentence_index,
            sentence_span=sentence_span,
        )

        return matches

    def _add_match_to_layer(
        self: Self,
        layer: Layer,
        match: ChainMatch,
        text: Text,
    ) -> None:
        """
        Add a single ChainMatch as annotation to the layer.

        Uses the anchor role strategy: the first role (typically "self") becomes
        the span for the Layer annotation, while all role information is stored
        in a JSON-serialized attribute.

        Args:
            layer (Layer): Target layer to which annotation is added.
            match (ChainMatch): The match to convert.
            text (Text): Source text object.

        Raises:
            ValueError: If match has no roles or anchor role not found.
        """
        pattern = self._pattern_by_name.get(match.pattern_name)
        anchor_role = (
            pattern.anchor_role
            if pattern and pattern.anchor_role in match.role_to_node
            else None
        )
        if anchor_role is None:
            anchor_role = (
                "self"
                if "self" in match.role_to_node
                else next(iter(match.role_to_token_id.keys()))
            )

        # roles_info = self._build_roles_info(match)
        match_id = f"{match.sentence_index}:{match.pattern_name}:{hash(tuple(sorted(match.role_to_token_id.items())))}"

        for role, node in match.role_to_node.items():
            annotation_dict = {
                "pattern_name": match.pattern_name,
                "matched_text": match.matched_text,
                "upostag": getattr(node, "upostag", None),
                "xpostag": getattr(node, "xpostag", None),
                "feats": getattr(node, "feats", None),
                "lemma": getattr(node, "lemma", None),
                "deprel": getattr(node, "deprel", None),
                # "roles_info": json.dumps(roles_info[role]),
                "role": role,
                "is_anchor": role == anchor_role,
                "match_id": match_id,
            }
            layer.add_annotation((node.start, node.end), annotation_dict)

    def _build_roles_info(self: Self, match: ChainMatch) -> Dict[str, Any]:
        """
        Build a dictionary containing information about all matched roles.

        For each role in the match, extracts the most relevant linguistic properties
        from the matched Span annotation.

        Args:
            match (ChainMatch): The match object containing role-to-node mapping.

        Returns:
            Dict[str, Any]: Serializable dictionary with role information.
                Keys are role names, values are dicts containing:
                - text: surface text of the node
                - start: character start position
                - end: character end position
                - upostag: universal POS tag
                - xpostag: extended POS tag
                - feats: morphological features
                - lemma: lemmatized form
                - deprel: dependency relation
        """
        roles_info: Dict[str, Any] = {}
        for role, node in match.role_to_node.items():
            roles_info[role] = {
                "text": getattr(node, "text", ""),
                "start": node.start,
                "end": node.end,
                "upostag": getattr(node, "upostag", None),
                "xpostag": getattr(node, "xpostag", None),
                "feats": getattr(node, "feats", None),
                "lemma": getattr(node, "lemma", None),
                "deprel": getattr(node, "deprel", None),
            }
        return roles_info
