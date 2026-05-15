import estnltk
from typing import (
    Dict,
    Optional,
    Tuple,
    Self,
    Any,
    Callable,
)
from dataclasses import dataclass

from .types import ConditionMode, DirectionMode, NodePredicate, EdgeContext

from .config import (
    DEFAULT_MISSING_MARKERS,
    SELECTIVITY_WEIGHT_EXACT,
    SELECTIVITY_WEIGHT_NEGATION,
    SELECTIVITY_WEIGHT_EXTRA_PREDICATE,
)


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
    missing_markers: Tuple[Any, ...] = DEFAULT_MISSING_MARKERS

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
                    score += SELECTIVITY_WEIGHT_EXACT
                elif (
                    cond.mode == ConditionMode.NEGATION
                ):  # Negation is less selective than exact but more than wildcard
                    score += SELECTIVITY_WEIGHT_NEGATION
        if self.feats_condition is not None:
            if (
                self.feats_condition.mode == ConditionMode.EXACT
            ):  # Exact match on features is very selective due to combinatorial nature of features
                score += SELECTIVITY_WEIGHT_EXACT
            elif (
                self.feats_condition.mode == ConditionMode.NEGATION
            ):  # Negation on features is less selective than exact but still adds some selectivity
                score += SELECTIVITY_WEIGHT_NEGATION
        if self.extra_predicates:
            score += SELECTIVITY_WEIGHT_EXTRA_PREDICATE * len(
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
