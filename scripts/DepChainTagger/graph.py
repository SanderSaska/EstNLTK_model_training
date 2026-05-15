import estnltk
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Iterable,
    Self,
    Any,
)

from .types import DirectionMode


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
        self: Self, direction: DirectionMode = DirectionMode.BOTH
    ) -> Iterable[Tuple[Optional[estnltk.Span], Optional[estnltk.Span], str]]:
        """
        Iterates over all edges in the graph, optionally filtering by direction (up, down, or both).

        Args:
            self (Self): The instance of the SyntaxGraphIndex being iterated over.
            direction (DirectionMode, optional): The direction of edges to iterate over. Can be DirectionMode.UP for parent-child edges, DirectionMode.DOWN for child-parent edges, or DirectionMode.BOTH for all edges. Defaults to DirectionMode.BOTH.

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
                if direction in [
                    DirectionMode.BOTH,
                    DirectionMode.UP,
                ]:  # Move from id to head (up the tree)
                    yield (node, parent_node, DirectionMode.UP.value)
                if direction in [
                    DirectionMode.BOTH,
                    DirectionMode.DOWN,
                ]:  # Move from head to id (down the tree)
                    if (
                        parent_id != 0
                    ):  # Skip the root node which has no parent (head = 0), so we don't yield a down edge for it
                        yield (parent_node, node, DirectionMode.DOWN.value)

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
