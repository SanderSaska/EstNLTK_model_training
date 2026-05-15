import estnltk
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Self,
)
from dataclasses import dataclass

from .types import DirectionMode, EdgeContext
from .graph import SyntaxGraphIndex
from .patterns import PathPattern
from .conditions import EdgeConstraint
from .patterns import ChainMatch, MatchCollector

from .config import (
    DEFAULT_MAX_MATCHES_PER_SENTENCE,
    DEFAULT_DEDUP_MODE_MATCHER,
    VALID_DEDUP_MODES,
)


@dataclass(slots=True)
class DepChainMatcher:
    """
    Match dependency path patterns against one sentence-level syntax graph.

    This class performs the core graph-search phase of the pipeline:
    1. choose anchor-node candidates,
    2. expand the pattern step-by-step along dependency edges,
    3. materialise successful paths as `ChainMatch` objects,
    4. deduplicate/limit matches via `MatchCollector`.

    ## Attributes:
    - **patterns** (`Tuple[PathPattern, ...]`): A tuple of PathPattern objects that define the patterns to match against the syntax graph.
    - **dedup_mode** (`str`): The deduplication mode to use when collecting matches. Allowed values are "none" (no deduplication), "exact" (deduplicate based on exact match of `ChainMatch`), and "role_based" (deduplicate based on the combination of pattern name, sentence index, and role-to-token ID mapping).
    - **max_matches_per_sentence** (`int`): The maximum number of matches to collect for each sentence. Once this limit is reached, no new matches will be added for that sentence.
    - **allow_role_node_overlap** (`bool`): Whether to allow matches where the same node in the syntax graph is assigned to multiple roles in the pattern. If False, matches where a single node would be assigned to more than one role will be rejected. This can help prevent semantically confusing matches but may also exclude valid cases where a node legitimately fulfills multiple roles.
    """

    patterns: Tuple[PathPattern, ...]
    dedup_mode: str = DEFAULT_DEDUP_MODE_MATCHER
    max_matches_per_sentence: int = DEFAULT_MAX_MATCHES_PER_SENTENCE
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
        sentence_spans: Optional[List[Tuple[int, int]]] = None,
    ) -> List[ChainMatch]:
        """
        Match all configured patterns against one sentence graph.

        Args:
            graph_index (SyntaxGraphIndex): Sentence-level dependency graph.
            sentence_index (int): Zero-based sentence index in the source text.
            sentence_span (Optional[Tuple[int, int]], optional): Sentence
                character span override. If None, uses graph metadata when
                available.
            sentence_spans (Optional[List[Tuple[int, int]]], optional): All
                sentence character spans in the document, ordered by sentence
                index. When provided, the matcher can compute whether an edge
                crosses a sentence boundary. When None (default), every edge
                is assumed to stay within its sentence.

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
                sentence_spans=sentence_spans,
            )
            collector.extend(pattern_matches)

        return collector.all()

    def match_pattern_in_sentence(
        self: Self,
        pattern: PathPattern,
        graph_index: SyntaxGraphIndex,
        sentence_index: int,
        sentence_span: Optional[Tuple[int, int]] = None,
        sentence_spans: Optional[List[Tuple[int, int]]] = None,
    ) -> List[ChainMatch]:
        """
        Match one path pattern against one sentence graph.

        Args:
            pattern (PathPattern): Pattern to match.
            graph_index (SyntaxGraphIndex): Sentence-level dependency graph.
            sentence_index (int): Zero-based sentence index in the source text.
            sentence_span (Optional[Tuple[int, int]], optional): Sentence
                character span override.
            sentence_spans (Optional[List[Tuple[int, int]]], optional): All
                sentence character spans in the document, for cross-sentence
                edge detection.

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
                sentence_spans=sentence_spans,
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
        sentence_spans: Optional[List[Tuple[int, int]]] = None,
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
            sentence_spans (Optional[List[Tuple[int, int]]], optional): All
                sentence character spans, for cross-sentence edge detection.

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
                sentence_spans=sentence_spans,
            )
        else:
            candidate_pairs = self._enumerate_sources_to_target(
                graph_index=graph_index,
                target_node=known_node,
                edge_constraint=pattern.edge_steps[edge_index],
                sentence_spans=sentence_spans,
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
                    sentence_spans=sentence_spans,
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
        sentence_spans: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Tuple[estnltk.Span, EdgeContext]]:
        """
        Enumerate candidate target nodes reachable from `source_node`.

        Each candidate is returned together with the concrete `EdgeContext` that
        satisfied `edge_constraint`.

        Args:
            graph_index (SyntaxGraphIndex): Sentence-level dependency graph.
            source_node (estnltk.Span): Node from which to enumerate.
            edge_constraint (EdgeConstraint): Constraint the edge must satisfy.
            sentence_spans (Optional[List[Tuple[int, int]]], optional): All
                sentence character spans, for cross-sentence edge detection.
                When None, `crosses_sentence` is always False.
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
                    crosses = self._crosses_sentence(source_node, node, sentence_spans)
                    edge_context = self._build_edge_context(
                        direction=direction,
                        deprel=deprel,
                        hops=hops,
                        crosses_sentence=crosses,
                    )
                    if edge_constraint.matches(edge_context):
                        candidates.append((node, edge_context))
        return candidates

    def _enumerate_sources_to_target(
        self: Self,
        graph_index: SyntaxGraphIndex,
        target_node: estnltk.Span,
        edge_constraint: EdgeConstraint,
        sentence_spans: Optional[List[Tuple[int, int]]] = None,
    ) -> List[Tuple[estnltk.Span, EdgeContext]]:
        """
        Enumerate candidate source nodes that can reach `target_node`.

        This is used when filling pattern steps to the left of the anchor index.

        Args:
            graph_index (SyntaxGraphIndex): Sentence-level dependency graph.
            target_node (estnltk.Span): The target node to reach.
            edge_constraint (EdgeConstraint): Constraint the edge must satisfy.
            sentence_spans (Optional[List[Tuple[int, int]]], optional): All
                sentence character spans, for cross-sentence edge detection.
        """
        candidates: List[Tuple[estnltk.Span, EdgeContext]] = []
        for source_node in graph_index.iter_nodes():
            for candidate_node, edge_context in self._enumerate_from_node(
                graph_index=graph_index,
                source_node=source_node,
                edge_constraint=edge_constraint,
                sentence_spans=sentence_spans,
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

    def _sentence_index_for_node(
        self: Self,
        node: estnltk.Span,
        sentence_spans: List[Tuple[int, int]],
    ) -> Optional[int]:
        """
        Return the index of the sentence whose character span contains *node*.

        A node belongs to a sentence when its [start, end) is fully contained
        within the sentence's [span_start, span_end).

        Args:
            node (estnltk.Span): The node to locate.
            sentence_spans (List[Tuple[int, int]]): Ordered list of
                (start, end) character spans, one per sentence.

        Returns:
            Optional[int]: Sentence index, or None if the node does not
            fall inside any known sentence span.
        """
        for idx, (span_start, span_end) in enumerate(sentence_spans):
            if node.start >= span_start and node.end <= span_end:
                return idx
        return None

    def _crosses_sentence(
        self: Self,
        node_a: estnltk.Span,
        node_b: estnltk.Span,
        sentence_spans: Optional[List[Tuple[int, int]]],
    ) -> bool:
        """
        Determine whether two nodes belong to different sentences.

        If *sentence_spans* is None (no boundary information available),
        the method returns False, preserving backward-compatible behaviour.

        Args:
            node_a (estnltk.Span): First node.
            node_b (estnltk.Span): Second node.
            sentence_spans (Optional[List[Tuple[int, int]]]): Ordered list
                of (start, end) character spans for every sentence in the
                document, or None when boundary information is unavailable.

        Returns:
            bool: True if the two nodes fall inside different sentences,
            False otherwise (including when sentence_spans is None).
        """
        if sentence_spans is None:
            return False

        idx_a = self._sentence_index_for_node(node_a, sentence_spans)
        idx_b = self._sentence_index_for_node(node_b, sentence_spans)

        if idx_a is None or idx_b is None:
            return False

        return idx_a != idx_b

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
