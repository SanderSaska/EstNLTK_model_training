from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Iterable,
    Self,
    Any,
)
from dataclasses import dataclass

from .patterns import ChainMatch, MatchCollector


@dataclass(slots=True)
class PhraseDecorator:
    """
    Transform `ChainMatch` objects into output rows for downstream layers.

    The decorator is intentionally independent of traversal logic: it only
    handles output shaping/serialisation. This keeps the matcher focused on
    search and keeps output schema changes local to one class.

    ## Attributes:
    - **include_pattern_name** (`bool`): Whether to include the pattern name in the output.
    - **include_sentence_context** (`bool`): Whether to include sentence index and span in the output.
    - **include_role_token_ids** (`bool`): Whether to include the mapping of roles to token IDs in the output.
    - **include_role_texts** (`bool`): Whether to include the mapping of roles to their corresponding text spans in the output.
    - **include_role_spans** (`bool`): Whether to include the mapping of roles to their corresponding character spans (start, end) in the output.
    - **include_traversed_edges** (`bool`): Whether to include the list of traversed edges with their contexts in the output.
    - **include_metadata** (`bool`): Whether to include the metadata dictionary from the match in the output.
    - **output_text_roles** (`Optional[Tuple[str, ...]]`): If set, only include the text of the specified roles in the `matched_text` field. If None, include all roles specified in the pattern's `emit_roles`.
    - **output_field_prefix** (`str`): Optional prefix to add to all output field names. This can help avoid key collisions when combining outputs from multiple decorators or when integrating with other data sources.
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
