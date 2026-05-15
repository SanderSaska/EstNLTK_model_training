from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Self,
    Any,
    Literal,
)
import hashlib

from estnltk.taggers import Tagger
from estnltk import Layer, Text

from .patterns import ChainMatch, PathPattern
from .orchestrator import DepChainTaggerOrchestrator
from .graph import SyntaxGraphIndex

from .config import (
    DEFAULT_DEDUP_MODE_GLOBAL,
    DEFAULT_DEDUP_MODE_SENTENCE,
    DEFAULT_OUTPUT_LAYER_NAME,
    DEFAULT_OUTPUT_ATTRIBUTES,
    DEFAULT_SYNTAX_LAYER_NAME,
    DEFAULT_ANCHOR_ROLE,
    DEFAULT_MAX_MATCHES_PER_SENTENCE,
    DEFAULT_MAX_TOTAL_MATCHES,
    VALID_DEDUP_MODES,
)


def _deterministic_hash(items: Tuple[Any, ...]) -> str:
    """
    Compute a short, deterministic hex digest from a tuple of items.

    Unlike Python's built-in ``hash()``, this function is stable across
    interpreter sessions (no PYTHONHASHSEED randomisation).

    Args:
        items (Tuple[Any, ...]): Arbitrary tuple to hash.

    Returns:
        str: 12-character hexadecimal digest string.
    """
    serialized = str(items).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:12]


class DepChainTagger(Tagger):
    """
    EstNLTK-compatible wrapper for DepChainTaggerOrchestrator.

    This class integrates the DepChainTaggerOrchestrator into the EstNLTK pipeline by implementing
    the Tagger interface. It processes sentence syntax layers and produces a new layer
    containing dependency chain matches.

    ## Attributes:
    - **patterns** (`Tuple[PathPattern, ...]`): A tuple of PathPattern objects that define the patterns to match against the syntax graphs.
    - **output_layer** (`str`): Name of the output layer to create with matches.
    - **output_attributes** (`Optional[Tuple[str, ...]]`): Optional tuple of attribute names for the output layer. If None, a default set of attributes will be used.
    - **sentence_match_dedup_mode** (`str`): Deduplication mode to apply within each sentence's matches. Allowed values are "none", "exact", and "role_based". This controls how matches that are similar within the same sentence are filtered before being added to the output layer.
    - **max_matches_per_sentence** (`int`): Maximum number of matches to accept for each individual sentence. This can help prevent combinatorial explosion in very complex sentences.
    - **allow_role_node_overlap** (`bool`): Whether to allow matches where the same node in the syntax graph is assigned to multiple roles in the pattern. This setting is passed down to the matcher and can help control match quality.
    - **global_dedup_mode** (`str`): Deduplication mode to apply across all matches from all sentences before adding to the output layer. Allowed values are "none", "exact", and "role_based". This controls how matches that are similar across different sentences are filtered before being included in the output layer.
    - **max_total_matches** (`int`): Maximum total number of matches to include in the output layer across all sentences. This can help prevent memory issues when processing large documents with many matches.
        - **_depchain_tagger** (`DepChainTaggerOrchestrator`): Internal instance of the orchestrator that runs the matching and decoration pipeline.
        - **_pattern_by_name** (`Dict[str, PathPattern]`): Internal mapping of pattern names to pattern objects for quick lookup during matching and decoration.
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
        output_layer: str = DEFAULT_OUTPUT_LAYER_NAME,
        output_attributes: Optional[Tuple[str, ...]] = None,
        sentence_match_dedup_mode: Literal[
            "none", "exact", "role_based"
        ] = DEFAULT_DEDUP_MODE_SENTENCE,
        max_matches_per_sentence: int = DEFAULT_MAX_MATCHES_PER_SENTENCE,
        allow_role_node_overlap: bool = False,
        global_dedup_mode: Literal[
            "none", "exact", "role_based"
        ] = DEFAULT_DEDUP_MODE_GLOBAL,
        max_total_matches: int = DEFAULT_MAX_TOTAL_MATCHES,
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
        self.input_layers = (DEFAULT_SYNTAX_LAYER_NAME, "sentences")
        self.output_layer = output_layer
        if output_attributes is None:
            self.output_attributes = DEFAULT_OUTPUT_ATTRIBUTES
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
        layer = self._make_layer_template()
        layer.text_object = text

        if (
            layers is None
            or DEFAULT_SYNTAX_LAYER_NAME not in layers
            or "sentences" not in layers
        ):
            return layer

        sentences_layer = getattr(layers, "sentences")

        # Collect all sentence spans upfront for cross-sentence detection
        all_sentence_spans: List[Tuple[int, int]] = [
            (sentence.start, sentence.end) for sentence in sentences_layer
        ]

        try:
            for i, sentence in enumerate(sentences_layer):
                sentence_span = (sentence.start, sentence.end)
                sentence_syntax = getattr(layers, DEFAULT_SYNTAX_LAYER_NAME)
                sentence_matches = self._run_depchain_matcher(
                    i, sentence_span, sentence_syntax, all_sentence_spans
                )
                try:
                    for match in sentence_matches:
                        self._add_match_to_layer(layer, match, text)
                except Exception as exc:
                    if status is not None:
                        status.setdefault("errors", []).append(
                            f"Error adding matches to layer for sentence {i}: {str(exc)}"
                        )

        except Exception as exc:
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
        sentence_spans: Optional[List[Tuple[int, int]]] = None,
    ) -> List[ChainMatch]:
        """
        Run the DepChainMatcher on a single sentence's syntax graph to obtain matches.

        Args:
            sentence_index (int): Index of the sentence being processed.
            sentence_span (Tuple[int, int]): Character span of the sentence in the text.
            stanza_syntax_layer (Layer): The syntax layer for the sentence, containing token annotations.
            sentence_spans (Optional[List[Tuple[int, int]]]): List of all sentence spans in the text for cross-sentence detection.
        """

        if stanza_syntax_layer is None or len(stanza_syntax_layer) == 0:
            return []

        graph_index = SyntaxGraphIndex(
            stanza_syntax_layer=stanza_syntax_layer,
            sentence_id=sentence_index,
            sentence_span=sentence_span,
        )

        matcher = self._depchain_tagger._get_matcher()
        matches = matcher.match_sentence(
            graph_index=graph_index,
            sentence_index=sentence_index,
            sentence_span=sentence_span,
            sentence_spans=sentence_spans,
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

        Uses emit_roles to filter which roles actually get annotated.
        Uses anchor_role to flag which emitted role is the primary pivot.

        Args:
            layer (Layer): Target layer to which annotation is added.
            match (ChainMatch): The match to convert.
            text (Text): Source text object.

        Raises:
            ValueError: If match has no roles or anchor role not found.
        """
        pattern = self._pattern_by_name.get(match.pattern_name)

        # 1. Determine which roles to emit (fallback to all roles if not specified)
        if pattern and pattern.emit_roles:
            emit_roles = set(pattern.emit_roles)
        else:
            emit_roles = set(match.role_to_node.keys())

        # 2. Determine the anchor role (for the is_anchor flag)
        anchor_role = (
            pattern.anchor_role
            if pattern and pattern.anchor_role in match.role_to_node
            else None
        )
        if anchor_role is None:
            anchor_role = (
                DEFAULT_ANCHOR_ROLE
                if DEFAULT_ANCHOR_ROLE in match.role_to_node
                else next(iter(match.role_to_token_id.keys()), None)
            )

        match_id = f"{match.sentence_index}:{match.pattern_name}:{_deterministic_hash(tuple(sorted(match.role_to_token_id.items())))}"

        # 3. Iterate ONLY over the roles the pattern designates for emission
        for role, node in match.role_to_node.items():
            if role not in emit_roles:
                continue  # Skip roles that are not meant to be emitted

            annotation_dict = {
                "pattern_name": match.pattern_name,
                "matched_text": match.matched_text,
                "upostag": getattr(node, "upostag", None),
                "xpostag": getattr(node, "xpostag", None),
                "feats": getattr(node, "feats", None),
                "lemma": getattr(node, "lemma", None),
                "deprel": getattr(node, "deprel", None),
                "role": role,
                # is_anchor is True only if this emitted role happens to be the anchor
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
