"""
Centralised configuration constants for the dep_chain_tagger package.

Every default value, limit, heuristic weight, and sentinel used across the
package is defined here so that:
  * users can discover and override defaults in one place,
  * adding a new dedup mode / missing marker / output attribute requires
    editing only this file,
  * the constants can be imported into tests for deterministic assertions.

Constants are intentionally *module-level* (not class-level) so that they can
be referenced before any class is instantiated.
"""

# ──────────────────────────────────────────────────────────────
# 1.  ValueCondition — missing-value sentinels
# ──────────────────────────────────────────────────────────────
DEFAULT_MISSING_MARKERS: tuple = (None, "", "_")
"""
Tuple of values that `ValueCondition._is_missing()` treats as absent.

Extend or replace this when working with corpora that use alternative
missing-value conventions (e.g. ``"NA"``, ``"--"``, ``"null"``).

Example::

    from dep_chain_tagger.config import DEFAULT_MISSING_MARKERS
    markers = DEFAULT_MISSING_MARKERS + ("NA", "--")
    cond = ValueCondition(mode=ConditionMode.EXACT, value="N",
                          allow_missing=True, missing_markers=markers)
"""

# ──────────────────────────────────────────────────────────────
# 2.  NodeConstraint — selectivity scoring weights
# ──────────────────────────────────────────────────────────────
SELECTIVITY_WEIGHT_EXACT: float = 1.0
"""Score added when a ValueCondition or FeatureCondition uses EXACT mode."""

SELECTIVITY_WEIGHT_NEGATION: float = 0.5
"""Score added when a ValueCondition or FeatureCondition uses NEGATION mode."""

SELECTIVITY_WEIGHT_EXTRA_PREDICATE: float = 0.5
"""Score added per extra predicate in `NodeConstraint.extra_predicates`."""

# ──────────────────────────────────────────────────────────────
# 3.  Match capacity limits
# ──────────────────────────────────────────────────────────────
DEFAULT_MAX_MATCHES_PER_COLLECTOR: int = 100_000
"""Default `max_matches` for `MatchCollector`."""

DEFAULT_MAX_MATCHES_PER_SENTENCE: int = 100_000
"""Default `max_matches_per_sentence` for `DepChainMatcher`."""

DEFAULT_MAX_TOTAL_MATCHES: int = 1_000_000
"""Default `max_total_matches` for `DepChainTaggerOrchestrator`."""

# ──────────────────────────────────────────────────────────────
# 4.  Deduplication
# ──────────────────────────────────────────────────────────────
VALID_DEDUP_MODES: frozenset = frozenset({"none", "exact", "role_based"})
"""All recognised dedup_mode string values.

Every site that validates `dedup_mode` should test membership against
this set instead of repeating the literal strings.
"""

DEFAULT_DEDUP_MODE_MATCHER: str = "role_based"
"""Default `dedup_mode` for `DepChainMatcher`."""

DEFAULT_DEDUP_MODE_COLLECTOR: str = "none"
"""Default `dedup_mode` for `MatchCollector`."""

DEFAULT_DEDUP_MODE_SENTENCE: str = "role_based"
"""Default `sentence_match_dedup_mode` for `DepChainTaggerOrchestrator`."""

DEFAULT_DEDUP_MODE_GLOBAL: str = "none"
"""Default `global_dedup_mode` for `DepChainTaggerOrchestrator`."""

# ──────────────────────────────────────────────────────────────
# 5.  Tagger — output layer defaults  (H11)
# ──────────────────────────────────────────────────────────────
DEFAULT_OUTPUT_LAYER_NAME: str = "dep_chains"
"""Default name of the estnltk Layer produced by `DepChainTagger`."""

DEFAULT_OUTPUT_ATTRIBUTES: tuple = (
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
"""Default attribute names on the output Layer."""

# ──────────────────────────────────────────────────────────────
# 6.  Tagger — input layer names  (H2 / H12)
# ──────────────────────────────────────────────────────────────
DEFAULT_SYNTAX_LAYER_NAME: str = "stanza_syntax"
"""Name of the input syntax layer that `DepChainTagger` reads from.

Currently hard-wired to Stanza Syntax output.  Making this configurable
is the first step toward supporting other syntax providers.
"""

# ──────────────────────────────────────────────────────────────
# 7.  Anchor role fallback  (H3)
# ──────────────────────────────────────────────────────────────
DEFAULT_ANCHOR_ROLE: str = "self"
"""Role name used as fallback anchor when the pattern's `anchor_role`
is not present in the match.  Override this if your patterns use a
different convention for the "self" / pivot role.
"""
