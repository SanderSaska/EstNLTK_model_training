from typing import (
    Optional,
    Any,
    Callable,
    TypeAlias,
)
from enum import Enum

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
