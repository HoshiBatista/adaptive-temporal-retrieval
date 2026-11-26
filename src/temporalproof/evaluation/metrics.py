"""Binary qrel-based metrics, independent of the reference implementation."""

import math
from collections.abc import Sequence, Set


def _validate(ranking: Sequence[int], k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking must contain unique document IDs")


def recall_at_k(ranking: Sequence[int], relevant: Set[int], k: int) -> float:
    """Fraction of all judged-positive IDs retrieved; empty qrels yield zero."""
    _validate(ranking, k)
    return len(set(ranking[:k]) & relevant) / len(relevant) if relevant else 0.0


def ndcg_at_k(ranking: Sequence[int], relevant: Set[int], k: int) -> float:
    """Normalize by ALL positive qrels, including those missing from ranking."""
    _validate(ranking, k)
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(k, len(relevant))))
    actual = sum(
        1 / math.log2(rank + 2) for rank, doc_id in enumerate(ranking[:k]) if doc_id in relevant
    )
    return actual / ideal if ideal else 0.0
