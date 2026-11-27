"""Pure interpolation of two ranks on an identical frozen candidate pool."""

import math
from collections.abc import Sequence

from temporalproof.retrieval.bm25 import Hit


def blend_ranks(
    fast: Sequence[Hit], deep: Sequence[Hit], weight: float, offset: int = 60
) -> list[Hit]:
    """(1-w)/(offset+fast_rank) + w/(offset+deep_rank); no qrels or metadata.

    Weight zero is a true fast-only bypass; weight one preserves the deep endpoint.
    Positive weights require the complete expensive ranking, however small the weight.
    """
    if isinstance(weight, bool) or not math.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("deep weight must be finite in [0, 1]")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 1:
        raise ValueError("rank offset must be a positive integer")
    a, b = [h.doc_id for h in fast], [h.doc_id for h in deep]
    if not a or len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) != set(b):
        raise ValueError("blend requires the same nonempty unique candidate pool")
    if not all(math.isfinite(h.score) for h in [*fast, *deep]):
        raise ValueError("nonfinite source score")
    if weight == 0:
        return list(fast)
    if weight == 1:
        return list(deep)
    deep_rank = {h.doc_id: rank for rank, h in enumerate(deep, 1)}
    hits = [
        Hit(h.doc_id, (1 - weight) / (offset + rank) + weight / (offset + deep_rank[h.doc_id]))
        for rank, h in enumerate(fast, 1)
    ]
    return sorted(hits, key=lambda h: (-h.score, h.doc_id))
