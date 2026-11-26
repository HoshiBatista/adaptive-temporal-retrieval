"""Pure rank fusion and insertion-time diagnostics; no qrels or validity inference."""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from temporalproof.data.timelyqa import parse_timestamp
from temporalproof.retrieval.bm25 import Hit


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hit]], top_k: int, offset: int = 60
) -> list[Hit]:
    if top_k < 1 or offset < 1:
        raise ValueError("fusion cutoff and offset must be positive")
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        if len({hit.doc_id for hit in ranking}) != len(ranking):
            raise ValueError("input ranking contains duplicate IDs")
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.doc_id] += 1 / (offset + rank)
    return [
        Hit(doc_id, score)
        for doc_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    ]


@dataclass(frozen=True)
class InsertionFeatures:
    known: bool
    signed_gap_days: float | None
    inserted_after_query: bool | None
    affinity: float | None


def insertion_features(
    query_insert_time: str | None, doc_insert_time: str | None
) -> InsertionFeatures:
    query, document = parse_timestamp(query_insert_time), parse_timestamp(doc_insert_time)
    if query is None or document is None:
        return InsertionFeatures(False, None, None, None)
    gap = (document - query).total_seconds() / 86400
    return InsertionFeatures(True, gap, gap > 0, 1 / (1 + abs(gap) / 365))


def insertion_time_rerank(
    hits: Sequence[Hit],
    query_insert_time: str | None,
    document_times: Mapping[int, str | None],
    *,
    weight: float = 0.1,
) -> list[Hit]:
    """Min-max RRF score + fixed soft insertion affinity; unknown dates add zero.

    Both past and future insertion gaps are symmetric. This is NOT a known-at
    filter, event-time match, legal validity score or the official TimelyRAG method.
    """
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("time weight must be finite and nonnegative")
    if not hits:
        return []
    scores = np.array([hit.score for hit in hits], dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite ranking score")
    span = np.ptp(scores)
    normalized = (scores - scores.min()) / span if span else np.zeros(len(hits))
    ranked = []
    for hit, score in zip(hits, normalized, strict=True):
        affinity = insertion_features(query_insert_time, document_times.get(hit.doc_id)).affinity
        ranked.append(Hit(hit.doc_id, float(score + weight * (affinity or 0))))
    return sorted(ranked, key=lambda hit: (-hit.score, hit.doc_id))
