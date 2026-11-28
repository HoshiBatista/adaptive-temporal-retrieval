"""Frozen explicit-query-date policy: text-only decisions and exact fast-path bypass."""

import numpy as np

from temporalproof.retrieval.bm25 import Hit
from temporalproof.retrieval.query_time import analyze_query_time
from temporalproof.retrieval.timelyrag import ReferenceCode, automatic_rerank_with_qet


def role_aware_rerank(
    reference: ReferenceCode,
    scores: np.ndarray,
    document_ids: list[int],
    document_insert_times: np.ndarray,
    document_event_times: np.ndarray,
    query: str,
    query_insert_time: str | None,
    fast_ranking: list[Hit],
    *,
    before_characters: int = 16,
    after_characters: int = 32,
) -> tuple[list[Hit], dict]:
    """No query ID, qrels, answer or expensive outcome enters the decision.

    Input scores/IDs share a stable source-corpus axis; fast_ranking preserves the
    cached fast order, including ties, when the policy abstains.
    """
    if (
        scores.ndim != 1
        or not len(scores)
        or not np.isfinite(scores).all()
        or len(scores) != len(document_ids)
        or len(document_ids) != len(set(document_ids))
        or len(document_insert_times) != len(scores)
        or len(document_event_times) != len(scores)
        or len(fast_ranking) != len(scores)
        or len({h.doc_id for h in fast_ranking}) != len(scores)
        or {h.doc_id: h.score for h in fast_ranking}
        != dict(zip(document_ids, scores.tolist(), strict=True))
    ):
        raise ValueError("frozen fast candidates/scores/timestamps are unaligned")
    selection = analyze_query_time(
        query, before_characters=before_characters, after_characters=after_characters
    )
    info = {"selection": selection.model_dump(mode="json"), "temporal_rerank_called": False}
    if selection.action == "keep_fast_ranking":
        return list(fast_ranking), info
    hits, native_info = automatic_rerank_with_qet(
        reference,
        scores,
        document_ids,
        document_insert_times,
        document_event_times,
        query,
        query_insert_time,
        query_event_time=selection.selected_time,
    )
    return hits, {**info, "temporal_rerank_called": True, "native": native_info}
