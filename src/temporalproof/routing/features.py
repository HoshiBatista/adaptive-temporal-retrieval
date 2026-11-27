"""Inference-available features: no qrels, outcomes or reranker scores accepted."""

import re
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from temporalproof.data.splits import normalize
from temporalproof.data.timelyqa import Document, parse_timestamp
from temporalproof.retrieval.bm25 import Hit
from temporalproof.retrieval.fusion import insertion_features

FEATURE_VERSION = "pre_deep_v1"


class PreDeepFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    query_characters: int = Field(strict=True, ge=0)
    query_whitespace_tokens: int = Field(strict=True, ge=0)
    explicit_year_count: int = Field(strict=True, ge=0)
    top10_overlap_fraction: float = Field(ge=0, le=1)
    top1_agreement: bool
    bm25_relative_margin: float = Field(ge=0, le=2)
    dense_margin: float = Field(ge=0)
    hybrid_relative_margin: float = Field(ge=0, le=2)
    hybrid_normalized_entropy: float | None = Field(default=None, ge=0, le=1)
    query_insertion_known: bool
    candidate_insertion_known_fraction: float = Field(ge=0, le=1)
    candidate_after_query_fraction: float | None = Field(default=None, ge=0, le=1)
    median_abs_insertion_gap_days: float | None = Field(default=None, ge=0)
    candidate_distinct_insertion_days: int = Field(strict=True, ge=0)
    candidate_unique_text_fraction: float = Field(ge=0, le=1)


def extract_features(
    query: str,
    query_insert_time: str | None,
    sparse: Sequence[Hit],
    dense: Sequence[Hit],
    hybrid: Sequence[Hit],
    documents: Mapping[int, Document],
) -> PreDeepFeatures:
    """Dates/text diversity are metadata proxies, not validity or true version chains."""
    for hits in (sparse, dense, hybrid):
        if len(hits) < 2 or len({h.doc_id for h in hits}) != len(hits):
            raise ValueError("features require at least two unique candidates per ranking")
        if not np.isfinite([h.score for h in hits]).all():
            raise ValueError("nonfinite fast scores")
        if any(
            (-b.score, b.doc_id) < (-a.score, a.doc_id)
            for a, b in zip(hits, hits[1:], strict=False)
        ):
            raise ValueError("fast ranking violates score/ID ordering")
    cutoff = min(10, len(sparse), len(dense))
    overlap = len({h.doc_id for h in sparse[:cutoff]} & {h.doc_id for h in dense[:cutoff]})

    def margin(hits):
        return (hits[0].score - hits[1].score) / max(abs(hits[0].score), abs(hits[1].score), 1e-12)

    scores = np.array([h.score for h in hybrid], dtype=float)
    if (scores < 0).any():
        raise ValueError("hybrid entropy requires nonnegative RRF scores")
    weights = scores[scores > 0] / scores.sum() if scores.sum() else np.array([])
    entropy = (
        float(np.clip(-(weights * np.log(weights)).sum() / np.log(len(scores)), 0, 1))
        if len(weights)
        else None
    )
    candidates = [documents[h.doc_id] for h in hybrid]
    dates = [parse_timestamp(doc.doc_insert_time) for doc in candidates]
    known = [date for date in dates if date is not None]
    pairs = [insertion_features(query_insert_time, doc.doc_insert_time) for doc in candidates]
    gaps = [p for p in pairs if p.known]
    return PreDeepFeatures(
        query_characters=len(query),
        query_whitespace_tokens=len(query.split()),
        explicit_year_count=len(set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", query))),
        top10_overlap_fraction=overlap / cutoff,
        top1_agreement=sparse[0].doc_id == dense[0].doc_id,
        bm25_relative_margin=margin(sparse),
        dense_margin=dense[0].score - dense[1].score,
        hybrid_relative_margin=margin(hybrid),
        hybrid_normalized_entropy=entropy,
        query_insertion_known=parse_timestamp(query_insert_time) is not None,
        candidate_insertion_known_fraction=len(known) / len(candidates),
        candidate_after_query_fraction=sum(p.inserted_after_query for p in gaps) / len(gaps)
        if gaps
        else None,
        median_abs_insertion_gap_days=float(np.median([abs(p.signed_gap_days) for p in gaps]))
        if gaps
        else None,
        candidate_distinct_insertion_days=len({date.date() for date in known}),
        candidate_unique_text_fraction=len({normalize(doc.text) for doc in candidates})
        / len(candidates),
    )
