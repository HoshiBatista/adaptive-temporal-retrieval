"""Offline outcomes/diagnostics from training queries only; never inference features."""

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from temporalproof.data.splits import SplitManifest
from temporalproof.data.timelyqa import DatasetError, Identifier
from temporalproof.routing.features import PreDeepFeatures


class TrainOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    query_id: Identifier
    group_id: str
    fast_ndcg_at_10: float = Field(ge=0, le=1)
    deep_ndcg_at_10: float = Field(ge=0, le=1)
    gain: float = Field(ge=-1, le=1)
    fast_ms: float = Field(gt=0)
    incremental_rerank_ms: float = Field(gt=0)
    candidate_recall_at_100: float = Field(ge=0, le=1)
    cost_sensitive_positive: dict[str, bool]


def training_groups(split: SplitManifest) -> dict[int, str]:
    return {
        q: hashlib.sha256(json.dumps(sorted(group)).encode()).hexdigest()
        for group in split.groups["train"]
        for q in group
    }


def build_train_outcomes(
    split: SplitManifest,
    features: Mapping[int, PreDeepFeatures],
    fast_rows: Sequence[dict],
    deep_rows: Sequence[dict],
    lambdas: Sequence[float],
) -> list[TrainOutcome]:
    fast = {r["query_id"]: r for r in fast_rows}
    deep = {r["query_id"]: r for r in deep_rows}
    if (
        not fast
        or len(fast) != len(fast_rows)
        or len(deep) != len(deep_rows)
        or fast.keys() != deep.keys()
        or fast.keys() != features.keys()
    ):
        raise DatasetError("outcomes require matching unique feature/fast/deep query IDs")
    groups = training_groups(split)
    if not fast.keys() <= groups.keys():
        raise DatasetError("router outcomes must use training queries only")
    if (
        not lambdas
        or not np.isfinite(lambdas).all()
        or min(lambdas) < 0
        or len(set(lambdas)) != len(lambdas)
    ):
        raise ValueError("lambda penalties must be unique finite nonnegative nDCG/second values")
    result = []
    for q in fast:
        a, b = fast[q], deep[q]
        if a["recall_at_100"] != b["recall_at_100"]:
            raise DatasetError("fast/deep candidate coverage differs")
        gain = float(b["ndcg_at_10"] - a["ndcg_at_10"])
        # M3 already measures INCREMENTAL reranking. Never subtract fast time again.
        cost_ms = float(b["retrieval_ms"])
        result.append(
            TrainOutcome(
                query_id=q,
                group_id=groups[q],
                fast_ndcg_at_10=a["ndcg_at_10"],
                deep_ndcg_at_10=b["ndcg_at_10"],
                gain=gain,
                fast_ms=a["retrieval_ms"],
                incremental_rerank_ms=cost_ms,
                candidate_recall_at_100=a["recall_at_100"],
                cost_sensitive_positive={
                    str(weight): gain > weight * cost_ms / 1000 + 1e-12 for weight in lambdas
                },
            )
        )
    return result


def diagnostic_policies(
    outcomes: Sequence[TrainOutcome],
    features: Mapping[int, PreDeepFeatures],
    *,
    budgets: Sequence[float],
    disagreement_threshold: float,
    seed: int,
) -> list[dict]:
    """In-sample quality and modeled additive costs, not trained/held-out/live routing."""
    by_id = {row.query_id: row for row in outcomes}
    if not by_id or len(by_id) != len(outcomes) or by_id.keys() != features.keys():
        raise ValueError("diagnostics require matching unique outcome/feature IDs")
    if (
        not budgets
        or not np.isfinite(budgets).all()
        or min(budgets) < 0
        or max(budgets) > 1
        or not 0 <= disagreement_threshold <= 1
    ):
        raise ValueError("invalid query budgets or disagreement threshold")
    ids = set(by_id)
    disagreement = {q for q in ids if features[q].top10_overlap_fraction < disagreement_threshold}
    selections = [
        ("always_fast", set(), False),
        ("always_deep", ids, False),
        ("low_agreement", disagreement, False),
        (
            "year_and_low_agreement",
            {q for q in disagreement if features[q].explicit_year_count > 0},
            False,
        ),
    ]
    random_order = sorted(
        ids, key=lambda q: (hashlib.sha256(f"{seed}:route:{q}".encode()).hexdigest(), q)
    )
    oracle = sorted((q for q in ids if by_id[q].gain > 1e-12), key=lambda q: (-by_id[q].gain, q))
    for budget in budgets:
        n = int(np.floor(len(ids) * budget))
        selections.extend(
            [
                (f"random_query_budget_{budget:g}", set(random_order[:n]), False),
                (f"oracle_query_budget_{budget:g}", set(oracle[:n]), True),
            ]
        )
    output = []
    for name, routed, uses_outcomes in selections:
        output.append(
            {
                "name": name,
                "uses_outcomes_at_decision": uses_outcomes,
                "scope": "offline training oracle upper bound; not deployable"
                if uses_outcomes
                else "predeclared in-sample diagnostic; no held-out claim",
                "routed_count": len(routed),
                "routed_fraction": len(routed) / len(ids),
                "routed_query_ids": sorted(routed),
                "ndcg_at_10": float(
                    np.mean(
                        [
                            r.deep_ndcg_at_10 if r.query_id in routed else r.fast_ndcg_at_10
                            for r in outcomes
                        ]
                    )
                ),
                "mean_gain_vs_fast": float(
                    np.mean([r.gain if r.query_id in routed else 0 for r in outcomes])
                ),
                "modeled_mean_additive_stage_ms": float(
                    np.mean(
                        [
                            r.fast_ms + (r.incremental_rerank_ms if r.query_id in routed else 0)
                            for r in outcomes
                        ]
                    )
                ),
                "routed_beneficial": sum(by_id[q].gain > 1e-12 for q in routed),
                "routed_harmful": sum(by_id[q].gain < -1e-12 for q in routed),
                "cost_scope": (
                    "sum of separately measured stages, including fast feature extraction; "
                    "not measured live latency or p95"
                ),
            }
        )
    return output


def feature_slices(
    outcomes: Sequence[TrainOutcome], features: Mapping[int, PreDeepFeatures], threshold: float
) -> dict:
    buckets = {
        "low_agreement": [
            r for r in outcomes if features[r.query_id].top10_overlap_fraction < threshold
        ],
        "other_agreement": [
            r for r in outcomes if features[r.query_id].top10_overlap_fraction >= threshold
        ],
        "explicit_year": [r for r in outcomes if features[r.query_id].explicit_year_count > 0],
        "no_explicit_year": [r for r in outcomes if features[r.query_id].explicit_year_count == 0],
        "unknown_query_insertion": [
            r for r in outcomes if not features[r.query_id].query_insertion_known
        ],
    }
    return {
        key: {
            "query_count": len(rows),
            "positive_gain_count": sum(r.gain > 1e-12 for r in rows),
            "mean_gain": float(np.mean([r.gain for r in rows])) if rows else None,
        }
        for key, rows in buckets.items()
    }
