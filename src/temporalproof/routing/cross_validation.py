"""Grouped out-of-fold training diagnostics; every budget is allocated within a fold."""

import hashlib
from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from temporalproof.routing.features import PreDeepFeatures
from temporalproof.routing.learned import fit_router, predict_scores
from temporalproof.routing.targets import TrainOutcome


class RouterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    seed: int = Field(strict=True, ge=0)
    query_count: int = Field(strict=True, ge=2)
    n_splits: int = Field(strict=True, ge=2)
    source_pilot_config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_data_snapshot_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_data_git_commit: str = Field(pattern=r"^[a-f0-9]{7,40}$")
    penalties_ndcg_per_second: tuple[float, ...]
    feature_sets: tuple[str, ...]
    regularization_c: float = Field(gt=0)
    class_weight: str
    max_iter: int = Field(strict=True, ge=1)
    tol: float = Field(gt=0)
    probability_threshold: float = Field(ge=0, le=1)
    query_route_budgets: tuple[float, ...]
    disagreement_threshold: float = Field(ge=0, le=1)
    bootstrap_samples: int = Field(strict=True, ge=100)
    primary_penalty: float = Field(ge=0)
    primary_feature_set: str
    primary_query_budget: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_controls(self):
        if (
            not self.feature_sets
            or len(set(self.feature_sets)) != len(self.feature_sets)
            or not set(self.feature_sets) <= {"full", "no_temporal"}
            or self.class_weight != "balanced"
            or self.primary_feature_set not in self.feature_sets
        ):
            raise ValueError("unsupported logistic controls")
        for values, upper in [
            (self.penalties_ndcg_per_second, None),
            (self.query_route_budgets, 1),
        ]:
            if (
                not values
                or list(values) != sorted(set(values))
                or min(values) < 0
                or (upper is not None and max(values) > upper)
            ):
                raise ValueError("invalid predeclared penalties/budgets")
        if (
            self.primary_penalty not in self.penalties_ndcg_per_second
            or self.primary_query_budget not in self.query_route_budgets
        ):
            raise ValueError("primary operating point must be predeclared")
        return self


def fold_assignment(outcomes: Sequence[TrainOutcome], config: RouterConfig) -> dict[int, int]:
    from sklearn.model_selection import GroupKFold

    ids = [r.query_id for r in outcomes]
    groups = [r.group_id for r in outcomes]
    if (
        len(ids) != len(set(ids))
        or len(ids) != config.query_count
        or len(set(groups)) < config.n_splits
    ):
        raise ValueError("insufficient unique training groups or wrong query count")
    order = np.argsort(ids)
    ordered_groups = np.array(groups)[order]
    folds = {}
    splitter = GroupKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    for fold, (_, held) in enumerate(
        splitter.split(np.zeros((len(ids), 1)), groups=ordered_groups)
    ):
        for i in held:
            folds[ids[order[i]]] = fold
    if len(folds) != len(ids):
        raise ValueError("incomplete out-of-fold assignment")
    return folds


def grouped_interval(
    values: Sequence[float], groups: Sequence[str], seed: int, samples: int
) -> list[float]:
    if len(values) != len(groups) or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("invalid paired group values")
    unique = sorted(set(groups))
    idx = {g: i for i, g in enumerate(unique)}
    positions = np.array([idx[g] for g in groups])
    sums = np.bincount(positions, weights=values)
    counts = np.bincount(positions)
    draws = np.random.default_rng(seed).integers(len(unique), size=(samples, len(unique)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def cross_fit(
    features: Mapping[int, PreDeepFeatures],
    outcomes: Sequence[TrainOutcome],
    folds: Mapping[int, int],
    config: RouterConfig,
    penalty: float,
    kind: str,
) -> tuple[list[dict], list[dict]]:
    rows = {r.query_id: r for r in outcomes}
    if rows.keys() != features.keys() or rows.keys() != folds.keys():
        raise ValueError("mismatched cross-fit rows")
    groups_to_fold = {}
    for q, r in rows.items():
        if r.group_id in groups_to_fold and groups_to_fold[r.group_id] != folds[q]:
            raise ValueError("group crosses folds")
        groups_to_fold[r.group_id] = folds[q]
    if set(folds.values()) != set(range(config.n_splits)):
        raise ValueError("incomplete fold set")
    predictions = []
    models = []
    for fold in range(config.n_splits):
        train = sorted(q for q in rows if folds[q] != fold)
        held = sorted(q for q in rows if folds[q] == fold)
        train_groups = {rows[q].group_id for q in train}
        held_groups = {rows[q].group_id for q in held}
        if train_groups & held_groups:
            raise ValueError("training/held-out groups overlap")
        labels = [
            int(rows[q].gain > penalty * rows[q].incremental_rerank_ms / 1000 + 1e-12)
            for q in train
        ]
        started = perf_counter()
        model, info = fit_router(
            [features[q] for q in train],
            labels,
            kind=kind,
            c=config.regularization_c,
            seed=config.seed,
            max_iter=config.max_iter,
            tol=config.tol,
        )
        fit_ms = (perf_counter() - started) * 1000
        for _ in range(3):
            predict_scores(model, [features[train[0]]])
        for q in held:
            started = perf_counter()
            score = float(predict_scores(model, [features[q]])[0])
            elapsed = (perf_counter() - started) * 1000
            predictions.append(
                {"query_id": q, "fold": fold, "score": score, "router_predict_ms": elapsed}
            )
        models.append(
            {
                "fold": fold,
                "train_query_ids": train,
                "held_out_query_ids": held,
                "train_group_count": len(train_groups),
                "held_group_count": len(held_groups),
                "fit_ms": fit_ms,
                "fit_info": info,
                "model": model.model_dump(),
            }
        )
    return sorted(predictions, key=lambda r: r["query_id"]), models


def select_fold_budget(predictions: Sequence[dict], budget: float) -> set[int]:
    """No cross-fold score ranking: other folds' models may have seen this fold's labels."""
    selected = set()
    for fold in sorted({p["fold"] for p in predictions}):
        rows = sorted(
            (p for p in predictions if p["fold"] == fold),
            key=lambda p: (-p["score"], p["query_id"]),
        )
        selected.update(p["query_id"] for p in rows[: int(np.floor(len(rows) * budget))])
    return selected


def control_scores(
    features: Mapping[int, PreDeepFeatures], folds: Mapping[int, int], seed: int, kind: str
) -> list[dict]:
    rows = []
    for q, f in features.items():
        if kind == "random":
            score = int(hashlib.sha256(f"{seed}:route:{q}".encode()).hexdigest(), 16) / 2**256
        elif kind == "agreement":
            score = -f.top10_overlap_fraction
        else:
            raise ValueError("unknown control")
        rows.append({"query_id": q, "fold": folds[q], "score": score})
    return rows


def evaluate_routes(
    name: str,
    routed: set[int],
    outcomes: Sequence[TrainOutcome],
    config: RouterConfig,
    predict_times: Mapping[int, float] | None = None,
) -> dict:
    if not routed <= {r.query_id for r in outcomes}:
        raise ValueError("unknown routing IDs")
    gains = [r.gain if r.query_id in routed else 0.0 for r in outcomes]
    quality = [r.deep_ndcg_at_10 if r.query_id in routed else r.fast_ndcg_at_10 for r in outcomes]
    base = [r.fast_ms + (r.incremental_rerank_ms if r.query_id in routed else 0) for r in outcomes]
    overhead = [predict_times[r.query_id] if predict_times is not None else 0 for r in outcomes]
    return {
        "name": name,
        "routed_query_ids": sorted(routed),
        "routed_count": len(routed),
        "routed_fraction": len(routed) / len(outcomes),
        "ndcg_at_10": float(np.mean(quality)),
        "mean_gain_vs_fast": float(np.mean(gains)),
        "paired_group_bootstrap_95_interval": grouped_interval(
            gains, [r.group_id for r in outcomes], config.seed, config.bootstrap_samples
        ),
        "modeled_mean_stage_ms": float(np.mean(base)),
        "modeled_mean_with_router_ms": float(np.mean(np.array(base) + overhead)),
        "routed_beneficial": sum(r.gain > 1e-12 and r.query_id in routed for r in outcomes),
        "routed_harmful": sum(r.gain < -1e-12 and r.query_id in routed for r in outcomes),
        "scope": "grouped OOF train-only; modeled stage sums, not measured routed service latency",
    }
