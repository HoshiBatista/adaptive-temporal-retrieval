"""Cached M4.2 diagnostics and fold-local rank-weight selection; no neural inference."""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from temporalproof.data.source import (
    ARCHIVE_SHA256,
    DEFAULT_ARCHIVE,
    load_archive_records,
    sha256_file,
)
from temporalproof.data.splits import DEFAULT_SPLIT, SplitManifest, select_queries, verify_split
from temporalproof.data.timelyqa import DatasetError, QueryRecord
from temporalproof.evaluation.baseline import _write_json, source_snapshot
from temporalproof.evaluation.metrics import ndcg_at_k, recall_at_k
from temporalproof.evaluation.pilot import read_rows, write_rows
from temporalproof.evaluation.router_cv import load_training_bundle
from temporalproof.evaluation.router_validation import validation_groups, verify_checkpoint
from temporalproof.retrieval.bm25 import Hit
from temporalproof.retrieval.rank_blend import blend_ranks
from temporalproof.routing.cross_validation import RouterConfig, fold_assignment, grouped_interval
from temporalproof.routing.targets import training_groups

METRICS = ("ndcg_at_10", "recall_at_10", "recall_at_20", "recall_at_100")
ROUTER_CONFIG = Path("configs/experiments/m4_logistic_groupcv_company_policies.json")


class BlendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    train_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    validation_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    router_config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    seed: int = Field(strict=True, ge=0)
    train_query_count: int = Field(strict=True, ge=2)
    validation_query_count: int = Field(strict=True, ge=1)
    n_splits: int = Field(strict=True, ge=2)
    top_k: Literal[100]
    rank_offset: int = Field(strict=True, ge=1)
    candidate_weights: tuple[float, ...]
    selection_tolerance: float = Field(ge=0, le=1e-12)
    bootstrap_samples: int = Field(strict=True, ge=100)
    selection_metric: Literal["mean_train_ndcg_at_10"]
    tie_break: Literal["smaller_deep_weight"]
    primary: Literal["fold_selected_all_query_blend"]
    routing_transfer_budget: Literal[0.1]

    @model_validator(mode="after")
    def conservative_grid(self) -> Self:
        w = self.candidate_weights
        if not w or list(w) != sorted(set(w)) or w[0] != 0 or w[-1] > 0.3:
            raise ValueError("sorted conservative weights must include zero and be <= .3")
        return self


@dataclass
class RankingBundle:
    partition: Literal["train", "validation"]
    ids: list[int]
    groups: dict[int, str]
    fast: dict[int, list[Hit]]
    deep: dict[int, list[Hit]]
    qrels: dict[int, set[int]]
    fast_rows: dict[int, dict]
    deep_rows: dict[int, dict]


def verify_source(root: Path, expected: str) -> dict:
    if sha256_file(root / "manifest.json") != expected:
        raise DatasetError("source manifest differs from predeclared pin")
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        p = root / name
        if not p.resolve().is_relative_to(root.resolve()) or sha256_file(p) != digest:
            raise DatasetError("source artifact checksum/path mismatch")
    return manifest


def load_trec(path: Path, ids: list[int], top_k: int) -> dict[int, list[Hit]]:
    result = {q: [] for q in ids}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 6:
            raise DatasetError("invalid TREC row")
        q, _, doc, rank, score, _ = parts
        q, doc, rank, score = int(q), int(doc), int(rank), float(score)
        if q not in result or doc < 0 or rank != len(result[q]) + 1 or not math.isfinite(score):
            raise DatasetError("invalid query/rank/score in source ranking")
        result[q].append(Hit(doc, score))
    for hits in result.values():
        if (
            len(hits) != top_k
            or len({h.doc_id for h in hits}) != top_k
            or hits != sorted(hits, key=lambda h: (-h.score, h.doc_id))
        ):
            raise DatasetError("source rankings must have sorted unique top-k candidates")
    return result


def metrics(hits: list[Hit], relevant: set[int]) -> dict[str, float]:
    ranking = [h.doc_id for h in hits]
    return {
        "ndcg_at_10": ndcg_at_k(ranking, relevant, 10),
        **{f"recall_at_{k}": recall_at_k(ranking, relevant, k) for k in (10, 20, 100)},
    }


def load_bundle(root, split, records, config, partition) -> RankingBundle:
    if partition not in {"train", "validation"}:
        raise DatasetError("test partition is forbidden")
    expected = (
        config.train_manifest_sha256 if partition == "train" else config.validation_manifest_sha256
    )
    manifest = verify_source(root, expected)
    if manifest["archive_sha256"] != ARCHIVE_SHA256:
        raise DatasetError("source archive mismatch")
    count = config.train_query_count if partition == "train" else config.validation_query_count
    ids = select_queries(split, partition, count, config.seed)
    if len(ids) != count or (
        partition == "validation" and len(ids) != len(split.query_ids(partition))
    ):
        raise DatasetError("wrong canonical query count")
    plan = json.loads((root / "plan.json").read_text())
    if plan["selected_query_ids"] != ids or plan["test_partition_evaluated"] is not False:
        raise DatasetError("source selection/scope differs")
    if partition == "validation":
        verify_checkpoint(root)
    qrels = {}
    lines = (root / "fast/qrels.tsv").read_text().splitlines()
    if (root / "fast/qrels.tsv").read_bytes() != (root / "deep/run/qrels.tsv").read_bytes():
        raise DatasetError("fast/deep qrels differ")
    for line in lines:
        q, zero, doc, grade = line.split()
        q, doc = int(q), int(doc)
        if q not in ids or zero != "0" or grade != "1" or doc in qrels.get(q, set()):
            raise DatasetError("invalid binary qrels")
        qrels.setdefault(q, set()).add(doc)
    official = {r.index: set(r.gold_docs_final_global) for r in records if r.index in ids}
    if qrels != official:
        raise DatasetError("cached qrels differ from unchanged official records")
    fast = load_trec(root / "fast/run.trec", ids, config.top_k)
    deep = load_trec(root / "deep/run/run.trec", ids, config.top_k)
    docs = {d.global_doc_id for r in records for d in r.docs_ko}
    rows = []
    for method, path in [(fast, "fast/per_query.jsonl"), (deep, "deep/run/per_query.jsonl")]:
        raw = read_rows(root / path)
        if [r["query_id"] for r in raw] != ids:
            raise DatasetError("metric selection/order differs")
        by_id = {r["query_id"]: r for r in raw}
        for q in ids:
            if not {h.doc_id for h in method[q]} <= docs:
                raise DatasetError("unknown candidate document")
            calculated = metrics(method[q], qrels[q])
            if any(abs(calculated[k] - by_id[q][k]) > 1e-12 for k in METRICS):
                raise DatasetError("source metric replay differs")
            if not math.isfinite(by_id[q]["retrieval_ms"]) or by_id[q]["retrieval_ms"] <= 0:
                raise DatasetError("invalid measured stage cost")
        rows.append(by_id)
    if any({h.doc_id for h in fast[q]} != {h.doc_id for h in deep[q]} for q in ids):
        raise DatasetError("fast/deep candidate pool differs")
    groups = training_groups(split) if partition == "train" else validation_groups(split)
    if plan["query_groups"] != {str(q): groups[q] for q in ids}:
        raise DatasetError("canonical source groups differ")
    return RankingBundle(partition, ids, {q: groups[q] for q in ids}, fast, deep, qrels, *rows)


def weight_metrics(bundle: RankingBundle, weight: float, offset: int) -> dict[int, dict]:
    return {
        q: metrics(blend_ranks(bundle.fast[q], bundle.deep[q], weight, offset), bundle.qrels[q])
        for q in bundle.ids
    }


def choose_weight(grid: dict[float, dict[int, dict]], ids: list[int], tolerance: float) -> float:
    """Selector only indexes its explicit training IDs; held-out rows never participate."""
    if not ids or len(ids) != len(set(ids)) or 0.0 not in grid:
        raise DatasetError("invalid weight-selection training rows")
    best, best_mean = 0.0, -math.inf
    for weight in sorted(grid):
        mean = float(np.mean([grid[weight][q]["ndcg_at_10"] for q in ids]))
        if not math.isfinite(mean):
            raise DatasetError("nonfinite selection metric")
        if mean > best_mean + tolerance:
            best, best_mean = weight, mean
    return best


def summarize_policy(bundle, rows, routed, config, name):
    if set(rows) != set(bundle.ids) or not routed <= set(bundle.ids):
        raise DatasetError("policy rows/routes differ from cohort")
    delta = [rows[q]["ndcg_at_10"] - bundle.fast_rows[q]["ndcg_at_10"] for q in bundle.ids]
    return {
        "name": name,
        "query_count": len(bundle.ids),
        "routed_count": len(routed),
        **{k: float(np.mean([rows[q][k] for q in bundle.ids])) for k in METRICS},
        "mean_ndcg_delta_vs_fast": float(np.mean(delta)),
        "paired_group_bootstrap_95_interval": grouped_interval(
            delta, [bundle.groups[q] for q in bundle.ids], config.seed, config.bootstrap_samples
        ),
        "gains_regressions_unchanged": [
            sum(v > 1e-12 for v in delta),
            sum(v < -1e-12 for v in delta),
            sum(abs(v) <= 1e-12 for v in delta),
        ],
        "modeled_source_stage_mean_ms": float(
            np.mean(
                [
                    bundle.fast_rows[q]["retrieval_ms"]
                    + (bundle.deep_rows[q]["retrieval_ms"] if q in routed else 0)
                    for q in bundle.ids
                ]
            )
        ),
        "cost_scope": "source fast + selected incremental deep; new fusion/decision overhead "
        "unmeasured and excluded, not zero; no live latency/GPU replay",
    }


def grouped_selection(bundle: RankingBundle, folds: dict[int, int], config: BlendConfig):
    if bundle.partition != "train" or set(folds) != set(bundle.ids):
        raise DatasetError("grouped selection requires canonical training rows only")
    seen = {}
    for q, fold in folds.items():
        group = bundle.groups[q]
        if group in seen and seen[group] != fold:
            raise DatasetError("source group crosses folds")
        seen[group] = fold
    if set(folds.values()) != set(range(config.n_splits)):
        raise DatasetError("incomplete grouped folds")
    grid = {w: weight_metrics(bundle, w, config.rank_offset) for w in config.candidate_weights}
    selected_rows, selected_weights, selections = {}, {}, []
    for fold in range(config.n_splits):
        train = sorted(q for q in bundle.ids if folds[q] != fold)
        held = sorted(q for q in bundle.ids if folds[q] == fold)
        weight = choose_weight(grid, train, config.selection_tolerance)
        selections.append(
            {
                "fold": fold,
                "selected_weight": weight,
                "selection_query_ids": train,
                "held_out_query_ids": held,
                "training_mean_by_weight": {
                    str(w): float(np.mean([grid[w][q]["ndcg_at_10"] for q in train])) for w in grid
                },
            }
        )
        for q in held:
            selected_rows[q], selected_weights[q] = grid[weight][q], weight
    final_weight = choose_weight(grid, bundle.ids, config.selection_tolerance)
    return (
        {
            "primary": summarize_policy(
                bundle,
                selected_rows,
                {q for q, w in selected_weights.items() if w > 0},
                config,
                "fold_selected_blend",
            ),
            "controls": [
                summarize_policy(bundle, bundle.fast_rows, set(), config, "fast"),
                summarize_policy(bundle, bundle.deep_rows, set(bundle.ids), config, "raw_deep"),
            ],
            "train_grid_in_sample": {
                str(w): float(np.mean([grid[w][q]["ndcg_at_10"] for q in bundle.ids])) for w in grid
            },
            "final_weight": final_weight,
            "scope": (
                "fold-local selection on training-only nDCG; "
                "held-out labels excluded from selection"
            ),
        },
        selections,
        [
            {
                "query_id": q,
                "fold": folds[q],
                "selected_weight": selected_weights[q],
                **selected_rows[q],
            }
            for q in bundle.ids
        ],
    )


def diagnose(bundle: RankingBundle) -> dict:
    gains, cases = [], []
    buckets = {k: [] for k in ["no_gold_in_pool", "partial_gold_in_pool", "all_gold_in_pool"]}
    top1_from_outside10, pushed_gold, ceilings = 0, 0, []
    for q in bundle.ids:
        a, b = [h.doc_id for h in bundle.fast[q]], [h.doc_id for h in bundle.deep[q]]
        relevant = bundle.qrels[q]
        found = set(a) & relevant
        bucket = (
            "no_gold_in_pool"
            if not found
            else ("all_gold_in_pool" if found == relevant else "partial_gold_in_pool")
        )
        gain = bundle.deep_rows[q]["ndcg_at_10"] - bundle.fast_rows[q]["ndcg_at_10"]
        gains.append(gain)
        buckets[bucket].append(q)
        ceilings.append(ndcg_at_k(sorted(a, key=lambda d: (d not in relevant, d)), relevant, 10))
        top1_from_outside10 += int(b[0] not in a[:10])
        pushed_gold += len((set(a[:10]) & relevant) - set(b[:10]))
        cases.append(
            {
                "query_id": q,
                "gain": gain,
                "fast_ndcg_at_10": bundle.fast_rows[q]["ndcg_at_10"],
                "deep_ndcg_at_10": bundle.deep_rows[q]["ndcg_at_10"],
                "first_positive_fast_rank": next(
                    (i for i, d in enumerate(a, 1) if d in relevant), None
                ),
                "first_positive_deep_rank": next(
                    (i for i, d in enumerate(b, 1) if d in relevant), None
                ),
            }
        )
    return {
        "query_count": len(bundle.ids),
        "group_count": len(set(bundle.groups.values())),
        "gains_regressions_unchanged": [
            sum(v > 1e-12 for v in gains),
            sum(v < -1e-12 for v in gains),
            sum(abs(v) <= 1e-12 for v in gains),
        ],
        "pool_coverage": {
            k: {
                "count": len(ids),
                "mean_fast_ndcg": float(np.mean([bundle.fast_rows[q]["ndcg_at_10"] for q in ids]))
                if ids
                else None,
                "mean_deep_ndcg": float(np.mean([bundle.deep_rows[q]["ndcg_at_10"] for q in ids]))
                if ids
                else None,
            }
            for k, ids in buckets.items()
        },
        "mean_pool_oracle_ndcg_ceiling": float(np.mean(ceilings)),
        "oracle_scope": "offline qrel-based pool upper bound; never a deployable policy/feature",
        "deep_top1_from_outside_fast_top10_count": top1_from_outside10,
        "positive_passages_displaced_from_top10_count": pushed_gold,
        "largest_regressions": sorted(cases, key=lambda r: (r["gain"], r["query_id"]))[:5],
        "largest_gains": sorted(cases, key=lambda r: (-r["gain"], r["query_id"]))[:5],
        "judgment_scope": "listed positives only; unjudged treated as zero for scoring, "
        "not established semantically irrelevant; no new manual gold labels",
    }


def validation_evaluation(bundle, weight, config, source_decisions):
    if bundle.partition != "validation" or weight not in config.candidate_weights:
        raise DatasetError("only one frozen train-selected weight may be evaluated on validation")
    blended = weight_metrics(bundle, weight, config.rank_offset)
    effective = set(bundle.ids) if weight > 0 else set()
    policies = [
        summarize_policy(bundle, bundle.fast_rows, set(), config, "fast"),
        summarize_policy(bundle, bundle.deep_rows, set(bundle.ids), config, "raw_deep"),
        summarize_policy(bundle, blended, effective, config, "frozen_blend"),
    ]
    for kind in ["logistic", "agreement", "random"]:
        ids = source_decisions[f"{kind}_budget_{config.routing_transfer_budget:g}"]
        if (
            len(ids) != int(len(bundle.ids) * config.routing_transfer_budget)
            or len(set(ids)) != len(ids)
            or not set(ids) <= set(bundle.ids)
        ):
            raise DatasetError("source transfer quota differs")
        routed = set(ids) if weight > 0 else set()
        rows = {q: blended[q] if q in routed else bundle.fast_rows[q] for q in bundle.ids}
        policy = summarize_policy(bundle, rows, routed, config, f"frozen_{kind}_transfer")
        policy["source_requested_calls"] = len(ids)
        policy["scope"] = "unchanged original routes, not a newly trained utility predictor"
        policies.append(policy)
    return {
        "selected_weight": weight,
        "policies": policies,
        "validation_weight_search": False,
        "router_refitted": False,
        "test_partition_evaluated": False,
    }


def write_inspection(root: Path, bundle: RankingBundle, records: list[QueryRecord], diagnostic):
    """Textual inspection remains in ignored local artifacts, never the public aggregate."""
    by_query = {r.index: r for r in records}
    docs = {d.global_doc_id: d for r in records for d in r.docs_ko}
    ids = [
        r["query_id"] for r in [*diagnostic["largest_regressions"], *diagnostic["largest_gains"]]
    ]
    _write_json(
        root / f"{bundle.partition}_inspection.json",
        [
            {
                "query_id": q,
                "query": by_query[q].query_ko,
                "query_insert_time": by_query[q].query_insert_time,
                "official_positive_ids": sorted(bundle.qrels[q]),
                "fast_top3": [docs[h.doc_id].model_dump() for h in bundle.fast[q][:3]],
                "deep_top3": [docs[h.doc_id].model_dump() for h in bundle.deep[q][:3]],
            }
            for q in ids
        ],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/m4_conservative_rankblend_company_policies.json"),
    )
    parser.add_argument(
        "--train-run", type=Path, default=Path("artifacts/m4/company_policies_train500_v1")
    )
    parser.add_argument(
        "--validation-run", type=Path, default=Path("artifacts/m4/router_validation495_v2")
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = BlendConfig.model_validate_json(args.config.read_text())
        if args.output.exists():
            raise DatasetError("output exists; choose a new directory")
        if sha256_file(ROUTER_CONFIG) != config.router_config_sha256:
            raise DatasetError("frozen fold configuration mismatch")
        router_config = RouterConfig.model_validate_json(ROUTER_CONFIG.read_text())
        if (router_config.query_count, router_config.seed, router_config.n_splits) != (
            config.train_query_count,
            config.seed,
            config.n_splits,
        ):
            raise DatasetError("fold setup differs from original grouped experiment")
        records = load_archive_records(args.archive)
        split = SplitManifest.model_validate_json(args.split.read_text())
        verify_split(split, records, ARCHIVE_SHA256)
        _, outcomes, _ = load_training_bundle(args.train_run, args.split, router_config)
        train = load_bundle(args.train_run, split, records, config, "train")
        folds = fold_assignment(outcomes, router_config)
        root = args.output
        root.mkdir(parents=True, exist_ok=False)
        _write_json(root / "config.json", config.model_dump())
        provenance = {
            **source_snapshot(Path.cwd()),
            "archive_sha256": ARCHIVE_SHA256,
            "split_sha256": sha256_file(args.split),
            "config_file_sha256": sha256_file(args.config),
        }
        _write_json(
            root / "plan.json",
            {
                "config": config.model_dump(),
                "provenance": provenance,
                "scope": (
                    "CPU replay; hypothesis designed after negative development validation; no test"
                ),
                "selection": (
                    "each fold uses other training groups only; final weight uses all train500"
                ),
                "validation_grid_evaluation_allowed": False,
                "new_router_fit": False,
                "test_partition_evaluated": False,
            },
        )
        cv, selection, oof = grouped_selection(train, folds, config)
        _write_json(root / "train_summary.json", cv)
        _write_json(root / "fold_selection.json", selection)
        write_rows(root / "train_oof.jsonl", oof)
        _write_json(
            root / "final_selection.json",
            {
                "weight": cv["final_weight"],
                "scope": "all selected train500 only; frozen before loading validation outcomes",
                "validation_outcomes_used": False,
            },
        )
        checkpoint_names = [p.name for p in root.iterdir() if p.is_file()]
        checkpoint = {n: sha256_file(root / n) for n in checkpoint_names}
        _write_json(root / "train_checkpoint.json", checkpoint)
        # Only now load/evaluate the validation source. Never sweep its candidate weights.
        validation_manifest = verify_source(args.validation_run, config.validation_manifest_sha256)
        if validation_manifest["split_sha256"] != provenance["split_sha256"]:
            raise DatasetError("validation split differs")
        val = load_bundle(args.validation_run, split, records, config, "validation")
        if set(train.groups.values()) & set(val.groups.values()):
            raise DatasetError("train/validation groups overlap")
        decisions = json.loads((args.validation_run / "decisions.json").read_text())
        val_result = validation_evaluation(val, cv["final_weight"], config, decisions)
        diagnostics = {b.partition: diagnose(b) for b in [train, val]}
        for b in [train, val]:
            write_inspection(root, b, records, diagnostics[b.partition])
        summary = {
            "train_cv": cv,
            "validation": val_result,
            "diagnostics": diagnostics,
            "new_neural_inference": False,
            "new_router_fit": False,
            "new_training_targets": False,
            "test_partition_evaluated": False,
            "interval_scope": "source-group resamples, fixed selected folds/weights without refit",
            "hypothesis_scope": (
                "new hypothesis motivated by previously observed development validation"
            ),
        }
        for n, digest in checkpoint.items():
            if sha256_file(root / n) != digest:
                raise DatasetError("train selection changed after validation")
        if (
            source_snapshot(Path.cwd())["source_snapshot_sha256"]
            != provenance["source_snapshot_sha256"]
        ):
            raise DatasetError("source changed during experiment")
        _write_json(root / "summary.json", summary)
        _write_json(
            root / "manifest.json",
            {
                **provenance,
                "train_manifest_sha256": config.train_manifest_sha256,
                "validation_manifest_sha256": config.validation_manifest_sha256,
                "files_sha256": {
                    p.name: sha256_file(p) for p in sorted(root.iterdir()) if p.is_file()
                },
            },
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"rank blend failed: {exc}; saved evidence retained", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"weight": cv["final_weight"], "oof_primary": cv["primary"], "validation": val_result},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
