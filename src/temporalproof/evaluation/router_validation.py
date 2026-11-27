"""Frozen train-fitted router evaluation on the complete development validation split."""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from temporalproof.data.source import (
    ARCHIVE_SHA256,
    DEFAULT_ARCHIVE,
    MEMBER,
    load_archive_records,
    sha256_file,
)
from temporalproof.data.source import REVISION as DATA_REVISION
from temporalproof.data.splits import DEFAULT_SPLIT, SplitManifest, select_queries, verify_split
from temporalproof.data.timelyqa import DatasetError
from temporalproof.evaluation.baseline import BaselineConfig, _write_json, source_snapshot
from temporalproof.evaluation.pilot import collect_fast_gpu, read_rows, write_rows
from temporalproof.evaluation.reranking import RerankingConfig, load_candidate_pool, run_reranking
from temporalproof.retrieval.encoder import DEFAULT_MODEL, MODEL_ID, REVISION
from temporalproof.retrieval.reranker import (
    KOREAN,
    KOREAN_MODEL_ID,
    KOREAN_REVISION,
    CrossEncoder,
    verify_model,
)
from temporalproof.routing.cross_validation import control_scores, grouped_interval
from temporalproof.routing.features import FEATURE_VERSION, PreDeepFeatures
from temporalproof.routing.learned import RouterModel, feature_names, predict_scores
from temporalproof.routing.targets import training_groups


class ValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    baseline: BaselineConfig
    encoder_model_id: Literal[MODEL_ID]
    encoder_revision: Literal[REVISION]
    encoder_batch_size: int = Field(strict=True, ge=1)
    encoder_max_length: int = Field(strict=True, ge=1, le=512)
    rrf_offset: int = Field(strict=True, ge=1)
    reranker_model_id: Literal[KOREAN_MODEL_ID]
    reranker_revision: Literal[KOREAN_REVISION]
    reranker_batch_size: int = Field(strict=True, ge=1)
    reranker_max_length: int = Field(strict=True, ge=1, le=512)
    training_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    final_model_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    training_config_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    query_route_budgets: tuple[float, ...]
    primary_query_budget: float = Field(ge=0, le=1)
    probability_threshold: float = Field(ge=0, le=1)
    disagreement_threshold: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validation_only(self) -> Self:
        if self.baseline.partition != "validation" or self.baseline.method != "hybrid_rrf":
            raise ValueError("frozen evaluation requires validation-only hybrid retrieval")
        if (
            list(self.query_route_budgets) != sorted(set(self.query_route_budgets))
            or not self.query_route_budgets
            or min(self.query_route_budgets) < 0
            or max(self.query_route_budgets) > 1
            or self.primary_query_budget not in self.query_route_budgets
        ):
            raise ValueError("invalid predeclared budgets")
        return self


def validation_ids(split: SplitManifest, config: ValidationConfig) -> list[int]:
    if config.baseline.query_limit != len(split.query_ids("validation")):
        raise DatasetError("evaluation must cover the complete validation partition")
    return select_queries(split, "validation", config.baseline.query_limit, config.baseline.seed)


def validation_groups(split: SplitManifest) -> dict[int, str]:
    return {
        q: hashlib.sha256(json.dumps(sorted(group)).encode()).hexdigest()
        for group in split.groups["validation"]
        for q in group
    }


def load_frozen_model(root: Path, split: SplitManifest, config: ValidationConfig) -> RouterModel:
    """Pin the training evidence and scorer implementation before reading held-out outcomes."""
    for name, expected in [
        ("manifest.json", config.training_manifest_sha256),
        ("final_model.json", config.final_model_sha256),
        ("config.json", config.training_config_sha256),
    ]:
        if sha256_file(root / name) != expected:
            raise DatasetError(f"frozen training checksum mismatch: {name}")
    manifest = json.loads((root / "manifest.json").read_text())
    expected_files = {
        "config.json",
        "data_manifest.json",
        "final_model.json",
        "fold_models.json",
        "oof_predictions.jsonl",
        "plan.json",
        "summary.json",
    }
    if set(manifest["files_sha256"]) != expected_files:
        raise DatasetError("unexpected training evidence list")
    for name, digest in manifest["files_sha256"].items():
        if sha256_file(root / name) != digest:
            raise DatasetError("training evidence checksum mismatch")
    cfg = json.loads((root / "config.json").read_text())
    if cfg != manifest["config"] or (
        cfg["primary_feature_set"] != "full"
        or cfg["primary_penalty"] != 0
        or cfg["primary_query_budget"] != config.primary_query_budget
        or cfg["probability_threshold"] != config.probability_threshold
        or cfg["disagreement_threshold"] != config.disagreement_threshold
        or cfg["query_route_budgets"] != list(config.query_route_budgets)
        or cfg["seed"] != config.baseline.seed
    ):
        raise DatasetError("validation policy differs from frozen primary training policy")
    plan = json.loads((root / "plan.json").read_text())
    canonical = select_queries(split, "train", cfg["query_count"], cfg["seed"])
    groups = training_groups(split)
    if plan["group_by_query"] != {str(q): groups[q] for q in canonical}:
        raise DatasetError("model training IDs/groups differ from canonical train selection")
    if set(canonical) & set(validation_ids(split, config)) or (
        {groups[q] for q in canonical} & set(validation_groups(split).values())
    ):
        raise DatasetError("training and validation overlap")
    summary = json.loads((root / "summary.json").read_text())
    if summary["validation_partition_evaluated"] or summary["test_partition_evaluated"]:
        raise DatasetError("training evidence contains held-out evaluation")
    repo = Path(__file__).resolve().parents[3]
    for name in ["features.py", "learned.py"]:
        relative = f"src/temporalproof/routing/{name}"
        if sha256_file(repo / relative) != manifest["source_files_sha256"][relative]:
            raise DatasetError("feature/scoring code differs from frozen training source")
    model = RouterModel.model_validate(json.loads((root / "final_model.json").read_text())["model"])
    if model.feature_names != feature_names("full"):
        raise DatasetError("frozen primary model must use the original full feature set")
    return model


def make_decisions(features: dict[int, PreDeepFeatures], model: RouterModel, config):
    """No labels, deep outputs or actual deep costs enter this function."""
    if not features:
        raise DatasetError("empty inference cohort")
    ids = sorted(features)
    for _ in range(3):
        predict_scores(model, [features[ids[0]]])
    predictions = []
    for q in ids:
        started = perf_counter()
        score = float(predict_scores(model, [features[q]])[0])
        elapsed = (perf_counter() - started) * 1000
        predictions.append({"query_id": q, "score": score, "router_predict_ms": elapsed})
    decisions = {"always_fast": [], "always_deep": ids}
    decisions["low_agreement"] = [
        q for q in ids if features[q].top10_overlap_fraction < config.disagreement_threshold
    ]
    decisions["year_and_low_agreement"] = [
        q for q in decisions["low_agreement"] if features[q].explicit_year_count > 0
    ]
    decisions["logistic_threshold"] = [
        p["query_id"] for p in predictions if p["score"] >= config.probability_threshold
    ]
    for kind in ["logistic", "random", "agreement"]:
        scores = (
            predictions
            if kind == "logistic"
            else control_scores(features, dict.fromkeys(ids, 0), config.baseline.seed, kind)
        )
        ranked = sorted(scores, key=lambda p: (-p["score"], p["query_id"]))
        for budget in config.query_route_budgets:
            decisions[f"{kind}_budget_{budget:g}"] = sorted(
                p["query_id"] for p in ranked[: int(np.floor(len(ids) * budget))]
            )
    return predictions, decisions


def checkpoint(root: Path, allocated_bytes: int) -> dict:
    result = {
        "files_sha256": {
            name: sha256_file(root / name)
            for name in [
                "config.json",
                "plan.json",
                "frozen_model.json",
                "features.jsonl",
                "fast/manifest.json",
                "predictions.jsonl",
                "decisions.json",
            ]
        },
        "feature_version": FEATURE_VERSION,
        "encoder_model_collected": True,
        "between_models_allocated_bytes": allocated_bytes,
        "scope": "decisions frozen before deep inference; no validation training targets",
    }
    _write_json(root / "pre_deep_checkpoint.json", result)
    return result


def verify_checkpoint(root: Path) -> None:
    evidence = json.loads((root / "pre_deep_checkpoint.json").read_text())
    for name, digest in evidence["files_sha256"].items():
        if (
            name
            not in {
                "config.json",
                "plan.json",
                "frozen_model.json",
                "features.jsonl",
                "fast/manifest.json",
                "predictions.jsonl",
                "decisions.json",
            }
            or sha256_file(root / name) != digest
        ):
            raise DatasetError("pre-deep decision checkpoint changed")
    if len(evidence["files_sha256"]) != 7:
        raise DatasetError("incomplete pre-deep checkpoint")


def evaluate(root: Path, split: SplitManifest, config: ValidationConfig) -> dict:
    """Join already frozen decisions to held-out metrics; never construct train targets."""
    verify_checkpoint(root)
    ids = validation_ids(split, config)
    groups = validation_groups(split)
    fast_rows = read_rows(root / "fast/per_query.jsonl")
    deep_rows = read_rows(root / "deep/run/per_query.jsonl")
    if [r["query_id"] for r in fast_rows] != ids or [r["query_id"] for r in deep_rows] != ids:
        raise DatasetError("validation metric rows differ from frozen selection")
    fast = {r["query_id"]: r for r in fast_rows}
    deep = {r["query_id"]: r for r in deep_rows}
    if (root / "fast/qrels.tsv").read_bytes() != (root / "deep/run/qrels.tsv").read_bytes():
        raise DatasetError("validation qrels changed during reranking")
    if any(fast[q]["recall_at_100"] != deep[q]["recall_at_100"] for q in ids):
        raise DatasetError("validation candidate coverage changed")
    features = {
        r["query_id"]: PreDeepFeatures.model_validate(r["features"])
        for r in read_rows(root / "features.jsonl")
    }
    model = RouterModel.model_validate_json((root / "frozen_model.json").read_text())
    replay_predictions, replay_decisions = make_decisions(features, model, config)
    decisions = json.loads((root / "decisions.json").read_text())
    predictions = read_rows(root / "predictions.jsonl")
    if (
        decisions != replay_decisions
        or [p["score"] for p in predictions] != [p["score"] for p in replay_predictions]
        or set(features) != set(ids)
        or [p["query_id"] for p in predictions] != sorted(ids)
    ):
        raise DatasetError("frozen decisions/scores do not replay from the model/features")
    times = {p["query_id"]: p["router_predict_ms"] for p in predictions}
    metric_names = ["ndcg_at_10", "recall_at_10", "recall_at_20", "recall_at_100"]
    gains = np.array([deep[q]["ndcg_at_10"] - fast[q]["ndcg_at_10"] for q in ids])
    policies = []
    quality = {}
    group_ids = [groups[q] for q in ids]
    for name, routed_ids in decisions.items():
        routed = set(routed_ids)
        values = np.array([(deep if q in routed else fast)[q]["ndcg_at_10"] for q in ids])
        quality[name] = values
        delta = values - np.array([fast[q]["ndcg_at_10"] for q in ids])
        costs = [
            fast[q]["retrieval_ms"]
            + (deep[q]["retrieval_ms"] if q in routed else 0)
            + (times[q] if name.startswith("logistic") else 0)
            for q in ids
        ]
        policies.append(
            {
                "name": name,
                "routed_count": len(routed),
                "routed_fraction": len(routed) / len(ids),
                **{
                    m: float(np.mean([(deep if q in routed else fast)[q][m] for q in ids]))
                    for m in metric_names
                },
                "mean_gain_vs_fast": float(delta.mean()),
                "paired_group_bootstrap_95_interval": grouped_interval(
                    delta, group_ids, config.baseline.seed, config.baseline.bootstrap_samples
                ),
                "beneficial_calls": sum(
                    int(q in routed and gains[i] > 1e-12) for i, q in enumerate(ids)
                ),
                "harmful_calls": sum(
                    int(q in routed and gains[i] < -1e-12) for i, q in enumerate(ids)
                ),
                "modeled_mean_with_router_ms": float(np.mean(costs)),
            }
        )
    primary_name = f"logistic_budget_{config.primary_query_budget:g}"
    matched = []
    for kind in ["random", "agreement"]:
        name = f"{kind}_budget_{config.primary_query_budget:g}"
        delta = quality[primary_name] - quality[name]
        matched.append(
            {
                "control": name,
                "mean_ndcg_delta": float(delta.mean()),
                "paired_group_bootstrap_95_interval": grouped_interval(
                    delta, group_ids, config.baseline.seed, config.baseline.bootstrap_samples
                ),
            }
        )
    return {
        "query_count": len(ids),
        "group_count": len(set(group_ids)),
        "primary": next(p for p in policies if p["name"] == primary_name),
        "primary_matched_comparisons": matched,
        "policies": policies,
        "router_predict_mean_ms": float(np.mean(list(times.values()))),
        "deep_gains_regressions_unchanged": [
            int((gains > 1e-12).sum()),
            int((gains < -1e-12).sum()),
            int((abs(gains) <= 1e-12).sum()),
        ],
        "no_gold_in_pool": sum(fast[q]["recall_at_100"] == 0 for q in ids),
        "router_training_targets_generated": False,
        "router_refitted": False,
        "validation_partition_evaluated": True,
        "test_partition_evaluated": False,
        "decision_replay_verified": True,
        "scope": "development validation includes previously used validation100; not final test",
        "budget_scope": "whole validation cohort quota, single frozen model; not online SLA",
        "interval_scope": (
            f"{config.baseline.bootstrap_samples} source-group resamples "
            "with fixed predictions, no refitting"
        ),
        "cost_scope": "offline sums of fast incl features + selected incremental deep + warm CPU "
        "router; excludes quota sorting, parsing/loading/HTTP; not live p95",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/m4_frozen_validation_company_policies.json"),
    )
    parser.add_argument(
        "--training-run", type=Path, default=Path("artifacts/m4/router_train500_groupcv_v1")
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/embeddings/e5_small"))
    parser.add_argument("--reranker", type=Path, default=KOREAN.directory)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = ValidationConfig.model_validate_json(args.config.read_text())
        if args.output.exists():
            raise DatasetError("output exists; choose a new validation directory")
        split = SplitManifest.model_validate_json(args.split.read_text())
        model = load_frozen_model(args.training_run, split, config)
        verify_model(args.reranker, KOREAN)
        records = load_archive_records(args.archive)
        verify_split(split, records, ARCHIVE_SHA256)
        # Match the exact train collection protocol except partition/name/query count.
        original = json.loads((args.training_run / "data_manifest.json").read_text())
        if original["archive_sha256"] != ARCHIVE_SHA256 or original["split_sha256"] != sha256_file(
            args.split
        ):
            raise DatasetError("validation archive/split differs from training")
        train_config = json.loads(
            Path("configs/experiments/m4_train500_company_policies.json").read_text()
        )
        if (
            sha256_file(Path("configs/experiments/m4_train500_company_policies.json"))
            != original["pilot_config_sha256"]
        ):
            raise DatasetError("original retrieval config checksum mismatch")
        actual = config.model_dump(mode="json")
        for k, value in train_config.items():
            if k == "baseline":
                for field, setting in value.items():
                    if (
                        field not in {"name", "partition", "query_limit"}
                        and actual[k][field] != setting
                    ):
                        raise DatasetError("retrieval settings differ from training")
            elif k != "cost_penalties_ndcg_per_second" and actual[k] != value:
                raise DatasetError("encoder/reranker/policy settings differ from training")
        root = args.output
        root.mkdir(parents=True, exist_ok=False)
        _write_json(root / "config.json", actual)
        _write_json(root / "frozen_model.json", model.model_dump())
        provenance = {
            "archive_sha256": ARCHIVE_SHA256,
            "dataset_revision": DATA_REVISION,
            "member": MEMBER,
            "split_sha256": sha256_file(args.split),
            "validation_config_sha256": sha256_file(args.config),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["torch", "transformers", "tokenizers", "faiss-cpu"]
            },
            **source_snapshot(Path.cwd()),
        }
        ids = validation_ids(split, config)
        _write_json(
            root / "plan.json",
            {
                "selected_query_ids": ids,
                "query_groups": validation_groups(split),
                "provenance": provenance,
                "training_manifest_sha256": config.training_manifest_sha256,
                "final_model_sha256": config.final_model_sha256,
                "primary": "full / penalty 0 / query cap 0.1",
                "router_refit_allowed": False,
                "test_partition_evaluated": False,
            },
        )
        print(f"Frozen validation: {len(ids)} queries; collecting hybrid/features", flush=True)
        encoder_ref = collect_fast_gpu(
            records, split, config, root, provenance, args.encoder, args.cache
        )
        import torch

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if encoder_ref() is not None:
            raise DatasetError("encoder remains referenced before reranker")
        features = {
            r["query_id"]: PreDeepFeatures.model_validate(r["features"])
            for r in read_rows(root / "features.jsonl")
        }
        predictions, decisions = make_decisions(features, model, config)
        write_rows(root / "predictions.jsonl", predictions)
        _write_json(root / "decisions.json", decisions)
        evidence = checkpoint(root, torch.cuda.memory_allocated())
        deep_config = RerankingConfig(
            baseline=config.baseline.model_copy(
                update={"method": "fixed_cross_encoder", "name": f"deep_{config.baseline.name}"}
            ),
            model_id=config.reranker_model_id,
            revision=config.reranker_revision,
            batch_size=config.reranker_batch_size,
            max_length=config.reranker_max_length,
            candidate_manifest_sha256=evidence["files_sha256"]["fast/manifest.json"],
        )
        _write_json(root / "reranking_config.json", deep_config.model_dump())
        pool = load_candidate_pool(
            root / "fast", records, split, deep_config, provenance["split_sha256"]
        )
        scorer = CrossEncoder(
            args.reranker,
            spec=KOREAN,
            batch_size=config.reranker_batch_size,
            max_length=config.reranker_max_length,
        )
        print("Decisions checkpointed; fixed Korean reranker on validation top100", flush=True)
        run_reranking(records, split, deep_config, pool, scorer, root / "deep", provenance)
        load_frozen_model(args.training_run, split, config)
        if (
            source_snapshot(Path.cwd())["source_snapshot_sha256"]
            != provenance["source_snapshot_sha256"]
        ):
            raise DatasetError("source changed during validation")
        summary = evaluate(root, split, config)
        _write_json(root / "summary.json", summary)
        _write_json(
            root / "manifest.json",
            {
                **provenance,
                "files_sha256": {
                    str(p.relative_to(root)): sha256_file(p)
                    for p in sorted(root.rglob("*"))
                    if p.is_file()
                },
            },
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"frozen validation failed: {exc}; saved evidence retained", file=sys.stderr)
        return 1
    print(json.dumps(summary["primary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
