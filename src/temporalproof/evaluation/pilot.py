"""M4 small train-only utility pilot; no fitted router or held-out evaluation."""

import argparse
import gc
import importlib.metadata
import json
import sys
import weakref
from collections.abc import Callable, Sequence
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
from temporalproof.data.timelyqa import DatasetError, QueryRecord
from temporalproof.evaluation.baseline import (
    BaselineConfig,
    _write_json,
    run_baseline,
    source_snapshot,
)
from temporalproof.evaluation.reranking import RerankingConfig, load_candidate_pool, run_reranking
from temporalproof.retrieval.bm25 import BM25Retriever, Hit
from temporalproof.retrieval.dense import corpus_cache
from temporalproof.retrieval.encoder import DEFAULT_MODEL, MODEL_ID, REVISION, E5Encoder
from temporalproof.retrieval.fusion import reciprocal_rank_fusion
from temporalproof.retrieval.reranker import (
    KOREAN,
    KOREAN_MODEL_ID,
    KOREAN_REVISION,
    CrossEncoder,
    verify_model,
)
from temporalproof.routing.features import FEATURE_VERSION, PreDeepFeatures, extract_features
from temporalproof.routing.targets import (
    build_train_outcomes,
    diagnostic_policies,
    feature_slices,
    training_groups,
)


class PilotConfig(BaseModel):
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
    cost_penalties_ndcg_per_second: tuple[float, ...]
    query_route_budgets: tuple[float, ...]
    disagreement_threshold: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def train_only(self) -> Self:
        if self.baseline.partition != "train" or self.baseline.method != "hybrid_rrf":
            raise ValueError("pilot requires train-only hybrid retrieval")
        for values, upper in [
            (self.cost_penalties_ndcg_per_second, None),
            (self.query_route_budgets, 1),
        ]:
            if (
                not values
                or list(values) != sorted(set(values))
                or min(values) < 0
                or (upper is not None and max(values) > upper)
            ):
                raise ValueError(
                    "penalties/budgets must be sorted unique nonnegative values in range"
                )
        return self


def write_rows(path: Path, rows: Sequence[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    )


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def collect_fast(
    records: Sequence[QueryRecord],
    split: SplitManifest,
    config: PilotConfig,
    root: Path,
    provenance: dict,
    search: Callable[[str, int], tuple[list[Hit], list[Hit]]],
    build_ms: float = 0,
) -> dict[int, PreDeepFeatures]:
    """Search/extraction callbacks see no QueryRecord/qrels; IDs are joins only."""
    selected = select_queries(split, "train", config.baseline.query_limit, config.baseline.seed)
    by_doc = {doc.global_doc_id: doc for record in records for doc in record.docs_ko}
    features, timings = {}, {}

    def query_search(query_id, text, cutoff, inserted):
        sparse, dense = search(text, cutoff)
        hybrid = reciprocal_rank_fusion([sparse, dense], cutoff, config.rrf_offset)
        started = perf_counter()
        row = extract_features(text, inserted, sparse, dense, hybrid, by_doc)
        elapsed = (perf_counter() - started) * 1000
        if query_id in selected:
            features[query_id] = row
            timings[query_id] = elapsed
        return hybrid

    run_baseline(
        records,
        split,
        config.baseline,
        root / "fast",
        provenance,
        query_search=query_search,
        index_build_ms=build_ms,
        timing_scope=(
            "warm sequential hybrid plus pre-deep feature extraction; synchronized GPU query "
            "encoding, CPU BM25/FAISS/RRF; concurrency 1; "
            "excludes offline build/metrics/files/tracking"
        ),
    )
    write_rows(
        root / "features.jsonl",
        [{"query_id": q, "features": features[q].model_dump()} for q in selected],
    )
    write_rows(
        root / "feature_timing.jsonl", [{"query_id": q, "feature_ms": timings[q]} for q in selected]
    )
    manifest = json.loads((root / "fast/manifest.json").read_text())
    manifest["feature_version"] = FEATURE_VERSION
    manifest["pre_deep_features_sha256"] = sha256_file(root / "features.jsonl")
    _write_json(root / "fast/manifest.json", manifest)
    return features


def collect_fast_gpu(records, split, config, root, provenance, encoder_path, cache_path):
    """No GPU model reference escapes this scope before loading the cross-encoder."""
    documents = [doc for record in records for doc in record.docs_ko]
    encoder = E5Encoder(
        encoder_path, batch_size=config.encoder_batch_size, max_length=config.encoder_max_length
    )
    index, cache = corpus_cache(documents, encoder, cache_path)
    started = perf_counter()
    sparse = BM25Retriever(
        documents, k1=config.baseline.k1, b=config.baseline.b, epsilon=config.baseline.epsilon
    )
    build_ms = (perf_counter() - started) * 1000 + cache["faiss_build_ms"]

    def search(text, cutoff):
        return sparse.search(text, cutoff), index.search(
            encoder.encode([text], kind="query")[0], cutoff
        )

    collect_fast(
        records, split, config, root, {**provenance, "encoder_cache": cache}, search, build_ms
    )
    manifest = json.loads((root / "fast/manifest.json").read_text())
    manifest["encoder"] = encoder.profile()
    _write_json(root / "fast/manifest.json", manifest)
    return weakref.ref(encoder.model)


def log_pilot(root: Path, summary: dict) -> None:
    from temporalproof.tracking.local import local_client, log_run_directory

    ids = {"fast": log_run_directory(root / "fast"), "deep": log_run_directory(root / "deep/run")}
    summary["mlflow_run_ids"] = ids
    _write_json(root / "summary.json", summary)
    client = local_client()
    for run_id in ids.values():
        client.set_tag(run_id, "temporalproof.phase", "M4_train_only_pilot")
    run_id = ids["deep"]
    client.log_metric(run_id, "pilot.mean_gain", summary["utility"]["mean_gain"])
    for key in ["beneficial_count", "harmful_count", "unchanged_count"]:
        client.log_metric(run_id, f"pilot.{key}", summary["utility"][key])
    for name in [
        "config.json",
        "plan.json",
        "pre_deep_checkpoint.json",
        "features.jsonl",
        "outcomes.jsonl",
        "summary.json",
    ]:
        client.log_artifact(run_id, str(root / name), artifact_path="pilot")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/m4_train_pilot_company_policies.json"),
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/embeddings/e5_small"))
    parser.add_argument("--reranker", type=Path, default=KOREAN.directory)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = PilotConfig.model_validate_json(args.config.read_text())
        if args.output.exists():
            raise DatasetError("output exists; choose a new pilot directory")
        verify_model(args.reranker, KOREAN)
        records = load_archive_records(args.archive)
        split = SplitManifest.model_validate_json(args.split.read_text())
        verify_split(split, records, ARCHIVE_SHA256)
        selected = select_queries(split, "train", config.baseline.query_limit, config.baseline.seed)
        groups = training_groups(split)
        provenance = {
            "dataset_revision": DATA_REVISION,
            "archive_sha256": ARCHIVE_SHA256,
            "member": MEMBER,
            "split_sha256": sha256_file(args.split),
            "pilot_config_sha256": sha256_file(args.config),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["torch", "transformers", "tokenizers", "faiss-cpu"]
            },
            **source_snapshot(Path.cwd()),
        }
        root = args.output
        root.mkdir(parents=True, exist_ok=False)
        _write_json(root / "config.json", config.model_dump())
        _write_json(
            root / "plan.json",
            {
                "selected_query_ids": selected,
                "query_groups": {str(q): groups[q] for q in selected},
                "config": config.model_dump(),
                "provenance": provenance,
                "scope": "train-only pilot; no fitted router or held-out evaluation",
                "feature_version": FEATURE_VERSION,
                "test_partition_evaluated": False,
                "validation_partition_evaluated": False,
            },
        )
        print(
            f"M4 pilot: {len(selected)} frozen train queries; hybrid + pre-deep features",
            flush=True,
        )
        encoder_model = collect_fast_gpu(
            records, split, config, root, provenance, args.encoder, args.cache
        )
        import torch

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated()
        if encoder_model() is not None:
            raise DatasetError("encoder model remains referenced before reranker load")
        checkpoint = {
            "features_sha256": sha256_file(root / "features.jsonl"),
            "fast_manifest_sha256": sha256_file(root / "fast/manifest.json"),
            "plan_sha256": sha256_file(root / "plan.json"),
            "between_models_allocated_bytes": allocated,
            "encoder_model_collected": True,
            "allocation_scope": (
                "allocator state may include backend workspaces; model removal verified "
                "by weak reference, not inferred from a zero allocation counter"
            ),
            "scope": "written before reranker inference; features contain no outcomes",
        }
        _write_json(root / "pre_deep_checkpoint.json", checkpoint)
        deep_config = RerankingConfig(
            baseline=config.baseline.model_copy(
                update={"method": "fixed_cross_encoder", "name": f"deep_{config.baseline.name}"}
            ),
            model_id=config.reranker_model_id,
            revision=config.reranker_revision,
            batch_size=config.reranker_batch_size,
            max_length=config.reranker_max_length,
            candidate_manifest_sha256=checkpoint["fast_manifest_sha256"],
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
        print("M4 pilot: fixed Korean reranker on the frozen train pool", flush=True)
        comparison = run_reranking(
            records, split, deep_config, pool, scorer, root / "deep", provenance
        )
        if sha256_file(root / "features.jsonl") != checkpoint["features_sha256"]:
            raise DatasetError("pre-deep features changed during reranking")
        features = {
            row["query_id"]: PreDeepFeatures.model_validate(row["features"])
            for row in read_rows(root / "features.jsonl")
        }
        outcomes = build_train_outcomes(
            split,
            features,
            read_rows(root / "fast/per_query.jsonl"),
            read_rows(root / "deep/run/per_query.jsonl"),
            config.cost_penalties_ndcg_per_second,
        )
        write_rows(root / "outcomes.jsonl", [row.model_dump() for row in outcomes])
        gains = np.array([r.gain for r in outcomes])
        summary = {
            "config": config.model_dump(),
            "query_count": len(selected),
            "group_count": len({groups[q] for q in selected}),
            "fast_summary": comparison["fast_summary"],
            "deep_summary": comparison["reranker_summary"],
            "paired_deltas": comparison["paired_deltas_vs_hybrid"],
            "utility": {
                "mean_gain": float(gains.mean()),
                "beneficial_count": int((gains > 1e-12).sum()),
                "harmful_count": int((gains < -1e-12).sum()),
                "unchanged_count": int((abs(gains) <= 1e-12).sum()),
                "positive_labels_by_lambda": {
                    str(w): sum(r.cost_sensitive_positive[str(w)] for r in outcomes)
                    for w in config.cost_penalties_ndcg_per_second
                },
                "no_gold_in_pool": sum(r.candidate_recall_at_100 == 0 for r in outcomes),
            },
            "feature_timing_mean_ms": float(
                np.mean([r["feature_ms"] for r in read_rows(root / "feature_timing.jsonl")])
            ),
            "policies": diagnostic_policies(
                outcomes,
                features,
                budgets=config.query_route_budgets,
                disagreement_threshold=config.disagreement_threshold,
                seed=config.baseline.seed,
            ),
            "feature_slices": feature_slices(outcomes, features, config.disagreement_threshold),
            "target_definition": (
                "gain > lambda * incremental_rerank_ms / 1000 + 1e-12; "
                "lambda in nDCG/second; training only"
            ),
            "router_training_targets_generated": True,
            "fitted_router": False,
            "test_partition_evaluated": False,
            "validation_partition_evaluated": False,
            "cost_scope": (
                "fast includes feature extraction; deep is incremental only; "
                "policy sums are offline stage-cost estimates, not live p95"
            ),
            "mlflow_run_ids": {},
        }
        _write_json(root / "summary.json", summary)
        if args.mlflow:
            log_pilot(root, summary)
        _write_json(
            root / "manifest.json",
            {
                **provenance,
                "pre_deep_checkpoint": checkpoint,
                "files_sha256": {
                    str(p.relative_to(root)): sha256_file(p)
                    for p in sorted(root.rglob("*"))
                    if p.is_file()
                },
            },
        )
        if args.mlflow:
            from temporalproof.tracking.local import local_client

            local_client().log_artifact(
                summary["mlflow_run_ids"]["deep"],
                str(root / "manifest.json"),
                artifact_path="pilot",
            )
    except (OSError, ValueError, KeyError) as exc:
        message = (
            str(exc)
            if isinstance(exc, DatasetError)
            else "input/model/evidence/tracking validation failed"
        )
        print(f"pilot failed: {message}; saved evidence remains intact", file=sys.stderr)
        return 1
    print(json.dumps(summary["utility"], sort_keys=True))
    print(f"Pilot artifacts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
