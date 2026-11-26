"""M2 fixed validation comparison: sparse, dense, hybrid and insertion heuristic."""

import argparse
import importlib.metadata
import json
import sys
from collections.abc import Sequence
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from temporalproof.data.source import (
    ARCHIVE_SHA256,
    DEFAULT_ARCHIVE,
    MEMBER,
    load_archive_records,
    sha256_file,
)
from temporalproof.data.source import (
    REVISION as DATA_REVISION,
)
from temporalproof.data.splits import DEFAULT_SPLIT, SplitManifest, verify_split
from temporalproof.data.timelyqa import DatasetError
from temporalproof.evaluation.baseline import BaselineConfig, run_baseline, source_snapshot
from temporalproof.retrieval.bm25 import BM25Retriever
from temporalproof.retrieval.dense import corpus_cache
from temporalproof.retrieval.encoder import DEFAULT_MODEL, MODEL_ID, REVISION, E5Encoder
from temporalproof.retrieval.fusion import insertion_time_rerank, reciprocal_rank_fusion

METHODS = ("bm25", "dense", "hybrid_rrf", "hybrid_insertion_time")


class ComparisonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    baseline: BaselineConfig
    model_id: Literal[MODEL_ID]
    revision: Literal[REVISION]
    batch_size: int = Field(strict=True, ge=1)
    max_length: int = Field(strict=True, ge=1, le=512)
    rrf_offset: int = Field(strict=True, ge=1)
    insertion_weight: float = Field(ge=0)


def paired_comparisons(
    baseline: Sequence[dict],
    candidate: Sequence[dict],
    seed: int,
    samples: int,
) -> dict:
    reference = {row["query_id"]: row for row in baseline}
    other = {row["query_id"]: row for row in candidate}
    if (
        len(reference) != len(baseline)
        or len(other) != len(candidate)
        or reference.keys() != other.keys()
    ):
        raise ValueError("paired comparisons require matching, unique query IDs")
    rng = np.random.default_rng(seed)
    result = {}
    for metric in ["ndcg_at_10", "recall_at_10", "recall_at_20", "recall_at_100"]:
        deltas = np.array([other[q][metric] - reference[q][metric] for q in sorted(reference)])
        means = rng.choice(deltas, size=(samples, len(deltas))).mean(axis=1)
        order = sorted(reference, key=lambda q: (other[q][metric] - reference[q][metric], q))
        result[metric] = {
            "mean_delta": float(deltas.mean()),
            "paired_query_bootstrap_95_percent_interval": np.quantile(
                means, [0.025, 0.975]
            ).tolist(),
            "improved": int((deltas > 1e-12).sum()),
            "regressed": int((deltas < -1e-12).sum()),
            "unchanged": int((np.abs(deltas) <= 1e-12).sum()),
            "worst_query_ids": order[:5],
            "best_query_ids": list(reversed(order[-5:])),
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/m2_company_policies.json")
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/embeddings/e5_small"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = ComparisonConfig.model_validate_json(args.config.read_text())
        if args.output.exists():
            raise DatasetError("comparison output exists; choose a new directory")
        records = load_archive_records(args.archive)
        split = SplitManifest.model_validate_json(args.split.read_text())
        verify_split(split, records, ARCHIVE_SHA256)
        documents = [doc for record in records for doc in record.docs_ko]
        encoder = E5Encoder(args.model, batch_size=config.batch_size, max_length=config.max_length)
        index, cache = corpus_cache(documents, encoder, args.cache)
        started = perf_counter()
        sparse = BM25Retriever(
            documents, k1=config.baseline.k1, b=config.baseline.b, epsilon=config.baseline.epsilon
        )
        sparse_build_ms = (perf_counter() - started) * 1000
        doc_times = {doc.global_doc_id: doc.doc_insert_time for doc in documents}

        def search(method, text, cutoff, inserted):
            if method == "bm25":
                return sparse.search(text, cutoff)
            dense = index.search(encoder.encode([text], kind="query")[0], cutoff)
            if method == "dense":
                return dense
            hybrid = reciprocal_rank_fusion(
                [sparse.search(text, cutoff), dense], cutoff, config.rrf_offset
            )
            return (
                insertion_time_rerank(hybrid, inserted, doc_times, weight=config.insertion_weight)
                if method == "hybrid_insertion_time"
                else hybrid
            )

        args.output.mkdir(parents=True, exist_ok=False)
        provenance = {
            "dataset_revision": DATA_REVISION,
            "archive_sha256": ARCHIVE_SHA256,
            "member": MEMBER,
            "split_sha256": sha256_file(args.split),
            "comparison_config_sha256": sha256_file(args.config),
            "encoder_cache": cache,
            "encoder": encoder.profile(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["torch", "transformers", "faiss-cpu", "tokenizers"]
            },
            "fusion": {
                "method": "equal_weight_rrf",
                "offset": config.rrf_offset,
                "input_top_k": config.baseline.top_k,
            },
            "temporal_policy": {
                "name": "soft_insertion_gap_v1",
                "weight": config.insertion_weight,
                "hard_filter": False,
                "event_time_inferred": False,
            },
            **source_snapshot(Path.cwd()),
        }
        summaries, rows, mlflow_ids = {}, {}, {}
        for method in METHODS:
            print(f"Evaluating {method}: fixed {config.baseline.partition} selection", flush=True)
            method_config = config.baseline.model_copy(
                update={
                    "method": method,
                    "name": (
                        f"{method}_m2_{config.baseline.partition}{config.baseline.query_limit}_v1"
                    ),
                }
            )
            build_ms = (sparse_build_ms if method != "dense" else 0) + (
                cache["faiss_build_ms"] if method != "bm25" else 0
            )
            summaries[method] = run_baseline(
                records,
                split,
                method_config,
                args.output / method,
                provenance,
                search=partial(search, method),
                index_build_ms=build_ms,
                timing_scope=(
                    f"sequential {method}; GPU query encoding if used, CPU scoring/search/fusion; "
                    "excludes offline embeddings and metrics"
                ),
            )
            rows[method] = [
                json.loads(line)
                for line in (args.output / method / "per_query.jsonl").read_text().splitlines()
            ]
            if args.mlflow:
                from temporalproof.tracking.local import log_run_directory

                mlflow_ids[method] = log_run_directory(args.output / method)
        comparison = {
            "config": config.model_dump(),
            "summaries": summaries,
            "paired_deltas_vs_bm25": {
                method: paired_comparisons(
                    rows["bm25"],
                    rows[method],
                    config.baseline.seed,
                    config.baseline.bootstrap_samples,
                )
                for method in METHODS[1:]
            },
            "encoder_profile": encoder.profile(),
            "encoder_cache": cache,
            "mlflow_run_ids": mlflow_ids,
            "test_partition_evaluated": False,
        }
        (args.output / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n"
        )
    except (OSError, ValueError, KeyError) as exc:
        message = (
            str(exc) if isinstance(exc, DatasetError) else "input, model or cache validation failed"
        )
        print(f"comparison failed: {message}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                method: result["metrics"]["ndcg_at_10"]["mean"]
                for method, result in summaries.items()
            },
            sort_keys=True,
        )
    )
    print(f"Comparison artifacts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
