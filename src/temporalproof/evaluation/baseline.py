"""Reproducible CPU BM25 evaluation on train/validation only (M1)."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from temporalproof.data.source import (
    ARCHIVE_SHA256,
    DEFAULT_ARCHIVE,
    MEMBER,
    REVISION,
    load_archive_records,
    sha256_file,
)
from temporalproof.data.splits import DEFAULT_SPLIT, SplitManifest, select_queries, verify_split
from temporalproof.data.timelyqa import DatasetError, QueryRecord, parse_timestamp
from temporalproof.evaluation.metrics import (
    ndcg_at_k,
    recall_at_k,
    reference_prefix_ndcg_at_k,
)
from temporalproof.retrieval.bm25 import BM25Retriever, Hit

DEFAULT_TIMING_SCOPE = (
    "sequential warm BM25 tokenization, scoring and top-k sorting; not API latency"
)


class BaselineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    method: Literal[
        "bm25", "dense", "hybrid_rrf", "hybrid_insertion_time", "fixed_cross_encoder"
    ] = "bm25"
    partition: Literal["train", "validation"]
    seed: int = Field(strict=True, ge=0)
    query_limit: int = Field(strict=True, ge=1)
    top_k: int = Field(strict=True, ge=100)
    k1: float = Field(gt=0)
    b: float = Field(ge=0, le=1)
    epsilon: float = Field(ge=0)
    tokenizer: Literal["python_str_split_v1"]
    tie_break: Literal["score_desc_doc_id_asc"]
    warmup_queries: int = Field(strict=True, ge=0)
    bootstrap_samples: int = Field(strict=True, ge=100)
    failure_examples: int = Field(strict=True, ge=1)


def evaluate_query(record: QueryRecord, hits: Sequence[Hit]) -> dict[str, object]:
    ranking = [hit.doc_id for hit in hits]
    positives = set(record.gold_docs_final_global)
    query_time = parse_timestamp(record.query_insert_time)
    future_gold = sum(
        doc.global_doc_id in positives
        and query_time is not None
        and (inserted := parse_timestamp(doc.doc_insert_time)) is not None
        and inserted > query_time
        for doc in record.docs_ko
    )
    return {
        "query_id": record.index,
        "ndcg_at_10": ndcg_at_k(ranking, positives, 10),
        "recall_at_10": recall_at_k(ranking, positives, 10),
        "recall_at_20": recall_at_k(ranking, positives, 20),
        "recall_at_100": recall_at_k(ranking, positives, 100),
        "reference_prefix_ndcg_at_10": reference_prefix_ndcg_at_k(ranking, positives, 10),
        "positive_count": len(positives),
        "future_inserted_positive_count": future_gold,
        "missing_positive_ids_at_100": sorted(positives - set(ranking[:100])),
    }


def summarize(
    per_query: Sequence[dict[str, object]],
    seed: int,
    bootstrap_samples: int,
    timing_scope: str = DEFAULT_TIMING_SCOPE,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    metrics = {}
    for key in (
        "ndcg_at_10",
        "recall_at_10",
        "recall_at_20",
        "recall_at_100",
        "reference_prefix_ndcg_at_10",
    ):
        values = np.array([row[key] for row in per_query], dtype=float)
        means = rng.choice(values, size=(bootstrap_samples, len(values))).mean(axis=1)
        metrics[key] = {
            "mean": float(values.mean()),
            "query_bootstrap_95_percent_interval": np.quantile(means, [0.025, 0.975]).tolist(),
        }
    latency = np.array([row["retrieval_ms"] for row in per_query], dtype=float)
    return {
        "query_count": len(per_query),
        "metrics": metrics,
        "queries_with_incomplete_top10": sum(row["recall_at_10"] < 1 for row in per_query),
        "queries_with_no_gold_in_top100": sum(row["recall_at_100"] == 0 for row in per_query),
        "retrieval_timing_ms": {
            "mean": float(latency.mean()),
            "p50": float(np.quantile(latency, 0.5)),
            "p95": float(np.quantile(latency, 0.95)),
            "concurrency": 1,
            "scope": timing_scope,
        },
        "interval_scope": (
            "query bootstrap conditional on selected development queries; "
            "no population or paired significance claim"
        ),
    }


def source_snapshot(repository: Path) -> dict[str, object]:
    paths = sorted((repository / "src" / "temporalproof").rglob("*.py"))
    paths += [repository / "pyproject.toml", repository / "uv.lock"]
    files = {str(path.relative_to(repository)): sha256_file(path) for path in paths}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()

    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=repository, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "source_files_sha256": files,
        "source_snapshot_sha256": digest,
    }


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_baseline(
    records: Sequence[QueryRecord],
    split: SplitManifest,
    config: BaselineConfig,
    output: Path,
    provenance: dict[str, object],
    *,
    search: Callable[[str, int, str | None], list[Hit]] | None = None,
    query_search: Callable[[int, str, int, str | None], list[Hit]] | None = None,
    warmup_query_id: int | None = None,
    index_build_ms: float | None = None,
    timing_scope: str = DEFAULT_TIMING_SCOPE,
) -> dict[str, object]:
    """No tuning, learned preprocessing or test partition access occurs here."""
    selected = select_queries(split, config.partition, config.query_limit, config.seed)
    by_query = {record.index: record for record in records}
    documents = [doc for record in records for doc in record.docs_ko]
    by_doc = {doc.global_doc_id: doc for doc in documents}
    if search is not None and query_search is not None:
        raise ValueError("provide only one retrieval callback")
    train_ids = split.query_ids("train")
    warmup_id = warmup_query_id
    if warmup_id is None:
        warmup_id = train_ids[0] if train_ids else selected[0]
    if warmup_id not in train_ids and warmup_id not in selected:
        raise ValueError("warmup query must be in train or selected development queries")
    output.mkdir(parents=True, exist_ok=False)
    if search is None and query_search is None:
        if config.method != "bm25":
            raise ValueError("non-BM25 runs require an explicit retrieval function")
        started = perf_counter()
        retriever = BM25Retriever(documents, k1=config.k1, b=config.b, epsilon=config.epsilon)
        build_ms = (perf_counter() - started) * 1000

        def search(text: str, cutoff: int, inserted: str | None) -> list[Hit]:
            return retriever.search(text, cutoff)
    else:
        build_ms = index_build_ms or 0.0
    if query_search is None:

        def query_search(query_id: int, text: str, cutoff: int, inserted: str | None) -> list[Hit]:
            return search(text, cutoff, inserted)

    # Default warmup uses train; a frozen-pool run can explicitly use a selected query.
    for _ in range(config.warmup_queries):
        record = by_query[warmup_id]
        query_search(warmup_id, record.query_ko, config.top_k, record.query_insert_time)
    per_query = []
    rankings: dict[int, list[Hit]] = {}
    with (
        (output / "run.trec").open("w", encoding="utf-8") as run_file,
        (output / "qrels.tsv").open("w", encoding="utf-8") as qrel_file,
    ):
        for query_id in selected:
            record = by_query[query_id]
            started = perf_counter()
            hits = query_search(query_id, record.query_ko, config.top_k, record.query_insert_time)
            elapsed_ms = (perf_counter() - started) * 1000
            row = evaluate_query(record, hits)
            row["retrieval_ms"] = elapsed_ms
            per_query.append(row)
            rankings[query_id] = hits
            for rank, hit in enumerate(hits, start=1):
                run_file.write(
                    f"{query_id} Q0 {hit.doc_id} {rank} {hit.score:.17g} {config.method}\n"
                )
            for doc_id in sorted(record.gold_docs_final_global):
                qrel_file.write(f"{query_id}\t0\t{doc_id}\t1\n")
    with (output / "per_query.jsonl").open("w", encoding="utf-8") as stream:
        for row in per_query:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = summarize(per_query, config.seed, config.bootstrap_samples, timing_scope)
    summary["index_build_ms"] = build_ms
    _write_json(output / "summary.json", summary)
    # Fixed selection rule; snippets stay in ignored local artifacts because rights are unresolved.
    failures = []
    for row in sorted(per_query, key=lambda row: (row["ndcg_at_10"], row["query_id"]))[
        : config.failure_examples
    ]:
        record = by_query[row["query_id"]]
        failures.append(
            {
                "metrics": row,
                "query": record.query_ko,
                "query_insert_time": record.query_insert_time,
                "gold": [by_doc[doc_id].model_dump() for doc_id in record.gold_docs_final_global],
                "top_three": [
                    by_doc[hit.doc_id].model_dump() for hit in rankings[record.index][:3]
                ],
            }
        )
    _write_json(output / "failure_cases.json", failures)
    _write_json(output / "config.json", config.model_dump())
    manifest = {
        **provenance,
        "config": config.model_dump(),
        "selected_query_ids": selected,
        "corpus_document_count": len(documents),
        "corpus_scope": "full_domain_union; no text deduplication or timestamp filtering",
        "qrel_scale": "binary; listed positives=1; unjudged treated as 0 for scoring only",
        "cutoffs": {"ndcg": [10], "recall": [10, 20, 100]},
        "warmup_query_id": warmup_id,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpu_count": os.cpu_count(),
        "packages": {
            **provenance.get("packages", {}),
            **{
                name: importlib.metadata.version(name)
                for name in ["numpy", "rank-bm25", "pydantic"]
            },
        },
        "thread_environment": {
            key: os.environ.get(key)
            for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
        },
        "output_sha256": {path.name: sha256_file(path) for path in sorted(output.iterdir())},
        "test_partition_evaluated": False,
    }
    _write_json(output / "manifest.json", manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--mlflow", action="store_true", help="also log completed results to local MLflow"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/bm25_company_policies.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new directory; existing runs are never overwritten",
    )
    args = parser.parse_args(argv)
    try:
        config = BaselineConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        if config.method != "bm25":
            raise DatasetError("use temporalproof-compare for non-BM25 methods")
        split = SplitManifest.model_validate_json(args.split.read_text(encoding="utf-8"))
        records = load_archive_records(args.archive)
        verify_split(split, records, ARCHIVE_SHA256)
        if args.output.exists():
            raise DatasetError("output exists; choose a new run directory")
        provenance = {
            "dataset_revision": REVISION,
            "archive_sha256": ARCHIVE_SHA256,
            "member": MEMBER,
            "split_sha256": sha256_file(args.split),
            "config_sha256": sha256_file(args.config),
            **source_snapshot(Path.cwd()),
        }
        print(
            f"BM25: {config.query_limit} {config.partition} queries, full domain corpus.",
            flush=True,
        )
        summary = run_baseline(records, split, config, args.output, provenance)
    except (OSError, ValueError, KeyError) as exc:
        message = (
            str(exc) if isinstance(exc, DatasetError) else "input/configuration validation failed"
        )
        print(f"baseline failed: {message}", file=sys.stderr)
        return 1
    print(json.dumps(summary["metrics"], sort_keys=True))
    print(f"Artifacts: {args.output}")
    if args.mlflow:
        from temporalproof.tracking.local import TrackingError, log_run_directory

        try:
            run_id = log_run_directory(args.output)
        except (TrackingError, OSError):
            print(
                "BM25 completed; local MLflow export failed. Saved file results are intact.",
                file=sys.stderr,
            )
            return 1
        print(f"Local MLflow run: {run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
