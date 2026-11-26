"""M3: fixed cross-encoder on verified frozen train/validation candidates."""

import argparse
import importlib.metadata
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

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
from temporalproof.evaluation.comparison import paired_comparisons
from temporalproof.retrieval.bm25 import Hit
from temporalproof.retrieval.reranker import (
    KOREAN_MODEL_ID,
    MODEL_ID,
    CrossEncoder,
    model_spec,
    rank_scores,
)

SOURCE_FILES = {
    "config.json",
    "summary.json",
    "per_query.jsonl",
    "run.trec",
    "qrels.tsv",
    "failure_cases.json",
}
TIMING_SCOPE = (
    "incremental rerank only: frozen candidate lookup, text lookup, pair tokenization, "
    "CPU/GPU transfers, synchronized CUDA forward, scalar scores and sorting; concurrency 1; "
    "excludes fast retrieval, model load, warmup, metrics, file writes and tracking"
)


class RerankingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    baseline: BaselineConfig
    model_id: Literal[MODEL_ID, KOREAN_MODEL_ID]
    revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    batch_size: int = Field(strict=True, ge=1)
    max_length: int = Field(strict=True, ge=1, le=512)
    candidate_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_model_revision(self) -> "RerankingConfig":
        if self.revision != model_spec(self.model_id).revision:
            raise ValueError("model revision differs from pinned specification")
        return self


@dataclass(frozen=True)
class CandidatePool:
    hits: dict[int, list[Hit]]
    manifest: dict
    manifest_sha256: str
    run_sha256: str
    rows: list[dict]
    summary: dict


class Scorer(Protocol):
    def score(self, query: str, passages: Sequence[str]) -> Sequence[float]: ...
    def profile(self) -> dict: ...


def load_candidate_pool(
    directory: Path,
    records: Sequence[QueryRecord],
    split: SplitManifest,
    config: RerankingConfig,
    split_sha256: str,
) -> CandidatePool:
    """Verify source evidence, IDs, exact selection, data/split and qrels before GPU work."""
    digest = sha256_file(directory / "manifest.json")
    if digest != config.candidate_manifest_sha256:
        raise DatasetError("candidate manifest checksum differs from frozen experiment config")
    manifest = json.loads((directory / "manifest.json").read_text())
    if set(manifest["output_sha256"]) != SOURCE_FILES:
        raise DatasetError("unexpected source artifact list")
    for name in SOURCE_FILES:
        if sha256_file(directory / name) != manifest["output_sha256"][name]:
            raise DatasetError("candidate source artifact checksum mismatch")
    original = BaselineConfig.model_validate(manifest["config"])
    if original.method != "hybrid_rrf" or config.baseline.method != "fixed_cross_encoder":
        raise DatasetError("M3 requires hybrid candidates and a fixed cross-encoder run")
    fields = (
        "partition",
        "seed",
        "query_limit",
        "top_k",
        "k1",
        "b",
        "epsilon",
        "tokenizer",
        "tie_break",
    )
    if any(getattr(original, name) != getattr(config.baseline, name) for name in fields):
        raise DatasetError(
            "reranker selection/cutoff/retrieval config differs from candidate source"
        )
    selected = select_queries(split, original.partition, original.query_limit, original.seed)
    documents = [doc for record in records for doc in record.docs_ko]
    doc_ids = {doc.global_doc_id for doc in documents}
    if len(doc_ids) != len(documents):
        raise DatasetError("corpus document IDs are not unique")
    if (
        manifest["archive_sha256"] != ARCHIVE_SHA256
        or manifest["split_sha256"] != split_sha256
        or manifest["selected_query_ids"] != selected
        or manifest["corpus_document_count"] != len(documents)
        or manifest["test_partition_evaluated"] is not False
    ):
        raise DatasetError("candidate data/split/selection provenance mismatch")
    pools: dict[int, list[Hit]] = {query_id: [] for query_id in selected}
    seen: set[tuple[int, int]] = set()
    for line in (directory / "run.trec").read_text().splitlines():
        query, iteration, doc, rank, score, method = line.split()
        query_id, doc_id, position, value = int(query), int(doc), int(rank), float(score)
        if query_id not in pools or doc_id not in doc_ids or (query_id, doc_id) in seen:
            raise DatasetError("candidate unknown/duplicate query or document ID")
        hits = pools[query_id]
        if (
            position != len(hits) + 1
            or not math.isfinite(value)
            or iteration != "Q0"
            or method != "hybrid_rrf"
        ):
            raise DatasetError("invalid candidate rank, score or method")
        if hits and (-value, doc_id) < (-hits[-1].score, hits[-1].doc_id):
            raise DatasetError("candidate ranking violates declared tie break")
        hits.append(Hit(doc_id, value))
        seen.add((query_id, doc_id))
    if any(len(hits) != original.top_k for hits in pools.values()):
        raise DatasetError("each query requires exactly the frozen top-k candidates")
    by_query = {record.index: record for record in records}
    expected_qrels = "".join(
        f"{q}\t0\t{doc}\t1\n"
        for q in selected
        for doc in sorted(by_query[q].gold_docs_final_global)
    )
    if (directory / "qrels.tsv").read_text() != expected_qrels:
        raise DatasetError("candidate judgments differ from unchanged source data")
    rows = [json.loads(line) for line in (directory / "per_query.jsonl").read_text().splitlines()]
    if [row["query_id"] for row in rows] != selected:
        raise DatasetError("candidate metric rows do not match selected queries")
    return CandidatePool(
        pools,
        manifest,
        digest,
        manifest["output_sha256"]["run.trec"],
        rows,
        json.loads((directory / "summary.json").read_text()),
    )


def run_reranking(
    records: Sequence[QueryRecord],
    split: SplitManifest,
    config: RerankingConfig,
    pool: CandidatePool,
    scorer: Scorer,
    output: Path,
    provenance: dict,
) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    by_doc = {doc.global_doc_id: doc.text for record in records for doc in record.docs_ko}
    identity = {
        "source_manifest_sha256": pool.manifest_sha256,
        "source_run_sha256": pool.run_sha256,
        "source_snapshot_sha256": pool.manifest["source_snapshot_sha256"],
        "query_count": len(pool.hits),
        "candidates_per_query": config.baseline.top_k,
        "score_cache": "run/run.trec; one raw cross-encoder logit per frozen query/document pair",
        "no_test_access": True,
    }
    frozen = output / "candidates"
    frozen.mkdir()
    _write_json(frozen / "source_manifest.json", pool.manifest)
    _write_json(
        frozen / "pool.json",
        {
            **identity,
            "ranked_candidates": {
                str(q): [{"doc_id": hit.doc_id, "score": hit.score} for hit in hits]
                for q, hits in pool.hits.items()
            },
        },
    )
    identity["frozen_pool_sha256"] = sha256_file(frozen / "pool.json")

    def search(query_id: int, text: str, cutoff: int, inserted: str | None) -> list[Hit]:
        candidates = pool.hits[query_id]
        if cutoff != len(candidates):
            raise DatasetError("cutoff differs from frozen pool")
        ids = [hit.doc_id for hit in candidates]
        # The scorer sees only query/passage text, never IDs, timestamps or qrels.
        return rank_scores(ids, scorer.score(text, [by_doc[doc_id] for doc_id in ids]))

    summary = run_baseline(
        records,
        split,
        config.baseline,
        output / "run",
        {**provenance, "candidate_pool": identity, "reranking_config": config.model_dump()},
        query_search=search,
        warmup_query_id=next(iter(pool.hits)),
        timing_scope=TIMING_SCOPE,
    )
    run = output / "run"
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["reranker"] = scorer.profile()
    _write_json(run / "manifest.json", manifest)
    rows = [json.loads(line) for line in (run / "per_query.jsonl").read_text().splitlines()]
    if (run / "qrels.tsv").read_text() != "".join(
        f"{q}\t0\t{doc}\t1\n"
        for q in pool.hits
        for doc in sorted(next(r for r in records if r.index == q).gold_docs_final_global)
    ):
        raise DatasetError("reranking changed evaluation judgments")
    if any(a["recall_at_100"] != b["recall_at_100"] for a, b in zip(pool.rows, rows, strict=True)):
        raise DatasetError("reranking changed candidate coverage")
    timing = summary["retrieval_timing_ms"]
    comparison = {
        "config": config.model_dump(),
        "candidate_pool": identity,
        "fast_summary": pool.summary,
        "reranker_summary": summary,
        "paired_deltas_vs_hybrid": paired_comparisons(
            pool.rows, rows, config.baseline.seed, config.baseline.bootstrap_samples
        ),
        "reranker_profile": scorer.profile(),
        "incremental_queries_per_second": 1000 / timing["mean"],
        "incremental_pairs_per_second": config.baseline.top_k * 1000 / timing["mean"],
        "timing_comparison_scope": (
            "fast timings retained from source M2 run; new timings are rerank-only on frozen "
            "candidates. No live end-to-end, concurrent encoder/reranker, service throughput "
            "or sum-of-p95 claim. Models loaded separately."
        ),
        "test_partition_evaluated": False,
        "router_training_targets_generated": False,
    }
    _write_json(output / "comparison.json", comparison)
    return comparison


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/m3_company_policies.json")
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("artifacts/m2/company_policies_validation100_verified_v1/hybrid_rrf"),
    )
    parser.add_argument(
        "--model", type=Path, help="local assets; defaults to configured model's path"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise DatasetError("output exists; choose a new run directory")
        config = RerankingConfig.model_validate_json(args.config.read_text())
        records = load_archive_records(args.archive)
        split = SplitManifest.model_validate_json(args.split.read_text())
        verify_split(split, records, ARCHIVE_SHA256)
        pool = load_candidate_pool(args.candidates, records, split, config, sha256_file(args.split))
        spec = model_spec(config.model_id)
        scorer = CrossEncoder(
            args.model or spec.directory,
            batch_size=config.batch_size,
            max_length=config.max_length,
            spec=spec,
        )
        provenance = {
            "dataset_revision": DATA_REVISION,
            "archive_sha256": ARCHIVE_SHA256,
            "member": MEMBER,
            "split_sha256": sha256_file(args.split),
            "config_sha256": sha256_file(args.config),
            "candidate_source_path": str(args.candidates),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ["torch", "transformers", "tokenizers"]
            },
            **source_snapshot(Path.cwd()),
        }
        comparison = run_reranking(records, split, config, pool, scorer, args.output, provenance)
        if args.mlflow:
            from temporalproof.tracking.local import log_run_directory

            comparison["mlflow_run_id"] = log_run_directory(args.output / "run")
            _write_json(args.output / "comparison.json", comparison)
    except (OSError, ValueError, KeyError) as exc:
        message = (
            str(exc) if isinstance(exc, DatasetError) else "input/model/evidence validation failed"
        )
        print(
            f"reranking failed: {message}; completed evidence, if any, remains intact",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(comparison["reranker_summary"]["metrics"], sort_keys=True))
    print(f"M3 artifacts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
