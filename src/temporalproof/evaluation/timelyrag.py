"""Pinned TimelyRAG auto-alpha on matched local validation; E5 runs on CUDA."""

import argparse
import importlib.metadata
import json
import platform
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from temporalproof.data.source import (
    ARCHIVE_SHA256,
    DEFAULT_ARCHIVE,
    REVISION,
    load_archive_records,
    sha256_file,
)
from temporalproof.data.splits import DEFAULT_SPLIT, SplitManifest, select_queries, verify_split
from temporalproof.data.timelyqa import DatasetError
from temporalproof.evaluation.baseline import (
    BaselineConfig,
    _write_json,
    run_baseline,
    source_snapshot,
)
from temporalproof.evaluation.pilot import read_rows, write_rows
from temporalproof.retrieval.bm25 import BM25Retriever, Hit
from temporalproof.retrieval.dense import corpus_cache, normalized_matrix
from temporalproof.retrieval.encoder import DEFAULT_MODEL, MODEL_ID, E5Encoder
from temporalproof.retrieval.encoder import REVISION as ENCODER_REVISION
from temporalproof.retrieval.fusion import reciprocal_rank_fusion
from temporalproof.retrieval.timelyrag import automatic_rerank, load_reference, native_top_indices
from temporalproof.routing.cross_validation import grouped_interval

REFERENCE_MANIFEST = Path("configs/datasets/timelyrag_reference.json")


class ReferenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    baseline: BaselineConfig
    reference_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reference_revision: Literal[REVISION]
    encoder_model_id: Literal[MODEL_ID]
    encoder_revision: Literal[ENCODER_REVISION]
    encoder_batch_size: int = Field(strict=True, ge=1)
    encoder_max_length: int = Field(strict=True, ge=1, le=512)
    rrf_offset: int = Field(strict=True, ge=1)
    methods: tuple[Literal["bm25", "dense", "hybrid_rrf"], ...]
    primary: Literal["bm25_timelyrag_auto"]
    oracle_evaluation: Literal[False]
    scope: str

    @model_validator(mode="after")
    def validation_only(self) -> Self:
        if (
            self.baseline.partition != "validation"
            or self.baseline.top_k != 100
            or self.baseline.tie_break != "numpy_argsort_reverse_reference_v1"
            or self.methods != ("bm25", "dense", "hybrid_rrf")
        ):
            raise ValueError("requires matched validation/native NumPy ties and all three controls")
        return self


def hits_from_scores(scores, ids, cutoff):
    return [Hit(ids[i], float(scores[i])) for i in native_top_indices(scores, cutoff)]


def collect_pair(
    records,
    split,
    config,
    method,
    root,
    provenance,
    score_fn,
    reference,
    ids,
    dits,
    dets,
    build_ms=0,
):
    """Measure the fast path once; temporal callback gets cached scores/text/time, no qrels."""
    selected = select_queries(
        split, "validation", config.baseline.query_limit, config.baseline.seed
    )
    cached, diagnostic = {}, {}

    def fast_search(q, text, cutoff, inserted):
        scores = np.asarray(score_fn(text))
        if len(scores) != len(ids):
            raise DatasetError("base score vector is not corpus-aligned")
        cached[q] = scores.copy()
        return hits_from_scores(scores, ids, cutoff)

    base_config = config.baseline.model_copy(
        update={"method": method, "name": f"{method}_reference_order_validation{len(selected)}_v1"}
    )
    run_baseline(
        records,
        split,
        base_config,
        root / method,
        provenance,
        query_search=fast_search,
        index_build_ms=build_ms,
        timing_scope="warm actual fast retrieval + native NumPy selection; concurrency 1; "
        "GPU E5 query encoding when used; excludes build/files/metrics",
    )
    cache_path = root / f"{method}_base_scores.npy"
    np.save(cache_path, np.stack([cached[q] for q in selected]), allow_pickle=False)
    cache_hash = sha256_file(cache_path)

    def auto_search(q, text, cutoff, inserted):
        hits, info = automatic_rerank(reference, cached[q], ids, dits, dets, text, inserted)
        if q in selected:
            diagnostic[q] = info
        if len(hits) != cutoff:
            raise DatasetError("reference pool cutoff differs")
        if set(h.doc_id for h in hits) != {
            h.doc_id for h in hits_from_scores(cached[q], ids, cutoff)
        }:
            raise DatasetError("temporal module changed candidate pool")
        return hits

    name = f"{method}_timelyrag_auto"
    auto_config = config.baseline.model_copy(
        update={
            "method": "timelyrag_reference_auto",
            "name": f"{name}_validation{len(selected)}_v1",
        }
    )
    run_baseline(
        records,
        split,
        auto_config,
        root / name,
        {
            **provenance,
            "base_method": method,
            "cached_base_scores_sha256": cache_hash,
            "base_manifest_sha256": sha256_file(root / method / "manifest.json"),
        },
        query_search=auto_search,
        timing_scope="incremental temporal auto-alpha only: cached score lookup, query date "
        "parsing, candidate selection/time helpers/sorting; concurrency 1; "
        "excludes source fast retrieval/build/metrics/files; not pipeline p95",
    )
    write_rows(
        root / f"{method}_temporal_diagnostics.jsonl",
        [{"query_id": q, **diagnostic[q]} for q in selected],
    )
    return {
        "base_score_file": cache_path.name,
        "base_score_dtype": str(cached[selected[0]].dtype),
        "base_scores_sha256": cache_hash,
        "alpha_histogram": dict(
            sorted(Counter(str(diagnostic[q]["alpha"]) for q in selected).items())
        ),
        "query_event_time_extracted_count": sum(
            diagnostic[q]["query_event_time_extracted"] for q in selected
        ),
        "unknown_delta_candidates": sum(
            diagnostic[q]["unknown_temporal_distance_count"] for q in selected
        ),
        "later_or_equal_insert_candidates": sum(
            diagnostic[q]["later_or_equal_insert_candidate_count"] for q in selected
        ),
    }


def comparison(root, method, groups, config):
    fast = read_rows(root / method / "per_query.jsonl")
    auto = read_rows(root / f"{method}_timelyrag_auto" / "per_query.jsonl")
    if [r["query_id"] for r in fast] != [r["query_id"] for r in auto]:
        raise DatasetError("paired query IDs differ")
    if (root / method / "qrels.tsv").read_bytes() != (
        root / f"{method}_timelyrag_auto" / "qrels.tsv"
    ).read_bytes():
        raise DatasetError("paired official qrels differ")
    if any(a["recall_at_100"] != b["recall_at_100"] for a, b in zip(fast, auto, strict=True)):
        raise DatasetError("candidate coverage changed")
    metric_names = [
        "ndcg_at_10",
        "recall_at_10",
        "recall_at_20",
        "recall_at_100",
        "reference_prefix_ndcg_at_10",
    ]
    delta = [b["ndcg_at_10"] - a["ndcg_at_10"] for a, b in zip(fast, auto, strict=True)]
    return {
        "base_method": method,
        "query_count": len(fast),
        "base_metrics": {m: float(np.mean([r[m] for r in fast])) for m in metric_names},
        "auto_metrics": {m: float(np.mean([r[m] for r in auto])) for m in metric_names},
        "mean_ndcg_delta": float(np.mean(delta)),
        "paired_group_bootstrap_95_interval": grouped_interval(
            delta,
            [groups[r["query_id"]] for r in fast],
            config.baseline.seed,
            config.baseline.bootstrap_samples,
        ),
        "gains_regressions_unchanged": [
            sum(v > 1e-12 for v in delta),
            sum(v < -1e-12 for v in delta),
            sum(abs(v) <= 1e-12 for v in delta),
        ],
        "source_fast_mean_ms": float(np.mean([r["retrieval_ms"] for r in fast])),
        "incremental_temporal_mean_ms": float(np.mean([r["retrieval_ms"] for r in auto])),
        "modeled_additive_mean_ms": float(
            np.mean(
                [a["retrieval_ms"] + b["retrieval_ms"] for a, b in zip(fast, auto, strict=True)]
            )
        ),
        "cost_scope": (
            "measured stages collected separately; sum is modeled, not live pipeline latency"
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/m4_timelyrag_reference_company_policies.json"),
    )
    parser.add_argument("--reference-source", type=Path, default=DEFAULT_ARCHIVE.parent)
    parser.add_argument("--reference-manifest", type=Path, default=REFERENCE_MANIFEST)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--encoder", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache", type=Path, default=Path("artifacts/embeddings/e5_small"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = ReferenceConfig.model_validate_json(args.config.read_text())
        if args.output.exists():
            raise DatasetError("output exists; choose a new reference run")
        reference = load_reference(
            args.reference_source, args.reference_manifest, config.reference_manifest_sha256
        )
        if reference.config.K != 10 or reference.config.STAGE1_TOPN != config.baseline.top_k:
            raise DatasetError("pinned upstream cutoffs differ")
        records = load_archive_records(args.archive)
        split = SplitManifest.model_validate_json(args.split.read_text())
        verify_split(split, records, ARCHIVE_SHA256)
        if config.baseline.query_limit != len(split.query_ids("validation")):
            raise DatasetError("reference evaluation must use complete validation")
        documents = [d for r in records for d in r.docs_ko]
        ids = [d.global_doc_id for d in documents]
        if ids != sorted(ids):
            raise DatasetError("reference corpus order differs from ordered BM25/FAISS corpus")
        groups = {
            q: ",".join(map(str, group)) for group in split.groups["validation"] for q in group
        }
        root = args.output
        root.mkdir(parents=True, exist_ok=False)
        provenance = {
            **source_snapshot(Path.cwd()),
            "archive_sha256": ARCHIVE_SHA256,
            "dataset_revision": REVISION,
            "split_sha256": sha256_file(args.split),
            "config_file_sha256": sha256_file(args.config),
            "reference_manifest_sha256": config.reference_manifest_sha256,
            "packages": {
                n: importlib.metadata.version(n)
                for n in ["numpy", "rank-bm25", "torch", "transformers", "faiss-cpu", "tokenizers"]
            },
            "python": platform.python_version(),
            "reference_execution": (
                "original reviewed helper modules; unchanged original function AST for parity"
            ),
            "oracle_evaluation": False,
            "hard_timestamp_filter": False,
        }
        _write_json(root / "config.json", config.model_dump())
        _write_json(root / "reference_manifest.json", reference.manifest)
        _write_json(
            root / "plan.json",
            {
                "config": config.model_dump(),
                "provenance": provenance,
                "selected_query_ids": select_queries(
                    split, "validation", config.baseline.query_limit, config.baseline.seed
                ),
                "scope": config.scope,
                "test_partition_evaluated": False,
                "primary": config.primary,
                "models_from_paper_reproduced": ["BM25"],
                "dense_backbone_substitution": MODEL_ID,
                "paper_whole_benchmark_reproduced": False,
            },
        )
        encoder = E5Encoder(
            args.encoder, batch_size=config.encoder_batch_size, max_length=config.encoder_max_length
        )
        index, cache = corpus_cache(documents, encoder, args.cache)
        build_started = perf_counter()
        sparse = BM25Retriever(
            documents, k1=config.baseline.k1, b=config.baseline.b, epsilon=config.baseline.epsilon
        )
        sparse_build_ms = (perf_counter() - build_started) * 1000
        dits = np.array([reference.utils.to_dt(d.doc_insert_time) for d in documents], dtype=object)
        dets = np.array([None for _ in documents], dtype=object)

        def dense_scores(text):
            query = normalized_matrix(encoder.encode([text], kind="query"))
            scores, positions = index.index.search(query, len(ids))
            vector = np.empty(len(ids), dtype=np.float32)
            vector[positions[0]] = scores[0]
            return vector

        positions_by_id = {d: i for i, d in enumerate(ids)}

        def score_vector(method, text):
            if method == "bm25":
                return sparse.index.get_scores(text.split())
            dense = dense_scores(text)
            if method == "dense":
                return dense
            sparse_hits = sparse.search(text, config.baseline.top_k)
            order = np.lexsort((np.array(ids), -dense))[: config.baseline.top_k]
            dense_hits = [Hit(ids[i], float(dense[i])) for i in order]
            hybrid = reciprocal_rank_fusion(
                [sparse_hits, dense_hits], config.baseline.top_k, config.rrf_offset
            )
            scores = np.zeros(len(ids), dtype=np.float64)
            for h in hybrid:
                scores[positions_by_id[h.doc_id]] = h.score
            return scores

        metadata, results = {}, {}
        for method in config.methods:
            print(
                f"TimelyRAG local reference: {method}, CUDA E5 where used; fixed validation495",
                flush=True,
            )
            metadata[method] = collect_pair(
                records,
                split,
                config,
                method,
                root,
                {**provenance, "encoder_cache": cache, "encoder": encoder.profile()},
                lambda text, m=method: score_vector(m, text),
                reference,
                ids,
                dits,
                dets,
                build_ms=(sparse_build_ms if method != "dense" else 0)
                + (cache["faiss_build_ms"] if method != "bm25" else 0),
            )
            results[method] = comparison(root, method, groups, config)
        np.save(root / "corpus_ids.npy", np.array(ids, dtype=np.int64), allow_pickle=False)
        summary = {
            "results": results,
            "diagnostics": metadata,
            "encoder_profile": encoder.profile(),
            "encoder_cache": cache,
            "index_build_ms": {
                "bm25": sparse_build_ms,
                "faiss": cache["faiss_build_ms"],
                "scope": "measured within collection; excludes cached corpus encoding",
            },
            "document_event_time_explicit_count": 0,
            "document_insert_time_unknown_count": int(sum(d is None for d in dits)),
            "test_partition_evaluated": False,
            "oracle_evaluation": False,
            "router_refitted": False,
            "paper_whole_benchmark_reproduced": False,
            "metric_policy": "full-qrel nDCG primary; reference-prefix nDCG separate column",
            "unknown_policy": "upstream NaN distance => neutral affinity .5; no hard exclusion",
            "dense_scope": "E5/hybrid are local extensions; no BGE-M3 weights downloaded",
            "tie_policy": (
                "native reversed numpy.argsort; differs from previous local doc-ID ascending ties"
            ),
        }
        if (
            source_snapshot(Path.cwd())["source_snapshot_sha256"]
            != provenance["source_snapshot_sha256"]
        ):
            raise DatasetError("source changed during GPU collection")
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
        print(f"reference experiment failed: {exc}; saved evidence retained", file=sys.stderr)
        return 1
    print(json.dumps(results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
