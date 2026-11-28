"""Replay all native/auto rankings on CPU; export metadata and both metric definitions."""

import argparse
import json
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np

from temporalproof.data.source import DEFAULT_ARCHIVE, load_archive_records, sha256_file
from temporalproof.data.splits import DEFAULT_SPLIT, SplitManifest, verify_split
from temporalproof.evaluation.baseline import _write_json, evaluate_query, source_snapshot
from temporalproof.evaluation.pilot import read_rows
from temporalproof.evaluation.timelyrag import ReferenceConfig, comparison, hits_from_scores
from temporalproof.retrieval.bm25 import BM25Retriever
from temporalproof.retrieval.timelyrag import automatic_rerank, load_reference


def trec(path):
    rows = {}
    for line in path.read_text().splitlines():
        q, _, doc, rank, score, _ = line.split()
        q, rank = int(q), int(rank)
        if rank != len(rows.get(q, [])) + 1:
            raise ValueError("nonconsecutive source ranks")
        rows.setdefault(q, []).append((int(doc), float(score)))
    return rows


def verify(root, manifest):
    for name, digest in manifest["files_sha256"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise ValueError("reference output checksum/path mismatch")


class AutoRankBoundary(Exception):
    def __init__(self, ranking):
        self.ranking = ranking


def original_auto_prefix(reference, scores, record, ids, dets, dits):
    """Stop original code at its first metric call, before any oracle branch executes.

    Ranking/helper code and original labels are unchanged. The metric boundary is
    instrumented only to observe the completed automatic ranking; no metric is replaced
    in the experiment and no best-alpha computation is reached.
    """
    namespace = reference.original_functions
    original_metric = namespace["evaluate_ranking_ids"]

    def capture(ranking, positives, cutoff):
        raise AutoRankBoundary(list(ranking))

    namespace["evaluate_ranking_ids"] = capture
    qit = reference.utils.to_dt(record.query_insert_time)
    qet, _ = reference.utils.extract_qet_from_query(record.query_ko, qit_hint=qit)
    try:
        namespace["evaluate_2stage"](
            "parity",
            lambda _: scores,
            [
                {
                    "query": record.query_ko,
                    "positive_ids": record.gold_docs_final_global,
                    "qit": qit,
                    "qet": qet,
                }
            ],
            ids,
            dets,
            dits,
        )
    except AutoRankBoundary as observed:
        return observed.ranking
    finally:
        namespace["evaluate_ranking_ids"] = original_metric
    raise ValueError("original automatic metric boundary was not reached")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--reference-source", type=Path, default=DEFAULT_ARCHIVE.parent)
    parser.add_argument(
        "--reference-manifest", type=Path, default=Path("configs/datasets/timelyrag_reference.json")
    )
    parser.add_argument("--earlier-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-on", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("aggregate exists; choose a new path")
    root = args.run
    manifest = json.loads((root / "manifest.json").read_text())
    verify(root, manifest)
    if manifest["source_snapshot_sha256"] != source_snapshot(Path.cwd())["source_snapshot_sha256"]:
        raise ValueError("measurement source differs")
    cfg = ReferenceConfig.model_validate_json((root / "config.json").read_text())
    reference = load_reference(
        args.reference_source, args.reference_manifest, cfg.reference_manifest_sha256
    )
    records = load_archive_records(args.archive)
    split = SplitManifest.model_validate_json(args.split.read_text())
    verify_split(split, records, manifest["archive_sha256"])
    by_query = {r.index: r for r in records}
    documents = [d for r in records for d in r.docs_ko]
    ids = np.load(root / "corpus_ids.npy", allow_pickle=False).tolist()
    if ids != [d.global_doc_id for d in documents]:
        raise ValueError("corpus ordering differs")
    selected = json.loads((root / "plan.json").read_text())["selected_query_ids"]
    groups = {q: ",".join(map(str, g)) for g in split.groups["validation"] for q in g}
    dits = np.array([reference.utils.to_dt(d.doc_insert_time) for d in documents], dtype=object)
    dets = np.array([None for _ in documents], dtype=object)
    saved = json.loads((root / "summary.json").read_text())
    official_columns = {}
    replay_queries = 0
    for method in cfg.methods:
        matrix = np.load(root / f"{method}_base_scores.npy", allow_pickle=False, mmap_mode="r")
        if matrix.shape != (len(selected), len(ids)) or not np.isfinite(matrix).all():
            raise ValueError("cached full score shape/values differ")
        base_run = trec(root / method / "run.trec")
        auto_run = trec(root / f"{method}_timelyrag_auto" / "run.trec")
        base_rows = {r["query_id"]: r for r in read_rows(root / method / "per_query.jsonl")}
        auto_rows = {
            r["query_id"]: r
            for r in read_rows(root / f"{method}_timelyrag_auto" / "per_query.jsonl")
        }
        diagnostic = {
            r["query_id"]: {k: v for k, v in r.items() if k != "query_id"}
            for r in read_rows(root / f"{method}_temporal_diagnostics.jsonl")
        }
        collected = {method: [], f"{method}_timelyrag_auto": []}
        for index, q in enumerate(selected):
            record = by_query[q]
            scores = matrix[index]
            fast = hits_from_scores(scores, ids, cfg.baseline.top_k)
            auto, info = automatic_rerank(
                reference, scores, ids, dits, dets, record.query_ko, record.query_insert_time
            )
            if (
                [(h.doc_id, h.score) for h in fast] != base_run[q]
                or [(h.doc_id, h.score) for h in auto] != auto_run[q]
                or info != diagnostic[q]
            ):
                raise ValueError("full CPU score/rank/alpha replay differs")
            if original_auto_prefix(reference, scores, record, ids, dets, dits) != [
                h.doc_id for h in auto[:10]
            ]:
                raise ValueError("original reference function automatic top10 differs")
            for name, hits, rows in [
                (method, fast, base_rows),
                (f"{method}_timelyrag_auto", auto, auto_rows),
            ]:
                expected = evaluate_query(record, hits)
                if any(rows[q][k] != v for k, v in expected.items()):
                    raise ValueError("per-query full-qrel metric replay differs")
                value = reference.metrics.evaluate_ranking_ids(
                    [h.doc_id for h in hits], record.gold_docs_final_global, 10
                )
                if (
                    abs(value[1] - rows[q]["recall_at_10"]) > 1e-12
                    or abs(value[3] - rows[q]["reference_prefix_ndcg_at_10"]) > 1e-12
                ):
                    raise ValueError("original upstream metric definition differs")
                collected[name].append(value)
            replay_queries += 1
        for name, values in collected.items():
            array = np.asarray(values, dtype=float)
            official_columns[name] = {
                k: float(array[:, i].mean())
                for i, k in enumerate(
                    [
                        "hit_at_10",
                        "recall_at_10",
                        "precision_at_10",
                        "reference_prefix_ndcg_at_10",
                        "reference_prefix_map_at_10",
                    ]
                )
            }
        if comparison(root, method, groups, cfg) != saved["results"][method]:
            raise ValueError("paired group statistics replay differs")
    # Independent CPU rebuild profile in addition to the in-collection index timings.
    started = perf_counter()
    sparse = BM25Retriever(
        documents, k1=cfg.baseline.k1, b=cfg.baseline.b, epsilon=cfg.baseline.epsilon
    )
    sparse_build_ms = (perf_counter() - started) * 1000
    # Check complete vectors from the same rank_bm25 call as the original BM25 wrapper.
    matrix = np.load(root / "bm25_base_scores.npy", allow_pickle=False, mmap_mode="r")
    for index, q in enumerate(selected):
        if not np.array_equal(sparse.index.get_scores(by_query[q].query_ko.split()), matrix[index]):
            raise ValueError("BM25 full vector replay differs")
    gpu_replay = None
    if args.earlier_run is not None:
        earlier = args.earlier_run
        earlier_manifest = json.loads((earlier / "manifest.json").read_text())
        verify(earlier, earlier_manifest)
        earlier_summary = json.loads((earlier / "summary.json").read_text())
        for method in cfg.methods:
            for relative in [
                f"{method}_base_scores.npy",
                f"{method}_temporal_diagnostics.jsonl",
                *[
                    f"{name}/{file}"
                    for name in [method, f"{method}_timelyrag_auto"]
                    for file in ["run.trec", "qrels.tsv", "config.json"]
                ],
            ]:
                if sha256_file(earlier / relative) != sha256_file(root / relative):
                    raise ValueError("GPU replay scores/ranks/qrels/configs/alpha differ")

            def non_timing(row):
                return {
                    k: v
                    for k, v in row.items()
                    if k
                    not in {
                        "source_fast_mean_ms",
                        "incremental_temporal_mean_ms",
                        "modeled_additive_mean_ms",
                    }
                }

            if non_timing(earlier_summary["results"][method]) != non_timing(
                saved["results"][method]
            ):
                raise ValueError("GPU replay quality/group intervals differ")
        gpu_replay = {
            "scores_rankings_qrels_configs_alpha_exact": True,
            "quality_and_group_intervals_identical": True,
            "earlier_manifest_sha256": sha256_file(earlier / "manifest.json"),
            "scope": (
                "final run adds a tie-callback guard and real index-build telemetry; "
                "ranking unchanged"
            ),
        }
    public = {
        "measured_on": str(args.measured_on),
        "config": cfg.model_dump(mode="json"),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "source_files_sha256": manifest["source_files_sha256"],
        "packages": manifest["packages"],
        "python": manifest["python"],
        "summary": saved,
        "official_metric_columns": official_columns,
        "reference_source_manifest": reference.manifest,
        "verification": {
            "exact_base_and_auto_rank_score_alpha_replay_query_pairs": replay_queries,
            "official_per_query_metric_parity": True,
            "original_auto_function_top10_parity_pairs": replay_queries,
            "official_metric_absolute_tolerance": 1e-12,
            "paired_group_statistics_replay_exact": True,
            "bm25_full_score_vectors_replayed": len(selected),
            "second_gpu_run": gpu_replay is not None,
            "gpu_replay": gpu_replay,
            "oracle_on_benchmark_queries": False,
            "test_partition_evaluated": False,
        },
        "build_profile": {
            "bm25_cpu_build_ms": sparse_build_ms,
            "scope": "independent same-corpus CPU rebuild after collection; not live query time",
            "faiss_build_ms_in_collection": saved["encoder_cache"]["faiss_build_ms"],
            "index_build_ms_in_collection": saved.get("index_build_ms"),
        },
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "limitations": [
            "matched local validation, not reproduction of published whole-benchmark numbers",
            "E5/hybrid are backbone substitutions, not BGE-M3 results",
            "original NumPy reverse-argsort tie policy differs from previous local doc-ID ties",
            "prefix-normalized metrics are separate from full-qrel primary nDCG",
            "document event times absent; query event times are heuristic, not new gold",
            "unknown/later-or-equal insertions get neutral affinity, not exclusion",
            "auto stage has no index build; source index build and cached encoding are separate",
            "stage means measured separately; combined time is modeled, not live latency",
            "per-stage encoder context precedes queries; final root profile contains actual peak",
            "source/model redistribution and independent decontamination remain unresolved",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, public)
    print(
        json.dumps(
            {
                "verified_pairs": replay_queries,
                "results": saved["results"],
                "official_columns": official_columns,
                "build_ms": sparse_build_ms,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
