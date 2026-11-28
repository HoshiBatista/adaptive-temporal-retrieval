"""Frozen train500 QET intervention on cached hybrid pools; no neural inference or tuning."""

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import FunctionType
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from temporalproof.data.source import (
    ARCHIVE_SHA256,
    DEFAULT_ARCHIVE,
    load_archive_records,
    sha256_file,
)
from temporalproof.data.splits import DEFAULT_SPLIT, SplitManifest, select_queries, verify_split
from temporalproof.data.timelyqa import DatasetError
from temporalproof.evaluation.baseline import _write_json, evaluate_query, source_snapshot
from temporalproof.evaluation.pilot import read_rows, write_rows
from temporalproof.evaluation.query_time_audit import AuditConfig
from temporalproof.evaluation.rank_blend import METRICS, load_trec, verify_source
from temporalproof.retrieval.bm25 import Hit
from temporalproof.retrieval.qet_policy import role_aware_rerank
from temporalproof.retrieval.timelyrag import automatic_rerank, load_reference, native_top_indices
from temporalproof.routing.cross_validation import grouped_interval
from temporalproof.routing.targets import training_groups

METHODS = ("fast", "native_fast", "native_auto", "native_auto_same_bypass", "role_qet")
ALL_METRICS = (*METRICS, "reference_prefix_ndcg_at_10")


class AblationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    partition: Literal["train"]
    seed: Literal[1729]
    query_count: Literal[500]
    top_k: Literal[100]
    bootstrap_samples: int = Field(strict=True, ge=100)
    audit_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    frozen_policy_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    reference_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    train_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    pool_axis: Literal["original_corpus_order_restricted_to_frozen_top100"]
    primary: Literal["role_qet_vs_unchanged_fast"]
    train_gate: Literal["positive_mean_and_group_interval_lower_above_zero_vs_fast"]
    validation_evaluation_allowed: Literal[False]
    test_evaluation_allowed: Literal[False]
    oracle_alpha: Literal[False]
    new_neural_inference: Literal[False]


class AutoBoundary(Exception):
    def __init__(self, ranking):
        self.ranking = list(ranking)


class GoldAccessForbidden:
    """Reach the original first metric boundary without supplying/reading any gold IDs."""

    def __bool__(self):
        return True

    def __iter__(self):
        raise AssertionError("native parity must not access relevance judgments")


def original_auto_prefix(reference, scores, ids, dits, dets, query, qit, qet):
    """Execute unchanged original auto body in a private namespace; stop before scoring.

    No shared reference globals are modified. Original best-alpha must never be reached.
    """
    namespace = dict(reference.original_functions)

    def capture(ranking, positives, cutoff):
        if not isinstance(positives, GoldAccessForbidden):
            raise AssertionError("parity received unexpected labels")
        raise AutoBoundary(ranking)

    def forbid_oracle(*args, **kwargs):
        raise AssertionError("oracle execution forbidden")

    namespace["evaluate_ranking_ids"] = capture
    namespace["rerank_with_alpha_grid"] = forbid_oracle
    original = reference.original_functions["evaluate_2stage"]
    function = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
    try:
        function(
            "frozen_pool_parity",
            lambda _: scores,
            [{"query": query, "positive_ids": GoldAccessForbidden(), "qit": qit, "qet": qet}],
            ids,
            dets,
            dits,
        )
    except AutoBoundary as boundary:
        return boundary.ranking
    raise DatasetError("native auto boundary not reached")


def canonical_train_ids(split, config, plan):
    ids = select_queries(split, "train", config.query_count, config.seed)
    if (
        len(ids) != config.query_count
        or plan["selected_query_ids"] != ids
        or plan["validation_partition_evaluated"] is not False
        or plan["test_partition_evaluated"] is not False
    ):
        raise DatasetError(
            "requires the complete canonical train500 source, without held-out scoring"
        )
    return ids


def paired_comparison(rows, groups, candidate, control, config, ids=None):
    selected = ids if ids is not None else [r["query_id"] for r in rows[candidate]]
    a = {r["query_id"]: r for r in rows[candidate]}
    b = {r["query_id"]: r for r in rows[control]}
    delta = [a[q]["ndcg_at_10"] - b[q]["ndcg_at_10"] for q in selected]
    return {
        "candidate": candidate,
        "control": control,
        "query_count": len(selected),
        "mean_ndcg_delta": float(np.mean(delta)),
        "paired_group_bootstrap_95_interval": grouped_interval(
            delta,
            [groups[q] for q in selected],
            config.seed,
            config.bootstrap_samples,
        ),
        "gains_regressions_unchanged": [
            sum(d > 1e-12 for d in delta),
            sum(d < -1e-12 for d in delta),
            sum(abs(d) <= 1e-12 for d in delta),
        ],
    }


def summarize(rows, diagnostics, groups, config):
    summary = {
        "partition": "train",
        "query_count": len(rows["fast"]),
        "group_count": len(set(groups.values())),
        "methods": {
            name: {metric: float(np.mean([r[metric] for r in values])) for metric in ALL_METRICS}
            for name, values in rows.items()
        },
        "comparisons": {},
        "policy": {
            "rerank_called_count": sum(d["role"]["temporal_rerank_called"] for d in diagnostics),
            "reason_counts": dict(Counter(d["role"]["selection"]["reason"] for d in diagnostics)),
        },
        "native_auto_parity_count": len(diagnostics),
        "override_auto_parity_count": sum(d["role"]["temporal_rerank_called"] for d in diagnostics),
        "all_candidate_pools_preserved": True,
        "all_recall_at_100_preserved": True,
        "validation_evaluated": False,
        "test_evaluated": False,
        "new_neural_inference": False,
        "new_router_fit": False,
        "oracle_alpha": False,
        "interpretation": (
            "Development train-only diagnostic after train-text engineering, not held-out proof. "
            "Intervals condition on the fixed policy/pools; no role gold or applicability claim."
        ),
    }
    for candidate, control in [
        ("role_qet", "fast"),
        ("role_qet", "native_auto"),
        ("role_qet", "native_auto_same_bypass"),
        ("native_auto_same_bypass", "fast"),
        ("native_auto", "native_fast"),
        ("native_fast", "fast"),
    ]:
        summary["comparisons"][f"{candidate}_vs_{control}"] = paired_comparison(
            rows,
            groups,
            candidate,
            control,
            config,
        )
    active = [d["query_id"] for d in diagnostics if d["role"]["temporal_rerank_called"]]
    if active:
        summary["active_query_diagnostic"] = paired_comparison(
            rows,
            groups,
            "role_qet",
            "native_auto_same_bypass",
            config,
            active,
        )
    primary = summary["comparisons"]["role_qet_vs_fast"]
    summary["train_gate_passed"] = (
        primary["mean_ndcg_delta"] > 0 and primary["paired_group_bootstrap_95_interval"][0] > 0
    )
    summary["next_validation_allowed_by_gate"] = summary["train_gate_passed"]
    return summary


def measure(function, *args, **kwargs):
    start = perf_counter()
    result = function(*args, **kwargs)
    return result, (perf_counter() - start) * 1000


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/m4_qet_ablation_train.json")
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--train-run", type=Path, default=Path("artifacts/m4/company_policies_train500_v1")
    )
    parser.add_argument(
        "--audit-run", type=Path, default=Path("artifacts/m4/query_time_roles_train_v2")
    )
    parser.add_argument("--reference-source", type=Path, default=DEFAULT_ARCHIVE.parent)
    parser.add_argument(
        "--reference-manifest", type=Path, default=Path("configs/datasets/timelyrag_reference.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = AblationConfig.model_validate_json(args.config.read_text())
        if args.output.exists():
            raise DatasetError("output exists; choose a new directory")
        source = verify_source(args.train_run, config.train_manifest_sha256)
        audit = verify_source(args.audit_run, config.audit_manifest_sha256)
        if sha256_file(args.audit_run / "frozen_policy.json") != config.frozen_policy_sha256:
            raise DatasetError("frozen policy differs")
        policy = json.loads((args.audit_run / "frozen_policy.json").read_text())
        module = Path(__file__).parents[1] / "retrieval/query_time.py"
        if sha256_file(module) != policy["query_time_module_sha256"]:
            raise DatasetError("date-role implementation changed after freezing")
        audit_config = AuditConfig.model_validate_json((args.audit_run / "config.json").read_text())
        if (
            audit_config.train500_manifest_sha256 != config.train_manifest_sha256
            or audit_config.reference_manifest_sha256 != config.reference_manifest_sha256
        ):
            raise DatasetError("audit/reference/train provenance differs")
        records = load_archive_records(args.archive)
        split = SplitManifest.model_validate_json(args.split.read_text())
        verify_split(split, records, ARCHIVE_SHA256)
        split_hash = sha256_file(args.split)
        if any(
            m["split_sha256"] != split_hash or m["archive_sha256"] != ARCHIVE_SHA256
            for m in [source, audit]
        ):
            raise DatasetError("source archive/split differs")
        source_plan = json.loads((args.train_run / "plan.json").read_text())
        ids = canonical_train_ids(split, config, source_plan)
        fast = load_trec(args.train_run / "fast/run.trec", ids, config.top_k)
        by_query = {r.index: r for r in records}
        docs = [d for r in records for d in r.docs_ko]
        by_doc = {d.global_doc_id: d for d in docs}
        corpus_order = {d.global_doc_id: i for i, d in enumerate(docs)}
        qrels = "".join(
            f"{q}\t0\t{d}\t1\n" for q in ids for d in sorted(by_query[q].gold_docs_final_global)
        )
        if qrels.encode() != (args.train_run / "fast/qrels.tsv").read_bytes():
            raise DatasetError("cached qrels differ from original published records")
        reference = load_reference(
            args.reference_source, args.reference_manifest, config.reference_manifest_sha256
        )
        frozen_rows = {r["query_id"]: r for r in read_rows(args.audit_run / "query_audit.jsonl")}
        old_rows = {r["query_id"]: r for r in read_rows(args.train_run / "fast/per_query.jsonl")}
        groups_all = training_groups(split)
        groups = {q: groups_all[q] for q in ids}
        root = args.output
        root.mkdir(parents=True, exist_ok=False)
        provenance = {
            **source_snapshot(Path.cwd()),
            "archive_sha256": ARCHIVE_SHA256,
            "split_sha256": split_hash,
            "config_sha256": sha256_file(args.config),
        }
        _write_json(root / "config.json", config.model_dump())
        _write_json(root / "frozen_policy.json", policy)
        _write_json(
            root / "plan.json",
            {
                "config": config.model_dump(),
                "provenance": provenance,
                "selected_query_ids": ids,
                "query_groups": {str(q): groups[q] for q in ids},
                "methods": METHODS,
                "secondary_controls": {
                    "native_fast": (
                        "native NumPy sorting on the same restricted corpus axis; ties only"
                    ),
                    "native_auto_same_bypass": (
                        "native auto on explicit-selected queries, unchanged fast otherwise"
                    ),
                },
                "pool_scope": (
                    "fixed source top100; no full-corpus retrieval "
                    "or outside-pool score reconstruction"
                ),
                "timing_scope": (
                    "incremental warm CPU callbacks only, serial concurrency1; "
                    "excludes I/O, parity, metrics and original retrieval"
                ),
                "selection": "no fit/grid; unchanged frozen train-engineered policy",
                "validation_evaluated": False,
                "test_evaluated": False,
            },
        )
        rows = {name: [] for name in METHODS}
        rankings = {name: [] for name in METHODS}
        diagnostics, timings = [], {"native_auto": [], "role_qet": []}
        for q in ids:
            record = by_query[q]
            ordered = sorted(fast[q], key=lambda h: corpus_order[h.doc_id])
            pool_ids = [h.doc_id for h in ordered]
            scores = np.array([h.score for h in ordered], dtype=np.float64)
            dits = np.array(
                [reference.utils.to_dt(by_doc[d].doc_insert_time) for d in pool_ids], dtype=object
            )
            dets = np.array([None] * len(ordered), dtype=object)
            inputs = (
                reference,
                scores,
                pool_ids,
                dits,
                dets,
                record.query_ko,
                record.query_insert_time,
            )
            role_args = {
                "before_characters": audit_config.before_characters,
                "after_characters": audit_config.after_characters,
            }
            if q == ids[0]:
                for _ in range(3):
                    automatic_rerank(*inputs)
                    role_aware_rerank(*inputs, fast[q], **role_args)
            (auto, auto_info), native_ms = measure(automatic_rerank, *inputs)
            (role, role_info), role_ms = measure(role_aware_rerank, *inputs, fast[q], **role_args)
            if role_info["selection"] != frozen_rows[q]["analysis"]:
                raise DatasetError("selection differs from frozen train audit")
            qit = reference.utils.to_dt(record.query_insert_time)
            qet, _ = reference.utils.extract_qet_from_query(record.query_ko, qit_hint=qit)
            prefix = original_auto_prefix(
                reference, scores, pool_ids, dits, dets, record.query_ko, qit, qet
            )
            if prefix != [h.doc_id for h in auto[:10]]:
                raise DatasetError("native auto ranking parity failed")
            if role_info["temporal_rerank_called"]:
                explicit = datetime.fromisoformat(role_info["selection"]["selected_time"])
                expected = original_auto_prefix(
                    reference, scores, pool_ids, dits, dets, record.query_ko, qit, explicit
                )
                if expected != [h.doc_id for h in role[:10]]:
                    raise DatasetError("QET-only ranking differs from original kernel")
            elif role != fast[q]:
                raise DatasetError("abstention changed the original fast ranking")
            methods = {
                "fast": fast[q],
                "native_fast": [
                    Hit(pool_ids[i], float(scores[i]))
                    for i in native_top_indices(scores, len(scores))
                ],
                "native_auto": auto,
                "role_qet": role,
                "native_auto_same_bypass": auto if role_info["temporal_rerank_called"] else fast[q],
            }
            fast_metric = evaluate_query(record, fast[q])
            if any(abs(fast_metric[m] - old_rows[q][m]) > 1e-12 for m in ALL_METRICS):
                raise DatasetError("original fast metric replay failed")
            for name, hits in methods.items():
                if len(hits) != config.top_k or {h.doc_id for h in hits} != set(pool_ids):
                    raise DatasetError("candidate pool changed")
                metric = evaluate_query(record, hits)
                if metric["recall_at_100"] != fast_metric["recall_at_100"]:
                    raise DatasetError("candidate recall changed")
                expected_prefix = reference.metrics.evaluate_ranking_ids(
                    [h.doc_id for h in hits[:10]], set(record.gold_docs_final_global), 10
                )[3]
                if abs(metric["reference_prefix_ndcg_at_10"] - expected_prefix) > 1e-12:
                    raise DatasetError("official prefix metric mismatch")
                rows[name].append(metric)
                rankings[name].extend(
                    f"{q} Q0 {h.doc_id} {i} {h.score:.17g} {name}\n" for i, h in enumerate(hits, 1)
                )
            diagnostics.append({"query_id": q, "native": auto_info, "role": role_info})
            timings["native_auto"].append({"query_id": q, "incremental_cpu_ms": native_ms})
            timings["role_qet"].append({"query_id": q, "incremental_cpu_ms": role_ms})
        summary = summarize(rows, diagnostics, groups, config)
        _write_json(root / "summary.json", summary)
        write_rows(root / "diagnostics.jsonl", diagnostics)
        for name in METHODS:
            output = root / name
            output.mkdir()
            write_rows(output / "per_query.jsonl", rows[name])
            (output / "run.trec").write_text("".join(rankings[name]))
            (output / "qrels.tsv").write_text(qrels)
        for name, values in timings.items():
            write_rows(root / f"{name}_timing.jsonl", values)
        _write_json(
            root / "timing.json",
            {
                "scope": (
                    "incremental CPU callbacks; not live end-to-end/service latency or new GPU work"
                ),
                "warmups": 3,
                "concurrency": 1,
                "platform": platform.platform(),
                "python": platform.python_version(),
                "numpy": importlib.metadata.version("numpy"),
                "threads": {
                    k: os.environ.get(k)
                    for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
                },
                "methods": {
                    name: {
                        "mean_ms": float(np.mean([r["incremental_cpu_ms"] for r in values])),
                        "p50_ms": float(
                            np.quantile([r["incremental_cpu_ms"] for r in values], 0.5)
                        ),
                        "p95_ms": float(
                            np.quantile([r["incremental_cpu_ms"] for r in values], 0.95)
                        ),
                    }
                    for name, values in timings.items()
                },
            },
        )
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
        print(f"QET ablation failed: {exc}; evidence retained", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
