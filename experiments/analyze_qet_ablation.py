"""Validate full quality/decision replay; export train-only aggregates without source text."""

import argparse
import json
from datetime import date
from pathlib import Path

from temporalproof.data.source import sha256_file
from temporalproof.evaluation.baseline import _write_json, source_snapshot
from temporalproof.evaluation.pilot import read_rows
from temporalproof.evaluation.qet_ablation import ALL_METRICS, METHODS, AblationConfig
from temporalproof.evaluation.rank_blend import verify_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-on", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("aggregate exists; choose a new path")
    root = args.run
    fingerprint = sha256_file(root / "manifest.json")
    manifest = verify_source(root, fingerprint)
    replay = verify_source(args.replay, sha256_file(args.replay / "manifest.json"))
    current = source_snapshot(Path.cwd())
    if manifest["source_snapshot_sha256"] != current["source_snapshot_sha256"]:
        raise ValueError("current implementation differs from measured source")
    excluded = {"timing.json", "native_auto_timing.jsonl", "role_qet_timing.jsonl"}
    if set(manifest["files_sha256"]) != set(replay["files_sha256"]):
        raise ValueError("replay evidence schema differs")
    stable = sorted(set(manifest["files_sha256"]) - excluded)
    if any(manifest["files_sha256"][name] != replay["files_sha256"][name] for name in stable):
        raise ValueError("ranking/metrics/decisions/plan replay differs")
    if manifest["source_snapshot_sha256"] != replay["source_snapshot_sha256"]:
        raise ValueError("replay source differs")
    config = AblationConfig.model_validate_json((root / "config.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if summary["test_evaluated"] or summary["validation_evaluated"]:
        raise ValueError("held-out evaluation forbidden")
    rows = {m: {r["query_id"]: r for r in read_rows(root / m / "per_query.jsonl")} for m in METHODS}
    diagnostics = {r["query_id"]: r for r in read_rows(root / "diagnostics.jsonl")}
    ids = plan["selected_query_ids"]
    if set(diagnostics) != set(ids) or any(set(r) != set(ids) for r in rows.values()):
        raise ValueError("query selection differs")
    for method in METHODS:
        if (root / method / "qrels.tsv").read_bytes() != (root / "fast/qrels.tsv").read_bytes():
            raise ValueError("qrels differ")
        for q in ids:
            if rows[method][q]["recall_at_100"] != rows["fast"][q]["recall_at_100"]:
                raise ValueError("candidate recall changed")
        for metric in ALL_METRICS:
            mean = sum(rows[method][q][metric] for q in ids) / len(ids)
            if abs(mean - summary["methods"][method][metric]) > 1e-12:
                raise ValueError("aggregate metric differs")
    delta = {q: rows["role_qet"][q]["ndcg_at_10"] - rows["fast"][q]["ndcg_at_10"] for q in ids}
    examples = {}
    for name, selected in [
        (
            "largest_regressions",
            sorted((q for q in ids if delta[q] < -1e-12), key=lambda q: (delta[q], q))[:3],
        ),
        (
            "largest_gains",
            sorted((q for q in ids if delta[q] > 1e-12), key=lambda q: (-delta[q], q))[:3],
        ),
    ]:
        examples[name] = [
            {
                "query_id": q,
                "ndcg_delta": delta[q],
                "fast_ndcg": rows["fast"][q]["ndcg_at_10"],
                "role_ndcg": rows["role_qet"][q]["ndcg_at_10"],
                "selection_reason": diagnostics[q]["role"]["selection"]["reason"],
                "qet_changed": diagnostics[q]["native"]["query_event_time_derived"]
                != diagnostics[q]["role"]["selection"]["selected_time"],
            }
            for q in selected
        ]
    active = [q for q in ids if diagnostics[q]["role"]["temporal_rerank_called"]]
    coverage = {
        "queries_without_positive_in_pool": sum(rows["fast"][q]["recall_at_100"] == 0 for q in ids),
        "active_queries_without_positive_in_pool": sum(
            rows["fast"][q]["recall_at_100"] == 0 for q in active
        ),
        "active_qet_changes": sum(
            diagnostics[q]["native"]["query_event_time_derived"]
            != diagnostics[q]["role"]["selection"]["selected_time"]
            for q in active
        ),
    }
    _write_json(
        args.output,
        {
            "stage": "M4.5 frozen train500 QET-only intervention",
            "measured_on": args.measured_on.isoformat(),
            "manifest_sha256": fingerprint,
            "source_snapshot_sha256": manifest["source_snapshot_sha256"],
            "source_files_sha256": manifest["source_files_sha256"],
            "git_parent": manifest["git_commit"],
            "git_dirty_at_measurement": manifest["git_dirty"],
            "config_sha256": manifest["config_sha256"],
            "archive_sha256": manifest["archive_sha256"],
            "split_sha256": manifest["split_sha256"],
            "config": config.model_dump(),
            "summary": summary,
            "coverage": coverage,
            "examples_metadata_only": examples,
            "timing": json.loads((root / "timing.json").read_text()),
            "replay_timing": json.loads((args.replay / "timing.json").read_text()),
            "replay": {
                "stable_files_byte_identical": stable,
                "timing_excluded": sorted(excluded),
                "source_snapshot_identical": True,
                "rankings_and_decisions_identical": True,
            },
            "limits": [
                "train-engineered policy evaluated on train, not independent generalization",
                "fixed top100 only; NumPy ties apply on the restricted original-corpus axis",
                "no paper-backbone/full-corpus score reproduction, new GPU job or router fit",
                "unjudged IDs are not established irrelevant; no temporal-role/applicability gold",
                "primary train gate failed; no validation/test evaluation or default adoption",
                "timing measures incremental CPU callbacks, not original retrieval/live service",
            ],
        },
    )
    print(
        json.dumps(
            {
                "manifest_sha256": fingerprint,
                "replayed_files": len(stable),
                "train_gate_passed": summary["train_gate_passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
