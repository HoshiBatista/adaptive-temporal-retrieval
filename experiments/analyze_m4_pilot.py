"""Verify two pilot runs and export aggregate evidence without raw text/target rows."""

import argparse
import json
from datetime import date
from pathlib import Path

from temporalproof.data.source import sha256_file
from temporalproof.evaluation.baseline import source_snapshot
from temporalproof.evaluation.comparison import paired_comparisons
from temporalproof.evaluation.pilot import read_rows


def load_verified(root: Path) -> tuple[dict, dict]:
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise ValueError("pilot artifact checksum/path mismatch")
    checkpoint = manifest["pre_deep_checkpoint"]
    if checkpoint["features_sha256"] != sha256_file(root / "features.jsonl"):
        raise ValueError("pre-deep features differ")
    if not checkpoint["encoder_model_collected"]:
        raise ValueError("sequential model guard not verified")
    return json.loads((root / "summary.json").read_text()), manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-on", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("aggregate exists; choose a new output")
    summary, manifest = load_verified(args.run)
    replay, replay_manifest = load_verified(args.replay)
    current = source_snapshot(Path.cwd())["source_snapshot_sha256"]
    if (
        manifest["source_snapshot_sha256"] != current
        or replay_manifest["source_snapshot_sha256"] != current
    ):
        raise ValueError("measured implementation differs from current source")
    for name in [
        "config.json",
        "plan.json",
        "features.jsonl",
        "fast/config.json",
        "fast/run.trec",
        "fast/qrels.tsv",
        "deep/run/run.trec",
        "deep/run/qrels.tsv",
    ]:
        if (args.run / name).read_bytes() != (args.replay / name).read_bytes():
            raise ValueError(f"replay mismatch: {name}")
    for stage in ["fast_summary", "deep_summary"]:
        if summary[stage]["metrics"] != replay[stage]["metrics"]:
            raise ValueError("replay quality differs")
    if summary["paired_deltas"] != replay["paired_deltas"]:
        raise ValueError("replay paired deltas differ")
    original = {r["query_id"]: r for r in read_rows(args.run / "outcomes.jsonl")}
    other = {r["query_id"]: r for r in read_rows(args.replay / "outcomes.jsonl")}
    if original.keys() != other.keys():
        raise ValueError("outcome query IDs differ")
    for q, row in original.items():
        for key in [
            "gain",
            "group_id",
            "fast_ndcg_at_10",
            "deep_ndcg_at_10",
            "candidate_recall_at_100",
        ]:
            if row[key] != other[q][key]:
                raise ValueError("outcome quality/group differs")
    label_changes = {
        key: sum(
            row["cost_sensitive_positive"][key] != other[q]["cost_sensitive_positive"][key]
            for q, row in original.items()
        )
        for key in next(iter(original.values()))["cost_sensitive_positive"]
    }
    fast = read_rows(args.run / "fast/per_query.jsonl")
    deep = {r["query_id"]: r for r in read_rows(args.run / "deep/run/per_query.jsonl")}
    paired = {}
    for policy in summary["policies"]:
        if policy["name"] in {"low_agreement", "year_and_low_agreement"}:
            ids = set(policy["routed_query_ids"])
            mixed = [deep[r["query_id"]] if r["query_id"] in ids else r for r in fast]
            paired[policy["name"]] = paired_comparisons(
                fast,
                mixed,
                summary["config"]["baseline"]["seed"],
                summary["config"]["baseline"]["bootstrap_samples"],
            )
    aggregate = {
        "summary": summary,
        "measured_on": args.measured_on.isoformat(),
        "manifest": manifest,
        "plan": json.loads((args.run / "plan.json").read_text()),
        "policy_paired_deltas": paired,
        "stage_manifests": {
            name: json.loads((args.run / path).read_text())
            for name, path in [("fast", "fast/manifest.json"), ("deep", "deep/run/manifest.json")]
        },
        "replay_verification": {
            "config_plan_features_fast_deep_rankings_scores_qrels_byte_identical": True,
            "quality_gains_groups_paired_deltas_identical": True,
            "cost_sensitive_label_changes_by_lambda": label_changes,
            "summary": replay,
            "manifest_sha256": sha256_file(args.replay / "manifest.json"),
            "pre_deep_checkpoint": replay_manifest["pre_deep_checkpoint"],
        },
        "analysis_script_sha256": sha256_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"replay_label_changes": label_changes, "utility": summary["utility"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
