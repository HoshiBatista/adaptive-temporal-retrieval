"""Verify exact train-audit replay and export only aggregates and provenance."""

import argparse
import json
from datetime import date
from pathlib import Path

from temporalproof.data.source import sha256_file
from temporalproof.evaluation.baseline import _write_json, source_snapshot
from temporalproof.evaluation.query_time_audit import AuditConfig
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
    if manifest != replay:
        raise ValueError("audit replay is not byte-identical")
    current = source_snapshot(Path.cwd())
    if manifest["source_snapshot_sha256"] != current["source_snapshot_sha256"]:
        raise ValueError("current audit implementation differs from the measured source")
    config = AuditConfig.model_validate_json((root / "config.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (
        summary["validation_queries_analyzed"]
        or summary["test_queries_analyzed"]
        or summary["new_neural_inference"]
        or summary["new_router_fit"]
        or len(plan["selected_train_query_ids"]) != config.query_count
        or len(plan["inspection_sample_ids"]) != config.sample_count
    ):
        raise ValueError("audit scope differs from the train-only plan")
    _write_json(
        args.output,
        {
            "stage": "M4.4 train-only date-role audit, not retrieval evaluation",
            "measured_on": args.measured_on.isoformat(),
            "manifest_sha256": fingerprint,
            "source_snapshot_sha256": manifest["source_snapshot_sha256"],
            "source_files_sha256": manifest["source_files_sha256"],
            "config_file_sha256": manifest["config_file_sha256"],
            "archive_sha256": manifest["archive_sha256"],
            "split_sha256": manifest["split_sha256"],
            "reference_manifest_sha256": config.reference_manifest_sha256,
            "train500_manifest_sha256": config.train500_manifest_sha256,
            "config": config.model_dump(),
            "summary": summary,
            "frozen_policy": json.loads((root / "frozen_policy.json").read_text()),
            "frozen_policy_sha256": sha256_file(root / "frozen_policy.json"),
            "inspection_query_ids": plan["inspection_sample_ids"],
            "replay": {
                "all_manifested_files_byte_identical": True,
                "manifest_byte_identical": fingerprint
                == sha256_file(args.replay / "manifest.json"),
                "verified_file_count": len(manifest["files_sha256"]),
            },
            "interpretation": (
                "Role cues are heuristic coverage, not gold or an error-rate estimate. "
                "No retrieval score, quality improvement or temporal correctness was measured. "
                "Local raw query/date annotations are excluded from this aggregate."
            ),
        },
    )
    print(
        json.dumps({"manifest_sha256": fingerprint, "replay": "exact", "output": str(args.output)})
    )


if __name__ == "__main__":
    main()
