"""Metadata-only local MLflow import; excludes raw queries and temporal annotations."""

import argparse
import json
from pathlib import Path

from temporalproof.data.source import sha256_file
from temporalproof.evaluation.baseline import _write_json
from temporalproof.evaluation.query_time_audit import AuditConfig
from temporalproof.evaluation.rank_blend import verify_source
from temporalproof.tracking.local import DEFAULT_STORE, EXPERIMENT, local_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    fingerprint = sha256_file(root / "manifest.json")
    verify_source(root, fingerprint)
    public = json.loads(args.results.read_text())
    summary = json.loads((root / "summary.json").read_text())
    config = AuditConfig.model_validate_json((root / "config.json").read_text())
    if public["manifest_sha256"] != fingerprint or public["summary"] != summary:
        raise ValueError("public aggregate differs from measured audit")
    if summary["test_queries_analyzed"] or summary["validation_queries_analyzed"]:
        raise ValueError("only train date audits may be imported here")
    client = local_client()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    expected = (DEFAULT_STORE / "artifacts" / EXPERIMENT).resolve().as_uri()
    if experiment is None or experiment.artifact_location != expected:
        raise ValueError("tracking artifact location must be the existing local store")
    previous = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.`temporalproof.date_audit_fingerprint` = '{fingerprint}' "
            "AND attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    if previous:
        rid = previous[0].info.run_id
    else:
        rid = client.create_run(
            experiment.experiment_id,
            tags={
                "mlflow.runName": config.name,
                "temporalproof.phase": "M4_4_train_only_date_roles",
                "temporalproof.date_audit_fingerprint": fingerprint,
                "temporalproof.source_snapshot_sha256": public["source_snapshot_sha256"],
            },
        ).info.run_id
        try:
            for name, value in {
                "partition": "train",
                "query_count": config.query_count,
                "inspected_groups": config.sample_count,
                "new_neural_inference": False,
                "retrieval_evaluated": False,
                "temporal_role_gold_available": False,
            }.items():
                client.log_param(rid, name, value)
            for name, value in {
                "selected_precise_anchor_count": summary["selection_action_counts"].get(
                    "use_explicit_point", 0
                ),
                "fast_bypass_count": summary["selection_action_counts"].get("keep_fast_ranking", 0),
                "reference_with_employment_cue_count": summary[
                    "reference_selected_with_employment_cue_count"
                ],
                "pool_unknown_insert_count": summary["candidate_pool_coverage"][
                    "insertion_unknown_count"
                ],
            }.items():
                client.log_metric(rid, name, value)
            # Explicit allowlist: never upload query_audit.jsonl / inspection40.json.
            for name in [
                "config.json",
                "plan.json",
                "summary.json",
                "frozen_policy.json",
                "manifest.json",
            ]:
                client.log_artifact(rid, str(root / name), artifact_path="audit")
            client.log_artifact(rid, str(args.results), artifact_path="analysis")
            client.set_terminated(rid, status="FINISHED")
        except Exception:
            client.set_terminated(rid, status="FAILED")
            raise
    _write_json(root / "tracking.json", {"mlflow_run_id": rid})
    print(json.dumps({"mlflow_run_id": rid}))


if __name__ == "__main__":
    main()
