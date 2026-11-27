"""Log verified frozen validation outputs into the existing local MLflow store."""

import argparse
import json
from pathlib import Path

from temporalproof.data.source import sha256_file
from temporalproof.evaluation.baseline import _write_json
from temporalproof.evaluation.router_validation import ValidationConfig, verify_checkpoint
from temporalproof.tracking.local import DEFAULT_STORE, EXPERIMENT, local_client, log_run_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise ValueError("validation evidence checksum/path mismatch")
    verify_checkpoint(root)
    aggregate = json.loads(args.results.read_text())
    fingerprint = sha256_file(root / "manifest.json")
    if aggregate["manifest_sha256"] != fingerprint:
        raise ValueError("aggregate references a different run")
    config = ValidationConfig.model_validate_json((root / "config.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    if aggregate["summary"] != summary or summary["test_partition_evaluated"]:
        raise ValueError("validation summary scope differs")
    ids = {"fast": log_run_directory(root / "fast"), "deep": log_run_directory(root / "deep/run")}
    client = local_client()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    expected = (DEFAULT_STORE / "artifacts" / EXPERIMENT).resolve().as_uri()
    if experiment is None or experiment.artifact_location != expected:
        raise ValueError("tracking experiment is not local")
    previous = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.`temporalproof.frozen_validation_fingerprint` = '{fingerprint}' "
            "AND attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    if previous:
        ids["router"] = previous[0].info.run_id
    else:
        rid = client.create_run(
            experiment.experiment_id,
            tags={
                "mlflow.runName": "frozen_logistic_validation495_v1",
                "temporalproof.phase": "M4_frozen_development_validation",
                "temporalproof.frozen_validation_fingerprint": fingerprint,
                "temporalproof.final_model_sha256": config.final_model_sha256,
                "temporalproof.source_snapshot_sha256": manifest["source_snapshot_sha256"],
            },
        ).info.run_id
        try:
            for k, v in {
                "partition": "validation",
                "query_count": summary["query_count"],
                "query_budget": config.primary_query_budget,
                "router_refitted": False,
                "test_partition_evaluated": False,
                "training_manifest_sha256": config.training_manifest_sha256,
            }.items():
                client.log_param(rid, k, v)
            for policy in summary["policies"]:
                for metric in [
                    "ndcg_at_10",
                    "recall_at_10",
                    "recall_at_20",
                    "recall_at_100",
                    "routed_fraction",
                    "modeled_mean_with_router_ms",
                ]:
                    client.log_metric(rid, f"{policy['name']}.{metric}", policy[metric])
            # Explicit metadata-only list: no weights, source passage text or raw failure cases.
            for name in [
                "config.json",
                "plan.json",
                "pre_deep_checkpoint.json",
                "summary.json",
                "manifest.json",
                "decisions.json",
                "predictions.jsonl",
            ]:
                client.log_artifact(rid, str(root / name), artifact_path="frozen_router")
            for path in [args.results, args.figure, args.figure.with_suffix(".svg")]:
                client.log_artifact(rid, str(path), artifact_path="analysis")
            client.set_terminated(rid, status="FINISHED")
        except Exception:
            client.set_terminated(rid, status="FAILED")
            raise
        ids["router"] = rid
    _write_json(root / "tracking.json", {"mlflow_run_ids": ids})
    print(json.dumps(ids, sort_keys=True))


if __name__ == "__main__":
    main()
