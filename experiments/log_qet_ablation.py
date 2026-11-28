"""Idempotent metadata-only local MLflow import for the frozen train500 intervention."""

import argparse
import json
from pathlib import Path

from temporalproof.data.source import sha256_file
from temporalproof.evaluation.baseline import _write_json
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
    if public["manifest_sha256"] != fingerprint or public["summary"] != summary:
        raise ValueError("aggregate differs from measured evidence")
    if summary["validation_evaluated"] or summary["test_evaluated"]:
        raise ValueError("train-only import required")
    client = local_client()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    expected = (DEFAULT_STORE / "artifacts" / EXPERIMENT).resolve().as_uri()
    if experiment is None or experiment.artifact_location != expected:
        raise ValueError("local artifact store required")
    old = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.`temporalproof.qet_ablation_fingerprint` = '{fingerprint}' "
            "AND attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    if old:
        rid = old[0].info.run_id
    else:
        rid = client.create_run(
            experiment.experiment_id,
            tags={
                "mlflow.runName": "qet_override_train500_v1",
                "temporalproof.phase": "M4_5_frozen_train_qet_ablation",
                "temporalproof.qet_ablation_fingerprint": fingerprint,
                "temporalproof.source_snapshot_sha256": public["source_snapshot_sha256"],
            },
        ).info.run_id
        try:
            for name, value in {
                "partition": "train",
                "query_count": 500,
                "train_gate_passed": summary["train_gate_passed"],
                "new_neural_inference": False,
                "validation_evaluated": False,
                "test_evaluated": False,
                "oracle_alpha": False,
            }.items():
                client.log_param(rid, name, value)
            for method, values in summary["methods"].items():
                for metric, value in values.items():
                    client.log_metric(rid, f"{method}.{metric}", value)
            for method, values in public["timing"]["methods"].items():
                for metric, value in values.items():
                    client.log_metric(rid, f"{method}.incremental_cpu_{metric}", value)
            for name in [
                "config.json",
                "plan.json",
                "frozen_policy.json",
                "summary.json",
                "timing.json",
                "manifest.json",
            ]:
                client.log_artifact(rid, str(root / name), artifact_path="qet_ablation")
            client.log_artifact(rid, str(args.results), artifact_path="analysis")
            client.set_terminated(rid, status="FINISHED")
        except Exception:
            client.set_terminated(rid, status="FAILED")
            raise
    _write_json(root / "tracking.json", {"mlflow_run_id": rid})
    print(json.dumps({"mlflow_run_id": rid}))


if __name__ == "__main__":
    main()
