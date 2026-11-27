"""Log checked cached-blend metadata/figures to the existing local MLflow experiment."""

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
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    fingerprint = sha256_file(root / "manifest.json")
    verify_source(root, fingerprint)
    public = json.loads(args.results.read_text())
    summary = json.loads((root / "summary.json").read_text())
    if (
        public["manifest_sha256"] != fingerprint
        or public["summary"] != summary
        or not public["exact_replay_all_files_including_manifest"]
        or summary["test_partition_evaluated"]
    ):
        raise ValueError("aggregate/evidence scope mismatch")
    client = local_client()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    expected = (DEFAULT_STORE / "artifacts" / EXPERIMENT).resolve().as_uri()
    if experiment is None or experiment.artifact_location != expected:
        raise ValueError("tracking experiment is not local")
    previous = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.`temporalproof.rankblend_fingerprint` = '{fingerprint}' "
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
                "mlflow.runName": "conservative_rankblend_train500_validation495_v1",
                "temporalproof.phase": "M4_2_cached_rankblend",
                "temporalproof.rankblend_fingerprint": fingerprint,
                "temporalproof.source_snapshot_sha256": public["source_snapshot_sha256"],
            },
        ).info.run_id
        try:
            for key, value in {
                "selected_weight": summary["validation"]["selected_weight"],
                "weight_selection": "train500 only; five grouped folds",
                "router_refitted": False,
                "test_partition_evaluated": False,
                "new_neural_inference": False,
            }.items():
                client.log_param(rid, key, value)
            client.log_metric(
                rid, "oof.primary.ndcg_at_10", summary["train_cv"]["primary"]["ndcg_at_10"]
            )
            for p in summary["validation"]["policies"]:
                for metric in [
                    "ndcg_at_10",
                    "recall_at_10",
                    "recall_at_20",
                    "recall_at_100",
                    "mean_ndcg_delta_vs_fast",
                    "modeled_source_stage_mean_ms",
                ]:
                    client.log_metric(rid, f"validation.{p['name']}.{metric}", p[metric])
            # Excludes inspection text, raw qrels, candidate scores and model parameters.
            for name in [
                "config.json",
                "plan.json",
                "train_summary.json",
                "final_selection.json",
                "train_checkpoint.json",
                "summary.json",
                "manifest.json",
            ]:
                client.log_artifact(rid, str(root / name), artifact_path="rankblend")
            for path in [args.results, args.figure, args.figure.with_suffix(".svg")]:
                client.log_artifact(rid, str(path), artifact_path="analysis")
            client.set_terminated(rid, status="FINISHED")
        except Exception:
            client.set_terminated(rid, status="FAILED")
            raise
    _write_json(root / "tracking.json", {"mlflow_run_id": rid})
    print(rid)


if __name__ == "__main__":
    main()
