"""Log six validated stage runs plus metadata-only reference comparison to local MLflow."""

import argparse
import json
from pathlib import Path

from temporalproof.data.source import sha256_file
from temporalproof.evaluation.baseline import _write_json
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
        p = root / name
        if not p.resolve().is_relative_to(root.resolve()) or sha256_file(p) != digest:
            raise ValueError("reference evidence checksum/path mismatch")
    public = json.loads(args.results.read_text())
    fingerprint = sha256_file(root / "manifest.json")
    if public["manifest_sha256"] != fingerprint or public["summary"]["test_partition_evaluated"]:
        raise ValueError("aggregate scope/source differs")
    ids = {}
    for method in ["bm25", "dense", "hybrid_rrf"]:
        for name in [method, f"{method}_timelyrag_auto"]:
            ids[name] = log_run_directory(root / name)
    client = local_client()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    expected = (DEFAULT_STORE / "artifacts" / EXPERIMENT).resolve().as_uri()
    if experiment is None or experiment.artifact_location != expected:
        raise ValueError("reference tracking location is not local")
    previous = client.search_runs(
        [experiment.experiment_id],
        filter_string=(
            f"tags.`temporalproof.reference_fingerprint` = '{fingerprint}' "
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
                "mlflow.runName": "timelyrag_native_auto_validation495_v1",
                "temporalproof.phase": "M4_3_reference_code_matched_validation",
                "temporalproof.reference_fingerprint": fingerprint,
                "temporalproof.source_snapshot_sha256": public["source_snapshot_sha256"],
            },
        ).info.run_id
        try:
            for key, value in {
                "partition": "validation",
                "query_count": 495,
                "oracle_evaluation": False,
                "paper_whole_benchmark_reproduced": False,
                "dense_backbone": "E5-small CUDA",
            }.items():
                client.log_param(rid, key, value)
            for method, result in public["summary"]["results"].items():
                for key in [
                    "mean_ndcg_delta",
                    "incremental_temporal_mean_ms",
                    "source_fast_mean_ms",
                ]:
                    client.log_metric(rid, f"{method}.{key}", result[key])
                for kind in ["base_metrics", "auto_metrics"]:
                    for metric, value in result[kind].items():
                        client.log_metric(rid, f"{method}.{kind}.{metric}", value)
            for name in [
                "config.json",
                "plan.json",
                "reference_manifest.json",
                "summary.json",
                "manifest.json",
            ]:
                client.log_artifact(rid, str(root / name), artifact_path="reference")
            for p in [args.results, args.figure, args.figure.with_suffix(".svg")]:
                client.log_artifact(rid, str(p), artifact_path="analysis")
            client.set_terminated(rid, status="FINISHED")
        except Exception:
            client.set_terminated(rid, status="FAILED")
            raise
    ids["comparison"] = rid
    _write_json(root / "tracking.json", {"mlflow_run_ids": ids})
    print(json.dumps(ids, sort_keys=True))


if __name__ == "__main__":
    main()
