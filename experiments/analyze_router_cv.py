"""Check grouped-CV replay and export aggregate metadata without model weights/text."""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from temporalproof.data.source import sha256_file
from temporalproof.evaluation.baseline import source_snapshot
from temporalproof.evaluation.pilot import read_rows
from temporalproof.routing.cross_validation import grouped_interval
from temporalproof.routing.features import PreDeepFeatures
from temporalproof.routing.learned import RouterModel, predict_scores


def verify(root):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files_sha256"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise ValueError("router evidence checksum/path mismatch")
    if manifest["source_snapshot_sha256"] != source_snapshot(Path.cwd())["source_snapshot_sha256"]:
        raise ValueError("router implementation differs from measurement")
    return manifest, json.loads((root / "summary.json").read_text())


def without_times(value):
    if isinstance(value, dict):
        return {
            k: without_times(v)
            for k, v in value.items()
            if k
            not in {
                "fit_ms",
                "router_predict_ms",
                "router_single_query_ms",
                "modeled_mean_with_router_ms",
            }
        }
    if isinstance(value, list):
        return [without_times(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-on", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("aggregate exists; choose a new path")
    manifest, summary = verify(args.run)
    _, replay = verify(args.replay)
    data_manifest = json.loads((args.data / "manifest.json").read_text())
    if sha256_file(args.data / "manifest.json") != manifest["data_manifest_sha256"]:
        raise ValueError("data manifest differs")
    for name, digest in data_manifest["files_sha256"].items():
        if sha256_file(args.data / name) != digest:
            raise ValueError("data evidence differs")
    for name in ["config.json", "plan.json", "final_model.json", "data_manifest.json"]:
        if (args.run / name).read_bytes() != (args.replay / name).read_bytes():
            raise ValueError(f"replay mismatch: {name}")
    original = read_rows(args.run / "oof_predictions.jsonl")
    again = read_rows(args.replay / "oof_predictions.jsonl")
    if without_times(original) != without_times(again):
        raise ValueError("OOF scores/folds differ")
    models = json.loads((args.run / "fold_models.json").read_text())
    repeated = json.loads((args.replay / "fold_models.json").read_text())
    if without_times(models) != without_times(repeated) or without_times(summary) != without_times(
        replay
    ):
        raise ValueError("models, controls or quality differ")
    features = {
        r["query_id"]: PreDeepFeatures.model_validate(r["features"])
        for r in read_rows(args.data / "features.jsonl")
    }
    scores = {(p["variant"], p["query_id"]): p["score"] for p in original}
    for item in models:
        state = RouterModel.model_validate(item["model"])
        ids = item["held_out_query_ids"]
        expected = [scores[(item["variant"], q)] for q in ids]
        if not np.allclose(
            predict_scores(state, [features[q] for q in ids]), expected, atol=1e-12, rtol=1e-10
        ):
            raise ValueError("serialized model predictions differ")
    outcomes = read_rows(args.data / "outcomes.jsonl")
    fast_rows = read_rows(args.data / "fast/per_query.jsonl")
    deep_rows = {r["query_id"]: r for r in read_rows(args.data / "deep/run/per_query.jsonl")}
    retrieval_metrics = {}
    for policy in [*summary["controls"], summary["primary"]]:
        selected = set(policy["routed_query_ids"])
        mixed = [deep_rows[r["query_id"]] if r["query_id"] in selected else r for r in fast_rows]
        metrics = {
            key: float(np.mean([r[key] for r in mixed]))
            for key in ["ndcg_at_10", "recall_at_10", "recall_at_20", "recall_at_100"]
        }
        if abs(metrics["ndcg_at_10"] - policy["ndcg_at_10"]) > 1e-12:
            raise ValueError("policy quality differs from source rankings")
        retrieval_metrics[policy["name"]] = metrics
    full = next(v for v in summary["variants"] if v["feature_set"] == "full" and v["penalty"] == 0)
    ablated = next(
        v for v in summary["variants"] if v["feature_set"] == "no_temporal" and v["penalty"] == 0
    )
    cap = manifest["config"]["primary_query_budget"]
    a = set(
        next(p for p in full["policies"] if p.get("query_budget_cap") == cap)["routed_query_ids"]
    )
    b = set(
        next(p for p in ablated["policies"] if p.get("query_budget_cap") == cap)["routed_query_ids"]
    )
    deltas = [r["gain"] * (int(r["query_id"] in a) - int(r["query_id"] in b)) for r in outcomes]
    aggregate = {
        "measured_on": args.measured_on.isoformat(),
        "summary": summary,
        "router_manifest": manifest,
        "router_plan": json.loads((args.run / "plan.json").read_text()),
        "data_summary": json.loads((args.data / "summary.json").read_text()),
        "data_manifest": data_manifest,
        "data_plan": json.loads((args.data / "plan.json").read_text()),
        "data_stage_manifests": {
            name: json.loads((args.data / path).read_text())
            for name, path in [("fast", "fast/manifest.json"), ("deep", "deep/run/manifest.json")]
        },
        "retrieval_metrics": retrieval_metrics,
        "explicit_temporal_ablation": {
            "scope": (
                "predeclared full minus no-explicit-temporal features, lambda 0, primary query cap"
            ),
            "mean_ndcg_delta": float(np.mean(deltas)),
            "paired_group_bootstrap_95_interval": grouped_interval(
                deltas,
                [r["group_id"] for r in outcomes],
                manifest["config"]["seed"],
                manifest["config"]["bootstrap_samples"],
            ),
        },
        "fold_audit": [
            {
                "variant": m["variant"],
                "fold": m["fold"],
                "fit_info": m["fit_info"],
                "train_group_count": m["train_group_count"],
                "held_group_count": m["held_group_count"],
                "fit_ms": m["fit_ms"],
            }
            for m in models
        ],
        "replay_verification": {
            "scores_folds_model_states_decisions_quality_exact": True,
            "final_model_and_plans_byte_identical": True,
            "summary": replay,
            "manifest_sha256": sha256_file(args.replay / "manifest.json"),
        },
        "tracking": json.loads((args.run / "tracking.json").read_text())
        if (args.run / "tracking.json").exists()
        else {},
        "analysis_script_sha256": sha256_file(Path(__file__)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(aggregate, sort_keys=True, indent=2) + "\n")
    print(
        json.dumps(
            {
                "replay": "verified",
                "primary": summary["primary"]["ndcg_at_10"],
                "temporal_ablation": aggregate["explicit_temporal_ablation"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
