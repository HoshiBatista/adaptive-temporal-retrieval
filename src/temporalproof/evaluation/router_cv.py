"""Fit/evaluate predeclared logistic routers using grouped OOF train data only."""

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

from temporalproof.data.source import ARCHIVE_SHA256, sha256_file
from temporalproof.data.splits import DEFAULT_SPLIT, SplitManifest, select_queries
from temporalproof.data.timelyqa import DatasetError
from temporalproof.evaluation.baseline import _write_json, source_snapshot
from temporalproof.evaluation.pilot import PilotConfig, read_rows, write_rows
from temporalproof.routing.cross_validation import (
    RouterConfig,
    control_scores,
    cross_fit,
    evaluate_routes,
    fold_assignment,
    grouped_interval,
    select_fold_budget,
)
from temporalproof.routing.features import FEATURE_VERSION, PreDeepFeatures
from temporalproof.routing.learned import fit_router
from temporalproof.routing.targets import TrainOutcome, training_groups


def load_training_bundle(root: Path, split_path: Path, config: RouterConfig):
    """Verify immutable source, pre-deep checkpoint, canonical groups and train IDs."""
    manifest = json.loads((root / "manifest.json").read_text())
    required = {
        "features.jsonl",
        "outcomes.jsonl",
        "plan.json",
        "config.json",
        "pre_deep_checkpoint.json",
        "fast/per_query.jsonl",
        "deep/run/per_query.jsonl",
    }
    if not required <= manifest["files_sha256"].keys():
        raise DatasetError("source training evidence incomplete")
    for name, digest in manifest["files_sha256"].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise DatasetError("source training artifact checksum/path mismatch")
    if (
        manifest["archive_sha256"] != ARCHIVE_SHA256
        or manifest["split_sha256"] != sha256_file(split_path)
        or manifest["pilot_config_sha256"] != config.source_pilot_config_sha256
        or not manifest["git_commit"].startswith(config.source_data_git_commit)
        or manifest["git_dirty"] is not False
        or manifest["source_snapshot_sha256"] != config.source_data_snapshot_sha256
    ):
        raise DatasetError("training source provenance mismatch")
    split = SplitManifest.model_validate_json(split_path.read_text())
    pilot = PilotConfig.model_validate_json((root / "config.json").read_text())
    if pilot.baseline.query_limit != config.query_count or pilot.baseline.seed != config.seed:
        raise DatasetError("training selection differs from predeclared config")
    ids = select_queries(split, "train", config.query_count, config.seed)
    groups = training_groups(split)
    plan = json.loads((root / "plan.json").read_text())
    if (
        plan["selected_query_ids"] != ids
        or plan["config"] != pilot.model_dump(mode="json")
        or plan["feature_version"] != FEATURE_VERSION
        or plan["test_partition_evaluated"] is not False
        or plan["validation_partition_evaluated"] is not False
    ):
        raise DatasetError("invalid training plan")
    checkpoint = json.loads((root / "pre_deep_checkpoint.json").read_text())
    if (
        checkpoint["features_sha256"] != sha256_file(root / "features.jsonl")
        or checkpoint["fast_manifest_sha256"] != sha256_file(root / "fast/manifest.json")
        or checkpoint["plan_sha256"] != sha256_file(root / "plan.json")
        or checkpoint["encoder_model_collected"] is not True
    ):
        raise DatasetError("features differ from pre-deep checkpoint")
    feature_rows = read_rows(root / "features.jsonl")
    outcome_rows = read_rows(root / "outcomes.jsonl")
    if [r["query_id"] for r in feature_rows] != ids or [r["query_id"] for r in outcome_rows] != ids:
        raise DatasetError("training feature/outcome query IDs differ from frozen selection")
    features = {}
    for row in feature_rows:
        if set(row) != {"query_id", "features"}:
            raise DatasetError("unexpected outer feature metadata")
        features[row["query_id"]] = PreDeepFeatures.model_validate(row["features"])
    outcomes = [TrainOutcome.model_validate(row) for row in outcome_rows]
    fast = {r["query_id"]: r for r in read_rows(root / "fast/per_query.jsonl")}
    deep = {r["query_id"]: r for r in read_rows(root / "deep/run/per_query.jsonl")}
    for row in outcomes:
        q = row.query_id
        if (
            q not in groups
            or row.group_id != groups[q]
            or plan["query_groups"][str(q)] != groups[q]
            or abs(row.gain - (row.deep_ndcg_at_10 - row.fast_ndcg_at_10)) > 1e-12
            or row.fast_ndcg_at_10 != fast[q]["ndcg_at_10"]
            or row.deep_ndcg_at_10 != deep[q]["ndcg_at_10"]
            or row.fast_ms != fast[q]["retrieval_ms"]
            or row.incremental_rerank_ms != deep[q]["retrieval_ms"]
        ):
            raise DatasetError("training-only groups/outcomes do not match source measurements")
    return features, outcomes, manifest


def summarize_classifier(predictions, outcomes, penalty):
    from sklearn.metrics import average_precision_score, roc_auc_score

    by_id = {r.query_id: r for r in outcomes}
    labels = np.array(
        [
            int(
                by_id[p["query_id"]].gain
                > penalty * by_id[p["query_id"]].incremental_rerank_ms / 1000 + 1e-12
            )
            for p in predictions
        ]
    )
    scores = np.array([p["score"] for p in predictions])
    ms = np.array([p["router_predict_ms"] for p in predictions])
    return {
        "positive_count": int(labels.sum()),
        "roc_auc_pooled_diagnostic": float(roc_auc_score(labels, scores))
        if len(set(labels)) == 2
        else None,
        "average_precision_diagnostic": float(average_precision_score(labels, scores))
        if labels.any()
        else None,
        "router_single_query_ms": {
            "mean": float(ms.mean()),
            "p50": float(np.quantile(ms, 0.5)),
            "p95": float(np.quantile(ms, 0.95)),
            "scope": (
                "warm CPU JSON-model inference including vector construction; "
                "concurrency 1; excludes feature extraction/files"
            ),
        },
        "score_scope": (
            "class-weighted logistic score, not calibrated probability; "
            "pooled scores use multiple fold models"
        ),
    }


def fit_evaluate(features, outcomes, config, folds):
    controls = []
    ids = set(features)
    for name, routed in [
        ("always_fast", set()),
        ("always_deep", ids),
        (
            "low_agreement",
            {
                q
                for q, f in features.items()
                if f.top10_overlap_fraction < config.disagreement_threshold
            },
        ),
        (
            "year_and_low_agreement",
            {
                q
                for q, f in features.items()
                if f.top10_overlap_fraction < config.disagreement_threshold
                and f.explicit_year_count > 0
            },
        ),
    ]:
        controls.append(evaluate_routes(name, routed, outcomes, config))
    for budget in config.query_route_budgets:
        for kind in ["random", "agreement"]:
            routed = select_fold_budget(control_scores(features, folds, config.seed, kind), budget)
            controls.append(evaluate_routes(f"{kind}_budget_{budget:g}", routed, outcomes, config))
    variants = []
    all_predictions = []
    all_models = []
    for penalty in config.penalties_ndcg_per_second:
        for kind in config.feature_sets:
            name = f"{kind}_lambda_{penalty:g}"
            predictions, models = cross_fit(features, outcomes, folds, config, penalty, kind)
            times = {p["query_id"]: p["router_predict_ms"] for p in predictions}
            policies = []
            for budget in config.query_route_budgets:
                policy = evaluate_routes(
                    f"{name}_budget_{budget:g}",
                    select_fold_budget(predictions, budget),
                    outcomes,
                    config,
                    times,
                )
                policy["query_budget_cap"] = budget
                policies.append(policy)
            threshold = {
                p["query_id"] for p in predictions if p["score"] >= config.probability_threshold
            }
            policies.append(
                evaluate_routes(
                    f"{name}_threshold_{config.probability_threshold:g}",
                    threshold,
                    outcomes,
                    config,
                    times,
                )
            )
            variants.append(
                {
                    "name": name,
                    "penalty": penalty,
                    "feature_set": kind,
                    "classifier": summarize_classifier(predictions, outcomes, penalty),
                    "policies": policies,
                }
            )
            all_predictions.extend({**p, "variant": name} for p in predictions)
            all_models.extend({**m, "variant": name} for m in models)
    primary_variant = next(
        v
        for v in variants
        if v["penalty"] == config.primary_penalty and v["feature_set"] == config.primary_feature_set
    )
    primary = next(
        p
        for p in primary_variant["policies"]
        if p.get("query_budget_cap") == config.primary_query_budget
    )
    matched = []
    for kind in ["random", "agreement"]:
        ref = next(
            p for p in controls if p["name"] == f"{kind}_budget_{config.primary_query_budget:g}"
        )
        a = set(primary["routed_query_ids"])
        b = set(ref["routed_query_ids"])
        deltas = [r.gain * (int(r.query_id in a) - int(r.query_id in b)) for r in outcomes]
        matched.append(
            {
                "control": ref["name"],
                "mean_ndcg_delta": float(np.mean(deltas)),
                "paired_group_bootstrap_95_interval": grouped_interval(
                    deltas, [r.group_id for r in outcomes], config.seed, config.bootstrap_samples
                ),
            }
        )
    final_labels = [
        int(r.gain > config.primary_penalty * r.incremental_rerank_ms / 1000 + 1e-12)
        for r in outcomes
    ]
    final_model, final_info = fit_router(
        [features[r.query_id] for r in outcomes],
        final_labels,
        kind=config.primary_feature_set,
        c=config.regularization_c,
        seed=config.seed,
        max_iter=config.max_iter,
        tol=config.tol,
    )
    return (
        {
            "controls": controls,
            "variants": variants,
            "primary": primary,
            "primary_matched_comparisons": matched,
            "scope": (
                "grouped out-of-fold on TRAIN only; fixed configuration, "
                "no validation/test selection"
            ),
            "bootstrap_scope": (
                "resamples source groups with fixed OOF predictions, without refitting; "
                "conditional diagnostic intervals"
            ),
            "final_model_fit_info": final_info,
        },
        all_predictions,
        all_models,
        final_model,
    )


def log_router_run(root):
    from temporalproof.tracking.local import DEFAULT_STORE, EXPERIMENT, local_client

    manifest = json.loads((root / "manifest.json").read_text())
    allowed = {
        "config.json",
        "plan.json",
        "summary.json",
        "oof_predictions.jsonl",
        "fold_models.json",
        "final_model.json",
        "data_manifest.json",
    }
    if set(manifest["files_sha256"]) != allowed:
        raise DatasetError("unexpected router artifact list")
    for name, digest in manifest["files_sha256"].items():
        if (
            not (root / name).resolve().is_relative_to(root.resolve())
            or sha256_file(root / name) != digest
        ):
            raise DatasetError("router output checksum mismatch")
    client = local_client()
    experiment = client.get_experiment_by_name(EXPERIMENT)
    expected_root = (DEFAULT_STORE / "artifacts" / EXPERIMENT).resolve().as_uri()
    if experiment is None:
        eid = client.create_experiment(EXPERIMENT, artifact_location=expected_root)
    else:
        if experiment.artifact_location != expected_root:
            raise DatasetError("router tracking experiment is not local")
        eid = experiment.experiment_id
    fingerprint = sha256_file(root / "manifest.json")
    previous = client.search_runs(
        [eid],
        filter_string=(
            f"tags.`temporalproof.router_fingerprint` = '{fingerprint}' "
            "AND attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    if previous:
        return previous[0].info.run_id
    summary = json.loads((root / "summary.json").read_text())
    rid = client.create_run(
        eid,
        tags={
            "mlflow.runName": manifest["config"]["name"],
            "temporalproof.phase": "M4_grouped_OOF_train_only",
            "temporalproof.router_fingerprint": fingerprint,
            "temporalproof.source_snapshot_sha256": manifest["source_snapshot_sha256"],
            "temporalproof.data_manifest_sha256": manifest["data_manifest_sha256"],
        },
    ).info.run_id
    try:
        for key, value in manifest["config"].items():
            client.log_param(
                rid, key, json.dumps(value) if isinstance(value, (dict, list)) else value
            )
        for key in [
            "ndcg_at_10",
            "mean_gain_vs_fast",
            "routed_fraction",
            "modeled_mean_with_router_ms",
        ]:
            client.log_metric(rid, f"primary.{key}", summary["primary"][key])
        for name in [*manifest["files_sha256"], "manifest.json"]:
            client.log_artifact(rid, str(root / name), artifact_path="router")
        client.set_terminated(rid, status="FINISHED")
    except Exception:
        client.set_terminated(rid, status="FAILED")
        raise
    return rid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/m4_logistic_groupcv_company_policies.json"),
    )
    parser.add_argument("--pilot-run", type=Path, required=True)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = RouterConfig.model_validate_json(args.config.read_text())
        if args.output.exists():
            raise DatasetError("output exists; choose a new router directory")
        features, outcomes, data_manifest = load_training_bundle(args.pilot_run, args.split, config)
        folds = fold_assignment(outcomes, config)
        root = args.output
        root.mkdir(parents=True, exist_ok=False)
        _write_json(root / "config.json", config.model_dump(mode="json"))
        _write_json(
            root / "plan.json",
            {
                "fold_by_query": {str(q): f for q, f in sorted(folds.items())},
                "group_by_query": {str(r.query_id): r.group_id for r in outcomes},
                "config": config.model_dump(mode="json"),
                "data_manifest_sha256": sha256_file(args.pilot_run / "manifest.json"),
                "feature_version": FEATURE_VERSION,
                "budget_rule": (
                    "floor(fold_size * query_cap), ranked within held-out fold only; "
                    "never by outcomes/cost"
                ),
            },
        )
        summary, predictions, models, final_model = fit_evaluate(features, outcomes, config, folds)
        summary.update(
            query_count=len(outcomes),
            group_count=len({r.group_id for r in outcomes}),
            test_partition_evaluated=False,
            validation_partition_evaluated=False,
            fitted_router=True,
        )
        _write_json(root / "summary.json", summary)
        write_rows(root / "oof_predictions.jsonl", predictions)
        _write_json(root / "fold_models.json", models)
        _write_json(
            root / "final_model.json",
            {
                "model": final_model.model_dump(),
                "scope": "fit on all selected train rows only; not evaluated on validation/test",
            },
        )
        _write_json(root / "data_manifest.json", data_manifest)
        _write_json(
            root / "manifest.json",
            {
                "config": config.model_dump(mode="json"),
                "config_file_sha256": sha256_file(args.config),
                "data_manifest_sha256": sha256_file(args.pilot_run / "manifest.json"),
                "data_source_snapshot_sha256": data_manifest["source_snapshot_sha256"],
                "packages": {
                    k: importlib.metadata.version(k)
                    for k in ["scikit-learn", "numpy", "scipy", "pydantic"]
                },
                **source_snapshot(Path.cwd()),
                "files_sha256": {
                    p.name: sha256_file(p) for p in sorted(root.iterdir()) if p.is_file()
                },
            },
        )
        if args.mlflow:
            _write_json(root / "tracking.json", {"mlflow_run_id": log_router_run(root)})
    except (OSError, ValueError, KeyError) as exc:
        message = (
            str(exc) if isinstance(exc, DatasetError) else "input/model/evidence validation failed"
        )
        print(f"router CV failed: {message}; saved evidence retained", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"primary": summary["primary"], "matched": summary["primary_matched_comparisons"]},
            sort_keys=True,
        )
    )
    print(f"Router artifacts: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
