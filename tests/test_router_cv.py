"""Synthetic CPU tests for fold isolation, serialization and immutable data loading."""

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from temporalproof.data.source import ARCHIVE_SHA256, sha256_file
from temporalproof.data.splits import SplitManifest, select_queries
from temporalproof.data.timelyqa import DatasetError
from temporalproof.evaluation.pilot import PilotConfig, write_rows
from temporalproof.evaluation.router_cv import load_training_bundle
from temporalproof.routing.cross_validation import (
    RouterConfig,
    cross_fit,
    fold_assignment,
    grouped_interval,
    select_fold_budget,
)
from temporalproof.routing.features import FEATURE_VERSION, PreDeepFeatures
from temporalproof.routing.learned import RouterModel, feature_matrix, fit_router, predict_scores
from temporalproof.routing.targets import TrainOutcome, training_groups

pytest.importorskip("sklearn")


@pytest.fixture
def synthetic():
    split = SplitManifest(
        dataset_sha256=ARCHIVE_SHA256,
        seed=1729,
        groups={
            "train": [[i, i + 1] for i in range(0, 40, 2)],
            "validation": [[100]],
            "test": [[101]],
        },
        audit={},
    )
    groups = training_groups(split)
    features = {}
    outcomes = []
    for q in range(40):
        features[q] = PreDeepFeatures(
            query_characters=30 + q,
            query_whitespace_tokens=4 + q % 6,
            explicit_year_count=q % 3,
            top10_overlap_fraction=(q % 5) / 5,
            top1_agreement=q % 2 == 0,
            bm25_relative_margin=0.01 * (1 + q % 3),
            dense_margin=0.02 * (1 + q % 5),
            hybrid_relative_margin=0.01,
            hybrid_normalized_entropy=0.95,
            query_insertion_known=True,
            candidate_insertion_known_fraction=1,
            candidate_after_query_fraction=None,
            median_abs_insertion_gap_days=float(q) if q % 3 else None,
            candidate_distinct_insertion_days=4,
            candidate_unique_text_fraction=0.8,
        )
        deep = 0.7 if q % 3 == 0 else 0.1
        outcomes.append(
            TrainOutcome(
                query_id=q,
                group_id=groups[q],
                fast_ndcg_at_10=0.2,
                deep_ndcg_at_10=deep,
                gain=deep - 0.2,
                fast_ms=90,
                incremental_rerank_ms=1000,
                candidate_recall_at_100=0.5,
                cost_sensitive_positive={"0.0": q % 3 == 0},
            )
        )
    cfg = json.loads(
        Path("configs/experiments/m4_logistic_groupcv_company_policies.json").read_text()
    )
    cfg.update(query_count=40, bootstrap_samples=100)
    return split, features, outcomes, RouterConfig.model_validate(cfg)


def test_fold_assignment_uses_groups_not_targets_and_is_order_independent(synthetic):
    _, _, rows, config = synthetic
    folds = fold_assignment(rows, config)
    assert folds == fold_assignment(list(reversed(rows)), config)
    assert len(set(folds.values())) == 5
    for q in range(0, 40, 2):
        assert folds[q] == folds[q + 1]
    changed = [r.model_copy(update={"gain": -r.gain}) for r in rows]
    assert fold_assignment(changed, config) == folds


def test_fold_models_predictions_and_budgets_ignore_own_held_out_labels(synthetic):
    pytest.importorskip("sklearn")
    _, features, rows, config = synthetic
    folds = fold_assignment(rows, config)
    original, models = cross_fit(features, rows, folds, config, 0, "full")
    altered = [
        r.model_copy(update={"gain": -r.gain}) if folds[r.query_id] == 0 else r for r in rows
    ]
    repeated, changed_models = cross_fit(features, altered, folds, config, 0, "full")
    assert models[0]["model"] == changed_models[0]["model"]
    a = [p for p in original if p["fold"] == 0]
    b = [p for p in repeated if p["fold"] == 0]
    assert [p["score"] for p in a] == [p["score"] for p in b]
    assert select_fold_budget(a, 0.25) == select_fold_budget(b, 0.25)
    assert set(models[0]["train_query_ids"]).isdisjoint(models[0]["held_out_query_ids"])


def test_preprocessing_does_not_fit_held_out_feature_outliers(synthetic):
    pytest.importorskip("sklearn")
    _, features, rows, config = synthetic
    folds = fold_assignment(rows, config)
    _, a = cross_fit(features, rows, folds, config, 0, "full")
    changed = {
        q: f.model_copy(update={"query_characters": 1_000_000}) if folds[q] == 0 else f
        for q, f in features.items()
    }
    _, b = cross_fit(changed, rows, folds, config, 0, "full")
    assert a[0]["model"] == b[0]["model"]
    train = a[0]["train_query_ids"]
    model = a[0]["model"]
    column = model["feature_names"].index("query_characters")
    assert model["scaler_mean"][column] == pytest.approx(
        np.mean([features[q].query_characters for q in train])
    )


def test_json_inference_matches_pipeline_on_unseen_missing_and_empty_columns(synthetic):
    sklearn = pytest.importorskip("sklearn")
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    _, features, _, _ = synthetic
    train = [features[q] for q in range(20)]
    held = [
        features[q].model_copy(
            update={"median_abs_insertion_gap_days": None, "candidate_after_query_fraction": 1.0}
        )
        for q in range(20, 30)
    ]
    labels = [q % 2 for q in range(20)]
    model, _ = fit_router(train, labels, kind="full", c=1, seed=1729, max_iter=2000, tol=1e-6)
    restored = RouterModel.model_validate_json(model.model_dump_json())
    pipe = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(
            C=1, class_weight="balanced", solver="lbfgs", random_state=1729, max_iter=2000, tol=1e-6
        ),
    )
    pipe.fit(feature_matrix(train, model.feature_names), labels)
    assert np.allclose(
        predict_scores(restored, held),
        pipe.predict_proba(feature_matrix(held, model.feature_names))[:, 1],
        atol=1e-12,
    )
    assert sklearn.__version__


def test_single_class_fallback_is_explicit_and_finite(synthetic):
    pytest.importorskip("sklearn")
    _, features, _, _ = synthetic
    model, info = fit_router(
        list(features.values()),
        [0] * 40,
        kind="no_temporal",
        c=1,
        seed=1729,
        max_iter=100,
        tol=1e-6,
    )
    assert info["single_class_fallback"] and model.algorithm == "constant"
    assert predict_scores(model, list(features.values())).tolist() == [0] * 40


@pytest.mark.parametrize("name", ["gain", "query_id", "deep_score", "incremental_rerank_ms"])
def test_saved_model_cannot_select_outcome_or_id_columns(name):
    with pytest.raises(ValueError):
        RouterModel(algorithm="constant", feature_names=(name,), constant_probability=0.5)


def test_query_cap_is_per_fold_not_a_global_oof_ranking():
    rows = [{"query_id": i, "fold": i // 4, "score": 100 - i} for i in range(8)]
    assert select_fold_budget(rows, 0.25) == {0, 4}


def test_intervals_resample_whole_groups():
    assert grouped_interval([0, 1], ["same", "same"], 1729, 100) == [0.5, 0.5]
    assert grouped_interval([0, 1, 1], ["a", "a", "b"], 1729, 100) == [0.5, 1]


def test_cross_fit_rejects_group_overlap(synthetic):
    _, features, rows, config = synthetic
    folds = fold_assignment(rows, config)
    folds[0] = (folds[1] + 1) % 5
    with pytest.raises(ValueError, match="group crosses"):
        cross_fit(features, rows, folds, config, 0, "full")


@pytest.fixture
def source_bundle(tmp_path, synthetic):
    split, features, outcomes, config = synthetic
    config = config.model_copy(
        update={"source_data_git_commit": "a" * 40, "source_data_snapshot_sha256": "b" * 64}
    )
    root = tmp_path / "source"
    root.mkdir()
    (root / "fast").mkdir()
    (root / "deep/run").mkdir(parents=True)
    split_path = tmp_path / "split.json"
    split_path.write_text(split.model_dump_json())
    pilot = json.loads(Path("configs/experiments/m4_train500_company_policies.json").read_text())
    pilot["baseline"]["query_limit"] = 40
    pilot = PilotConfig.model_validate(pilot)
    (root / "config.json").write_text(pilot.model_dump_json())
    config = config.model_copy(
        update={"source_pilot_config_sha256": sha256_file(root / "config.json")}
    )
    ids = select_queries(split, "train", 40, 1729)
    by_id = {r.query_id: r for r in outcomes}
    plan = {
        "selected_query_ids": ids,
        "query_groups": {str(q): by_id[q].group_id for q in ids},
        "config": pilot.model_dump(mode="json"),
        "feature_version": FEATURE_VERSION,
        "test_partition_evaluated": False,
        "validation_partition_evaluated": False,
    }
    (root / "plan.json").write_text(json.dumps(plan))
    write_rows(
        root / "features.jsonl",
        [{"query_id": q, "features": features[q].model_dump()} for q in ids],
    )
    write_rows(root / "outcomes.jsonl", [by_id[q].model_dump() for q in ids])
    for stage in ["fast", "deep/run"]:
        write_rows(
            root / stage / "per_query.jsonl",
            [
                {
                    "query_id": q,
                    "ndcg_at_10": by_id[q].fast_ndcg_at_10
                    if stage == "fast"
                    else by_id[q].deep_ndcg_at_10,
                    "retrieval_ms": 90 if stage == "fast" else 1000,
                }
                for q in ids
            ],
        )
    (root / "fast/manifest.json").write_text("{}")
    (root / "pre_deep_checkpoint.json").write_text(
        json.dumps(
            {
                "features_sha256": sha256_file(root / "features.jsonl"),
                "fast_manifest_sha256": sha256_file(root / "fast/manifest.json"),
                "plan_sha256": sha256_file(root / "plan.json"),
                "encoder_model_collected": True,
            }
        )
    )
    manifest = {
        "archive_sha256": ARCHIVE_SHA256,
        "split_sha256": sha256_file(split_path),
        "pilot_config_sha256": config.source_pilot_config_sha256,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "source_snapshot_sha256": "b" * 64,
        "files_sha256": {
            str(p.relative_to(root)): sha256_file(p) for p in root.rglob("*") if p.is_file()
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, split_path, config


def rehash(root):
    checkpoint = json.loads((root / "pre_deep_checkpoint.json").read_text())
    checkpoint["features_sha256"] = sha256_file(root / "features.jsonl")
    (root / "pre_deep_checkpoint.json").write_text(json.dumps(checkpoint))
    m = json.loads((root / "manifest.json").read_text())
    m["files_sha256"] = {
        str(p.relative_to(root)): sha256_file(p)
        for p in root.rglob("*")
        if p.is_file() and p != root / "manifest.json"
    }
    (root / "manifest.json").write_text(json.dumps(m))


def test_training_loader_verifies_source_and_canonical_groups(source_bundle):
    root, split, config = source_bundle
    features, outcomes, _ = load_training_bundle(root, split, config)
    assert len(features) == len(outcomes) == 40
    assert {r.query_id for r in outcomes} == set(features)


@pytest.mark.parametrize("mutation", ["test_id", "feature_label", "group", "bytes"])
def test_training_loader_rejects_corruption_even_with_rehashed_metadata(source_bundle, mutation):
    root, split, config = source_bundle
    if mutation == "bytes":
        (root / "features.jsonl").write_text("tampered")
    elif mutation == "group":
        rows = [json.loads(line) for line in (root / "outcomes.jsonl").read_text().splitlines()]
        rows[0]["group_id"] = "wrong"
        write_rows(root / "outcomes.jsonl", rows)
        rehash(root)
    else:
        rows = [json.loads(line) for line in (root / "features.jsonl").read_text().splitlines()]
        if mutation == "test_id":
            rows[0]["query_id"] = 101
        else:
            rows[0]["features"]["gain"] = 1
        write_rows(root / "features.jsonl", rows)
        rehash(root)
    with pytest.raises((DatasetError, ValidationError)):
        load_training_bundle(root, split, config)
