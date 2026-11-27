"""CPU-only frozen inference, held-out isolation and candidate-checkpoint contracts."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from temporalproof.data.source import ARCHIVE_SHA256, sha256_file
from temporalproof.data.splits import SplitManifest
from temporalproof.data.timelyqa import DatasetError, Document, QueryRecord
from temporalproof.evaluation.baseline import _write_json, source_snapshot
from temporalproof.evaluation.pilot import collect_fast, read_rows, write_rows
from temporalproof.evaluation.reranking import RerankingConfig, load_candidate_pool, run_reranking
from temporalproof.evaluation.router_validation import (
    ValidationConfig,
    checkpoint,
    evaluate,
    load_frozen_model,
    make_decisions,
    validation_ids,
    verify_checkpoint,
)
from temporalproof.retrieval.bm25 import Hit
from temporalproof.routing.features import extract_features
from temporalproof.routing.learned import RouterModel, feature_names
from temporalproof.routing.targets import training_groups


@pytest.fixture
def cfg():
    return ValidationConfig.model_validate_json(
        Path("configs/experiments/m4_frozen_validation_company_policies.json").read_text()
    )


@pytest.fixture
def split():
    return SplitManifest(
        dataset_sha256=ARCHIVE_SHA256,
        seed=1729,
        groups={"train": [[0]], "validation": [[1, 2], [3], [4]], "test": [[5]]},
        audit={},
    )


def model():
    return RouterModel(
        algorithm="constant", feature_names=feature_names("full"), constant_probability=0.6
    )


@pytest.mark.parametrize("partition", ["train", "test"])
def test_validation_config_rejects_other_partitions(cfg, partition):
    raw = cfg.model_dump()
    raw["baseline"]["partition"] = partition
    with pytest.raises(ValidationError):
        ValidationConfig.model_validate(raw)


def test_complete_validation_required(cfg, split):
    with pytest.raises(DatasetError, match="complete validation"):
        validation_ids(split, cfg)


def fake_feature(q):
    docs = {
        i: Document(global_doc_id=i, text=f"independent synthetic passage {i}") for i in range(2)
    }
    hits = [Hit(0, 1.0), Hit(1, 0.5)]
    return extract_features(f"synthetic {q}", None, hits, hits, hits, docs).model_copy(
        update={"top10_overlap_fraction": q / 10}
    )


def test_global_quota_matches_controls_and_id_tie_break(cfg):
    features = {q: fake_feature(q) for q in range(10)}
    predictions, decisions = make_decisions(features, model(), cfg)
    assert decisions["logistic_budget_0.1"] == [0]
    assert len(decisions["random_budget_0.1"]) == len(decisions["agreement_budget_0.1"]) == 1
    assert decisions["agreement_budget_0.1"] == [0]
    assert decisions["logistic_budget_0"] == []
    assert decisions["logistic_budget_1"] == list(range(10))
    assert [p["score"] for p in predictions] == [0.6] * 10
    _, repeated = make_decisions(dict(reversed(list(features.items()))), model(), cfg)
    assert repeated == decisions


def training_bundle(tmp_path, cfg, split):
    root = tmp_path / "training"
    root.mkdir()
    train_cfg = dict(
        primary_feature_set="full",
        primary_penalty=0,
        primary_query_budget=0.1,
        probability_threshold=0.5,
        disagreement_threshold=0.2,
        query_route_budgets=[0, 0.1, 0.25, 0.5, 1],
        seed=1729,
        query_count=1,
    )
    _write_json(root / "config.json", train_cfg)
    _write_json(root / "final_model.json", {"model": model().model_dump()})
    _write_json(root / "plan.json", {"group_by_query": {"0": training_groups(split)[0]}})
    _write_json(
        root / "summary.json",
        {"validation_partition_evaluated": False, "test_partition_evaluated": False},
    )
    for name in ["data_manifest.json", "fold_models.json", "oof_predictions.jsonl"]:
        (root / name).write_text("{}\n")
    _write_json(
        root / "manifest.json",
        {
            "config": train_cfg,
            "source_files_sha256": source_snapshot(Path.cwd())["source_files_sha256"],
            "files_sha256": {p.name: sha256_file(p) for p in root.iterdir()},
        },
    )
    return root, cfg.model_copy(
        update={
            "baseline": cfg.baseline.model_copy(update={"query_limit": 4}),
            "training_manifest_sha256": sha256_file(root / "manifest.json"),
            "training_config_sha256": sha256_file(root / "config.json"),
            "final_model_sha256": sha256_file(root / "final_model.json"),
        }
    )


def test_frozen_loader_uses_train_and_never_fits(tmp_path, cfg, split, monkeypatch):
    import temporalproof.routing.learned as learned

    monkeypatch.setattr(learned, "fit_router", lambda *a, **k: pytest.fail("unexpected fit"))
    root, cfg = training_bundle(tmp_path, cfg, split)
    assert load_frozen_model(root, split, cfg) == model()
    (root / "final_model.json").write_text("{}")
    with pytest.raises(DatasetError, match="checksum"):
        load_frozen_model(root, split, cfg)


@pytest.mark.parametrize("mutation", ["held_out", "groups", "source", "policy"])
def test_frozen_loader_rejects_rehashed_wrong_scope(tmp_path, cfg, split, mutation):
    root, cfg = training_bundle(tmp_path, cfg, split)
    manifest = json.loads((root / "manifest.json").read_text())
    if mutation == "held_out":
        _write_json(
            root / "summary.json",
            {"validation_partition_evaluated": True, "test_partition_evaluated": False},
        )
    elif mutation == "groups":
        _write_json(root / "plan.json", {"group_by_query": {"1": "held-out"}})
    elif mutation == "source":
        manifest["source_files_sha256"]["src/temporalproof/routing/features.py"] = "f" * 64
    else:
        raw = json.loads((root / "config.json").read_text())
        raw["primary_penalty"] = 0.05
        _write_json(root / "config.json", raw)
        manifest["config"] = raw
    manifest["files_sha256"] = {n: sha256_file(root / n) for n in manifest["files_sha256"]}
    _write_json(root / "manifest.json", manifest)
    cfg = cfg.model_copy(
        update={
            "training_manifest_sha256": sha256_file(root / "manifest.json"),
            "training_config_sha256": sha256_file(root / "config.json"),
        }
    )
    with pytest.raises(DatasetError):
        load_frozen_model(root, split, cfg)


def test_fake_validation_labels_do_not_change_predeep_decisions(tmp_path, cfg, split, monkeypatch):
    import temporalproof.routing.learned as learned

    monkeypatch.setattr(learned, "fit_router", lambda *a, **k: pytest.fail("unexpected fit"))
    cfg = cfg.model_copy(
        update={
            "baseline": cfg.baseline.model_copy(
                update={"query_limit": 4, "bootstrap_samples": 100, "warmup_queries": 1}
            )
        }
    )
    records = [
        QueryRecord(
            index=q,
            category="synthetic",
            query_ko=f"independent {q}",
            answer="synthetic",
            docs_ko=[
                Document(global_doc_id=q * 100 + i, text=f"synthetic {q}:{i}") for i in range(100)
            ],
            gold_docs_final=[0],
            gold_docs_final_global=[q * 100],
        )
        for q in range(6)
    ]
    saved = []
    for attempt in range(2):
        root = tmp_path / str(attempt)
        root.mkdir()
        if attempt:
            records = [
                r.model_copy(
                    update={"gold_docs_final": [99], "gold_docs_final_global": [r.index * 100 + 99]}
                )
                for r in records
            ]
        _write_json(root / "config.json", cfg.model_dump())
        _write_json(root / "plan.json", {"selected_query_ids": validation_ids(split, cfg)})
        _write_json(root / "frozen_model.json", model().model_dump())

        def search(text, cutoff):
            q = int(text.split()[-1])
            hits = [Hit(q * 100 + i, float(100 - i)) for i in range(cutoff)]
            return hits, hits

        features = collect_fast(
            records,
            split,
            cfg,
            root,
            {
                "archive_sha256": ARCHIVE_SHA256,
                "split_sha256": "b" * 64,
                "source_snapshot_sha256": "c" * 64,
            },
            search,
        )
        assert set(features) == {1, 2, 3, 4}
        predictions, decisions = make_decisions(features, model(), cfg)
        saved.append((features, decisions))
        write_rows(root / "predictions.jsonl", predictions)
        _write_json(root / "decisions.json", decisions)
        checkpoint(root, 0)
        deep_cfg = RerankingConfig(
            baseline=cfg.baseline.model_copy(update={"method": "fixed_cross_encoder"}),
            model_id=cfg.reranker_model_id,
            revision=cfg.reranker_revision,
            batch_size=16,
            max_length=512,
            candidate_manifest_sha256=sha256_file(root / "fast/manifest.json"),
        )
        pool = load_candidate_pool(root / "fast", records, split, deep_cfg, "b" * 64)

        class FakeScorer:
            def score(self, query, passages, evidence_root=root):
                verify_checkpoint(evidence_root)
                assert not query.endswith(("0", "5"))
                return [float(int(p.split(":")[-1])) for p in passages]

            def profile(self):
                return {"device": "cpu-fake"}

        run_reranking(
            records, split, deep_cfg, pool, FakeScorer(), root / "deep", {"split_sha256": "b" * 64}
        )
        summary = evaluate(root, split, cfg)
        assert json.loads(json.dumps(summary)) == summary
        assert summary["decision_replay_verified"]
        assert summary["test_partition_evaluated"] is False
        assert summary["router_training_targets_generated"] is False
        assert not (root / "outcomes.jsonl").exists()
        assert {p["recall_at_100"] for p in summary["policies"]} == {1.0}
        assert len(read_rows(root / "deep/run/per_query.jsonl")) == 4
        (root / "decisions.json").write_text("{}")
        with pytest.raises(DatasetError, match="checkpoint"):
            evaluate(root, split, cfg)
    assert saved[0] == saved[1]
