"""Synthetic CPU checks for rank-only inference and train-only weight selection."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from temporalproof.data.source import sha256_file
from temporalproof.data.timelyqa import DatasetError
from temporalproof.evaluation.baseline import _write_json
from temporalproof.evaluation.rank_blend import (
    BlendConfig,
    RankingBundle,
    choose_weight,
    diagnose,
    grouped_selection,
    load_trec,
    metrics,
    summarize_policy,
    validation_evaluation,
    verify_source,
    weight_metrics,
)
from temporalproof.retrieval.bm25 import Hit
from temporalproof.retrieval.rank_blend import blend_ranks


@pytest.fixture
def config():
    return BlendConfig.model_validate_json(
        Path("configs/experiments/m4_conservative_rankblend_company_policies.json").read_text()
    ).model_copy(
        update={
            "train_query_count": 12,
            "validation_query_count": 12,
            "n_splits": 3,
            "bootstrap_samples": 100,
        }
    )


@pytest.fixture
def bundle():
    ids = list(range(12))
    fast, deep, qrels = {}, {}, {}
    for q in ids:
        docs = list(range(q * 100, (q + 1) * 100))
        fast[q] = [Hit(d, float(100 - i)) for i, d in enumerate(docs)]
        swapped = docs.copy()
        # Weak deep improvements/losses visible to small rank interpolation weights.
        swapped[1], swapped[2] = swapped[2], swapped[1]
        deep[q] = [Hit(d, float(100 - i)) for i, d in enumerate(swapped)]
        qrels[q] = {docs[2] if q % 3 else docs[1]}
    rows = [
        {q: {"query_id": q, **metrics(method[q], qrels[q]), "retrieval_ms": cost} for q in ids}
        for method, cost in [(fast, 80), (deep, 1000)]
    ]
    return RankingBundle("train", ids, {q: f"group{q // 2}" for q in ids}, fast, deep, qrels, *rows)


def test_rank_blend_endpoints_monotone_and_scale_invariant():
    fast = [Hit(1, 100), Hit(2, 90), Hit(3, 80)]
    deep = [Hit(3, 10), Hit(2, 9), Hit(1, 8)]
    assert blend_ranks(fast, deep, 0) == fast
    assert blend_ranks(fast, deep, 1) == deep
    assert [h.doc_id for h in blend_ranks(fast, deep, 0.1)] == [1, 2, 3]
    assert [h.doc_id for h in blend_ranks(fast, deep, 0.9)] == [3, 2, 1]
    scaled = [Hit(h.doc_id, h.score * 1e8 - 5) for h in deep]
    assert blend_ranks(fast, deep, 0.2) == blend_ranks(fast, scaled, 0.2)
    # Symmetric equal ranks tie by document ID, not iteration order.
    assert [h.doc_id for h in blend_ranks([Hit(2, 2), Hit(1, 1)], [Hit(1, 2), Hit(2, 1)], 0.5)] == [
        1,
        2,
    ]


@pytest.mark.parametrize("weight", [-0.1, 1.1, float("nan"), float("inf"), True])
def test_bad_weights_fail(weight):
    with pytest.raises(ValueError):
        blend_ranks([Hit(0, 1)], [Hit(0, 1)], weight)


@pytest.mark.parametrize("deep", [[], [Hit(0, 1), Hit(0, 0)], [Hit(2, 1)], [Hit(0, float("nan"))]])
def test_pool_or_score_corruption_fails_even_at_zero_weight(deep):
    with pytest.raises(ValueError):
        blend_ranks([Hit(0, 1)], deep, 0)


def test_config_rejects_posthoc_broad_or_missing_zero_grid(config):
    for grid in [[0.1, 0.2], [0, 0.5], [0, 0.1, 0.1], [0, 0.2, 0.1], [0, float("nan")]]:
        with pytest.raises(ValidationError):
            BlendConfig.model_validate({**config.model_dump(), "candidate_weights": grid})


def test_choice_ties_prefer_fast_and_requires_only_declared_train_rows():
    # A poison held-out object must not be read by the selector.
    grid = {0.0: {0: {"ndcg_at_10": 0.5}, 1: object()}, 0.1: {0: {"ndcg_at_10": 0.5}, 1: object()}}
    assert choose_weight(grid, [0], 1e-12) == 0
    grid[0.1][0]["ndcg_at_10"] += 5e-13
    assert choose_weight(grid, [0], 1e-12) == 0
    grid[0.1][0]["ndcg_at_10"] += 1e-3
    assert choose_weight(grid, [0], 1e-12) == 0.1


def test_grouped_selection_excludes_own_heldout_labels_and_keeps_all_metrics(bundle, config):
    folds = {q: (q // 2) % 3 for q in bundle.ids}
    original, selections, rows = grouped_selection(bundle, folds, config)
    changed = copy.deepcopy(bundle)
    for q in changed.ids:
        if folds[q] == 0:
            changed.qrels[q] = {changed.fast[q][0].doc_id}
            changed.fast_rows[q].update(metrics(changed.fast[q], changed.qrels[q]))
            changed.deep_rows[q].update(metrics(changed.deep[q], changed.qrels[q]))
    _, repeated, changed_rows = grouped_selection(changed, folds, config)
    assert selections[0] == repeated[0]
    assert all(
        r["query_id"] not in selections[0]["selection_query_ids"] for r in rows if r["fold"] == 0
    )
    assert [r["selected_weight"] for r in rows if r["fold"] == 0] == [
        r["selected_weight"] for r in changed_rows if r["fold"] == 0
    ]
    assert {r["recall_at_100"] for r in rows} == {1.0}
    assert json.loads(json.dumps(original)) == original
    assert original["primary"]["routed_count"] <= len(bundle.ids)


def test_group_split_and_validation_training_refused(bundle, config):
    folds = {q: q % 3 for q in bundle.ids}
    with pytest.raises(DatasetError, match="crosses folds"):
        grouped_selection(bundle, folds, config)
    bundle.partition = "validation"
    with pytest.raises(DatasetError, match="training"):
        grouped_selection(bundle, {q: 0 for q in bundle.ids}, config)


def test_fixed_positive_weight_does_not_reduce_neural_call_cost(bundle, config):
    rows = weight_metrics(bundle, 0.05, config.rank_offset)
    summary = summarize_policy(bundle, rows, set(bundle.ids), config, "blend")
    assert summary["modeled_source_stage_mean_ms"] == 1080
    assert summary["routed_count"] == 12
    assert {r["recall_at_100"] for r in rows.values()} == {1.0}
    assert metrics(blend_ranks(bundle.fast[0], bundle.deep[0], 0), bundle.qrels[0]) == metrics(
        bundle.fast[0], bundle.qrels[0]
    )


def test_zero_weight_validation_is_real_fast_bypass_and_old_routes_stay_fixed(bundle, config):
    bundle.partition = "validation"
    decisions = {f"{kind}_budget_0.1": [3] for kind in ["logistic", "random", "agreement"]}
    result = validation_evaluation(bundle, 0.0, config, decisions)
    assert result["validation_weight_search"] is False
    assert result["router_refitted"] is False
    for policy in result["policies"]:
        if policy["name"] == "raw_deep":
            continue
        assert policy["routed_count"] == 0
        assert policy["modeled_source_stage_mean_ms"] == 80
        assert policy["mean_ndcg_delta_vs_fast"] == 0
    for policy in result["policies"][-3:]:
        assert policy["source_requested_calls"] == 1
    with pytest.raises(DatasetError):
        validation_evaluation(bundle, 0.25, config, decisions)
    decisions["logistic_budget_0.1"] = [999]
    with pytest.raises(DatasetError):
        validation_evaluation(bundle, 0.0, config, decisions)


def test_no_gold_pool_ceiling_and_judgment_limits(bundle):
    q = bundle.ids[0]
    bundle.qrels[q] = {99999}
    bundle.fast_rows[q].update(metrics(bundle.fast[q], bundle.qrels[q]))
    bundle.deep_rows[q].update(metrics(bundle.deep[q], bundle.qrels[q]))
    diagnostic = diagnose(bundle)
    assert diagnostic["pool_coverage"]["no_gold_in_pool"]["count"] == 1
    assert diagnostic["pool_coverage"]["all_gold_in_pool"]["count"] == 11
    assert "never a deployable" in diagnostic["oracle_scope"]
    assert "not established semantically irrelevant" in diagnostic["judgment_scope"]
    assert json.loads(json.dumps(diagnostic)) == diagnostic


def test_trec_refuses_unknown_queries_bad_rank_duplicate_and_unsorted(tmp_path):
    path = tmp_path / "run.trec"
    for content in [
        "9 Q0 1 1 2 m\n",
        "0 Q0 1 2 2 m\n",
        "0 Q0 1 1 2 m\n0 Q0 1 2 1 m\n",
        "0 Q0 1 1 1 m\n0 Q0 2 2 2 m\n",
    ]:
        path.write_text(content)
        with pytest.raises(DatasetError):
            load_trec(path, [0], 2)


def test_source_manifest_pin_and_output_tamper_refused(tmp_path):
    (tmp_path / "summary.json").write_text("{}")
    _write_json(
        tmp_path / "manifest.json",
        {"files_sha256": {"summary.json": sha256_file(tmp_path / "summary.json")}},
    )
    pin = sha256_file(tmp_path / "manifest.json")
    assert verify_source(tmp_path, pin)
    (tmp_path / "summary.json").write_text('{"changed":true}')
    with pytest.raises(DatasetError, match="checksum"):
        verify_source(tmp_path, pin)
    with pytest.raises(DatasetError, match="pin"):
        verify_source(tmp_path, "f" * 64)


def test_loader_replays_official_metrics_and_rejects_rehashed_relabeling(tmp_path, bundle, config):
    from temporalproof.data.source import ARCHIVE_SHA256
    from temporalproof.data.splits import SplitManifest, select_queries
    from temporalproof.data.timelyqa import Document, QueryRecord
    from temporalproof.evaluation.pilot import write_rows
    from temporalproof.evaluation.rank_blend import load_bundle
    from temporalproof.routing.targets import training_groups

    split = SplitManifest(
        dataset_sha256=ARCHIVE_SHA256,
        seed=config.seed,
        groups={
            "train": [[q, q + 1] for q in range(0, 12, 2)],
            "validation": [[20]],
            "test": [[21]],
        },
        audit={},
    )
    ids = select_queries(split, "train", 12, config.seed)
    records = [
        QueryRecord(
            index=q,
            category="synthetic",
            query_ko=f"independent {q}",
            answer="synthetic",
            docs_ko=[
                Document(global_doc_id=h.doc_id, text=f"passage {h.doc_id}") for h in bundle.fast[q]
            ],
            gold_docs_final=[next(iter(bundle.qrels[q])) - q * 100],
            gold_docs_final_global=sorted(bundle.qrels[q]),
        )
        for q in bundle.ids
    ]
    for method, dirname, rows in [
        (bundle.fast, "fast", bundle.fast_rows),
        (bundle.deep, "deep/run", bundle.deep_rows),
    ]:
        root = tmp_path / dirname
        root.mkdir(parents=True)
        (root / "run.trec").write_text(
            "".join(
                f"{q} Q0 {h.doc_id} {i} {h.score} synthetic\n"
                for q in ids
                for i, h in enumerate(method[q], 1)
            )
        )
        (root / "qrels.tsv").write_text(
            "".join(f"{q}\t0\t{d}\t1\n" for q in ids for d in sorted(bundle.qrels[q]))
        )
        write_rows(root / "per_query.jsonl", [rows[q] for q in ids])
    _write_json(
        tmp_path / "plan.json",
        {
            "selected_query_ids": ids,
            "test_partition_evaluated": False,
            "query_groups": {str(q): training_groups(split)[q] for q in ids},
        },
    )

    def repin():
        files = {
            str(p.relative_to(tmp_path)): sha256_file(p)
            for p in tmp_path.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        _write_json(
            tmp_path / "manifest.json", {"archive_sha256": ARCHIVE_SHA256, "files_sha256": files}
        )
        return config.model_copy(
            update={"train_manifest_sha256": sha256_file(tmp_path / "manifest.json")}
        )

    cfg = repin()
    loaded = load_bundle(tmp_path, split, records, cfg, "train")
    assert loaded.ids == ids
    assert loaded.qrels == bundle.qrels
    assert loaded.fast_rows == bundle.fast_rows
    for name in ["fast/qrels.tsv", "deep/run/qrels.tsv"]:
        path = tmp_path / name
        text = path.read_text().replace(
            f"{ids[0]}\t0\t{next(iter(bundle.qrels[ids[0]]))}\t1", f"{ids[0]}\t0\t99999\t1"
        )
        path.write_text(text)
    with pytest.raises(DatasetError, match="official"):
        load_bundle(tmp_path, split, records, repin(), "train")
    with pytest.raises(DatasetError, match="test"):
        load_bundle(tmp_path, split, records, cfg, "test")


def test_validation_evaluates_only_the_frozen_weight(bundle, config, monkeypatch):
    import temporalproof.evaluation.rank_blend as module

    bundle.partition = "validation"
    calls = []
    real = module.weight_metrics

    def recorded(data, weight, offset):
        calls.append(weight)
        return real(data, weight, offset)

    monkeypatch.setattr(module, "weight_metrics", recorded)
    decisions = {f"{kind}_budget_0.1": [3] for kind in ["logistic", "random", "agreement"]}
    module.validation_evaluation(bundle, 0.2, config, decisions)
    assert calls == [0.2]
