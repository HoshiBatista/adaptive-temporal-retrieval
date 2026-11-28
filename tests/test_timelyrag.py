"""Native ranking/schema tests; upstream parity is explicit opt-in and CPU-only."""

import inspect
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from pydantic import ValidationError

from temporalproof.data.timelyqa import DatasetError
from temporalproof.evaluation.baseline import BaselineConfig
from temporalproof.evaluation.timelyrag import ReferenceConfig, hits_from_scores
from temporalproof.retrieval.timelyrag import (
    automatic_rerank,
    load_reference,
    native_top_indices,
    verify_reference,
)


@pytest.fixture
def config():
    return ReferenceConfig.model_validate_json(
        Path("configs/experiments/m4_timelyrag_reference_company_policies.json").read_text()
    )


def test_native_ties_preserve_upstream_reverse_argsort_without_id_reordering():
    scores = np.array([1.0, 1.0, 2.0, 1.0])
    expected = np.argsort(scores)[::-1]
    assert native_top_indices(scores, 4).tolist() == expected.tolist()
    assert [h.doc_id for h in hits_from_scores(scores, [10, 11, 12, 13], 4)] == [
        [10, 11, 12, 13][i] for i in expected
    ]


@pytest.mark.parametrize(
    "scores", [np.array([]), np.array([float("nan")]), np.array([float("inf")]), np.ones((2, 2))]
)
def test_invalid_native_score_vectors_rejected(scores):
    with pytest.raises(ValueError):
        native_top_indices(scores, 1)


def test_native_config_roundtrip_and_old_configs_stay_valid(config):
    assert BaselineConfig.model_validate_json(config.baseline.model_dump_json()) == config.baseline
    old = BaselineConfig.model_validate_json(
        Path("configs/experiments/bm25_company_policies.json").read_text()
    )
    assert old.tie_break == "score_desc_doc_id_asc"
    raw = config.model_dump()
    raw["oracle_evaluation"] = True
    with pytest.raises(ValidationError):
        ReferenceConfig.model_validate(raw)


@pytest.mark.parametrize("mutation", ["train", "test", "tie", "pool", "methods"])
def test_reference_protocol_is_fixed(config, mutation):
    raw = config.model_dump()
    if mutation in ["train", "test"]:
        raw["baseline"]["partition"] = mutation
    elif mutation == "tie":
        raw["baseline"]["tie_break"] = "score_desc_doc_id_asc"
    elif mutation == "pool":
        raw["baseline"]["top_k"] = 200
    else:
        raw["methods"] = ["dense"]
    with pytest.raises(ValidationError):
        ReferenceConfig.model_validate(raw)


def test_inference_api_has_no_gold_or_queryrecord_argument():
    params = set(inspect.signature(automatic_rerank).parameters)
    assert params == {
        "reference",
        "scores",
        "document_ids",
        "document_insert_times",
        "document_event_times",
        "query",
        "query_insert_time",
    }


def test_reference_hash_check_precedes_any_execution(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(DatasetError, match="checksum"):
        verify_reference(tmp_path, manifest, "f" * 64)


requires_reference = pytest.mark.skipif(
    os.environ.get("TEMPORALPROOF_REFERENCE_TESTS") != "1",
    reason="explicit reviewed upstream source integration; set TEMPORALPROOF_REFERENCE_TESTS=1",
)


@pytest.fixture
def upstream(config):
    return load_reference(
        Path("data/raw/timelyqa/5213a0c1743f23fd52a258005c3887aaf7e8cd0f"),
        Path("configs/datasets/timelyrag_reference.json"),
        config.reference_manifest_sha256,
    )


@requires_reference
def test_original_auto_matches_adapter_for_36_synthetic_cases(upstream):
    """The upstream oracle executes only here on synthetic labels, never benchmark validation."""
    rng = np.random.default_rng(1729)
    ids = list(range(1000, 1160))
    original = upstream.original_functions
    actual_evaluator = original["evaluate_ranking_ids"]
    cases = []
    for case in range(36):
        scores = (
            rng.integers(0, 5, size=160).astype(np.float32)
            if case % 3 == 0
            else rng.normal(size=160).astype(np.float64)
        )
        if case == 0:
            scores[:] = 0
        query = "untyped synthetic" if case % 4 == 0 else "2020년 입사 2025년 6월 15일 정책"
        qit = None if case % 5 == 0 else datetime(2024, 1, 1)
        qit_text = qit.strftime("%Y-%m-%d %H:%M:%S") if qit else None
        dits = np.array(
            [
                None if case == 0 or i % 9 == 0 else datetime(2022, 1, 1) + timedelta(days=i * 9)
                for i in range(160)
            ],
            dtype=object,
        )
        dets = np.array([None if i % 5 else datetime(2025, 1, 1) for i in range(160)], dtype=object)
        if case == 0:
            dets[:] = None
        qet, _ = upstream.utils.extract_qet_from_query(query, qit_hint=qit)
        hits, info = automatic_rerank(upstream, scores, ids, dits, dets, query, qit_text)
        captured = []

        def capture(topk, pos, k, captured_rows=captured):
            captured_rows.append(list(topk))
            return actual_evaluator(topk, pos, k)

        original["evaluate_ranking_ids"] = capture
        try:
            a, _, hist, _ = original["evaluate_2stage"](
                "synthetic",
                lambda _, vector=scores: vector,
                [{"query": query, "positive_ids": [1001, 1035], "qit": qit, "qet": qet}],
                ids,
                dets,
                dits,
            )
        finally:
            original["evaluate_ranking_ids"] = actual_evaluator
        assert [h.doc_id for h in hits[:10]] == captured[0]
        assert hist[info["alpha"]] == 1
        values = actual_evaluator([h.doc_id for h in hits[:10]], [1001, 1035], 10)
        assert [a[k][0] for k in ["hit", "recall", "precision", "ndcg", "map"]] == list(values)
        assert len(hits) == 100
        assert {h.doc_id for h in hits} == {ids[i] for i in native_top_indices(scores, 100)}
        assert info["hard_filter_applied"] is False
        assert info["gold_used_for_alpha"] is False
        cases.append(info)
    assert len(cases) == 36


@requires_reference
def test_loading_source_restores_module_namespace_and_avoids_config_side_effect(
    config, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    sentinel = ModuleType("config")
    monkeypatch.setitem(sys.modules, "config", sentinel)
    repo = Path(__file__).resolve().parents[1]
    loaded = load_reference(
        repo / "data/raw/timelyqa/5213a0c1743f23fd52a258005c3887aaf7e8cd0f",
        repo / "configs/datasets/timelyrag_reference.json",
        config.reference_manifest_sha256,
    )
    assert sys.modules["config"] is sentinel
    assert not (tmp_path / "test").exists()
    assert loaded.config.STAGE1_TOPN == 100


@requires_reference
def test_unknown_and_later_insertions_remain_neutral_not_deleted(upstream):
    qit = datetime(2024, 1, 1)
    dits = np.array([None, qit, qit + timedelta(days=1)], dtype=object)
    dets = np.array([None, None, None], dtype=object)
    delta = upstream.scores.compute_single_delta(qit, None, dits, dets, np.array([0, 1, 2]), "day")
    assert np.isnan(delta).all()
    assert upstream.scores.time_affinity_from_delta(delta).tolist() == [0.5, 0.5, 0.5]
    assert upstream.metrics.evaluate_ranking_ids([1, 9], [1, 2], 2)[3] == 1.0


@requires_reference
def test_cpu_pair_collection_is_label_independent_and_never_scores_test(tmp_path, upstream, config):
    from temporalproof.data.source import ARCHIVE_SHA256
    from temporalproof.data.splits import SplitManifest
    from temporalproof.data.timelyqa import Document, QueryRecord
    from temporalproof.evaluation.timelyrag import collect_pair, comparison

    records = [
        QueryRecord(
            index=q,
            category="synthetic",
            query_ko=("TEST" if q >= 3 else f"synthetic {q}"),
            answer="synthetic",
            docs_ko=[
                Document(global_doc_id=q * 10 + i, text=f"passage {q}:{i}") for i in range(10)
            ],
            gold_docs_final=[0],
            gold_docs_final_global=[q * 10],
        )
        for q in range(12)
    ]
    split = SplitManifest(
        dataset_sha256=ARCHIVE_SHA256,
        seed=1729,
        groups={"train": [[0]], "validation": [[1, 2]], "test": [[q] for q in range(3, 12)]},
        audit={},
    )
    cfg = config.model_copy(
        update={
            "baseline": config.baseline.model_copy(
                update={"query_limit": 2, "warmup_queries": 1, "bootstrap_samples": 100}
            )
        }
    )
    ids = [d.global_doc_id for r in records for d in r.docs_ko]
    dits = np.array([None for _ in ids], dtype=object)

    def score(text):
        assert text != "TEST"
        return np.arange(len(ids), dtype=np.float64)

    roots = []
    for attempt in range(2):
        root = tmp_path / str(attempt)
        root.mkdir()
        roots.append(root)
        if attempt:
            records = [
                r.model_copy(
                    update={"gold_docs_final": [9], "gold_docs_final_global": [r.index * 10 + 9]}
                )
                for r in records
            ]
        collect_pair(
            records,
            split,
            cfg,
            "bm25",
            root,
            {
                "archive_sha256": ARCHIVE_SHA256,
                "split_sha256": "b" * 64,
                "source_snapshot_sha256": "c" * 64,
            },
            score,
            upstream,
            ids,
            dits,
            dits,
        )
        result = comparison(root, "bm25", {1: "one", 2: "one"}, cfg)
        assert result["base_metrics"]["recall_at_100"] == result["auto_metrics"]["recall_at_100"]
        assert json.loads(json.dumps(result)) == result
    for name in [
        "bm25/run.trec",
        "bm25_timelyrag_auto/run.trec",
        "bm25_temporal_diagnostics.jsonl",
    ]:
        assert (roots[0] / name).read_bytes() == (roots[1] / name).read_bytes()


def test_default_bm25_refuses_a_reference_tie_label_without_native_callback(tmp_path, config):
    from temporalproof.data.splits import SplitManifest
    from temporalproof.data.timelyqa import load_records
    from temporalproof.evaluation.baseline import run_baseline

    records = load_records(Path("tests/fixtures/timelyqa_synthetic.jsonl"))
    split = SplitManifest(
        dataset_sha256="a" * 64,
        seed=1729,
        groups={"train": [], "validation": [[0, 1]], "test": []},
        audit={},
    )
    cfg = config.baseline.model_copy(update={"method": "bm25", "query_limit": 2})
    with pytest.raises(ValueError, match="explicit retrieval callback"):
        run_baseline(records, split, cfg, tmp_path / "run", {})
    assert not (tmp_path / "run").exists()
