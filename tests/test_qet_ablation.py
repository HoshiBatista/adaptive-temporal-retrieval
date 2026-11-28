"""QET intervention contracts: no gold inputs, exact bypass, native parity and frozen scope."""

import inspect
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from temporalproof.data.splits import SplitManifest, select_queries
from temporalproof.data.timelyqa import DatasetError
from temporalproof.evaluation.qet_ablation import (
    AblationConfig,
    GoldAccessForbidden,
    canonical_train_ids,
    original_auto_prefix,
    summarize,
)
from temporalproof.retrieval import qet_policy
from temporalproof.retrieval.bm25 import Hit
from temporalproof.retrieval.timelyrag import (
    automatic_rerank,
    automatic_rerank_with_qet,
    load_reference,
)


@pytest.fixture
def config():
    return AblationConfig.model_validate_json(
        Path("configs/experiments/m4_qet_ablation_train.json").read_text()
    )


def inputs(query="2031년 입사한 직원의 휴가는?"):
    return (
        None,
        np.array([0.2, 0.2, 0.3]),
        [10, 11, 12],
        np.array([None, datetime(2020, 1, 1), datetime(2040, 1, 1)], dtype=object),
        np.array([None] * 3, dtype=object),
        query,
        "2035-01-01 00:00:00",
    )


def test_abstention_returns_exact_fast_scores_and_ties_without_fallback(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no temporal kernel allowed on abstention")

    monkeypatch.setattr(qet_policy, "automatic_rerank_with_qet", forbidden)
    fast = [Hit(12, 0.3), Hit(10, 0.2), Hit(11, 0.2)]
    hits, diagnostic = qet_policy.role_aware_rerank(*inputs(), fast)
    assert hits == fast and hits is not fast
    assert not diagnostic["temporal_rerank_called"]
    assert diagnostic["selection"]["selected_time"] is None


def test_selected_target_changes_only_qet_and_keeps_original_query_times_arrays(monkeypatch):
    args = inputs("2031년 입사. 2033년 6월 12일부터 시행된 정책")
    fast = [Hit(12, 0.3), Hit(10, 0.2), Hit(11, 0.2)]
    calls = []

    def capture(*passed, **kwargs):
        calls.append((passed, kwargs))
        return list(reversed(fast)), {"synthetic": True}

    monkeypatch.setattr(qet_policy, "automatic_rerank_with_qet", capture)
    hits, diagnostic = qet_policy.role_aware_rerank(*args, fast)
    assert hits == list(reversed(fast))
    passed, kwargs = calls[0]
    assert all(a is b for a, b in zip(args, passed, strict=True))
    assert kwargs == {"query_event_time": datetime(2033, 6, 12)}
    assert diagnostic["temporal_rerank_called"]


@pytest.mark.parametrize("mutation", ["scores", "duplicate", "missing", "time_length", "nan"])
def test_inference_rejects_corrupted_cached_pool(mutation):
    args = list(inputs())
    fast = [Hit(12, 0.3), Hit(10, 0.2), Hit(11, 0.2)]
    if mutation == "scores":
        fast[0] = Hit(12, 0.4)
    elif mutation == "duplicate":
        fast[-1] = fast[0]
    elif mutation == "missing":
        fast.pop()
    elif mutation == "time_length":
        args[3] = args[3][1:]
    else:
        args[1][0] = float("nan")
    with pytest.raises(ValueError, match="unaligned"):
        qet_policy.role_aware_rerank(*args, fast)


@pytest.mark.parametrize("point", [None, "2033-06-12", datetime(2033, 6, 12, tzinfo=UTC)])
def test_explicit_override_refuses_unknown_or_aware_times(point):
    with pytest.raises(ValueError, match="explicit naive"):
        automatic_rerank_with_qet(*inputs(), query_event_time=point)


def test_rank_api_has_no_gold_answer_or_expensive_outcome_inputs():
    forbidden = {"record", "query_id", "gold", "qrels", "answer", "outcome", "deep_scores"}
    for function in [qet_policy.role_aware_rerank, automatic_rerank_with_qet]:
        assert not forbidden & set(inspect.signature(function).parameters)


@pytest.mark.parametrize("change", ["test", "validation", "oracle", "pool", "source"])
def test_fixed_config_refuses_heldout_or_protocol_mutations(config, change):
    raw = config.model_dump()
    if change in {"test", "validation"}:
        raw["partition"] = change
        raw[f"{change}_evaluation_allowed"] = True
    elif change == "oracle":
        raw["oracle_alpha"] = True
    elif change == "pool":
        raw["top_k"] = 200
    else:
        raw["train_manifest_sha256"] = "not-a-hash"
    with pytest.raises(ValidationError):
        AblationConfig.model_validate(raw)


def test_canonical_selection_rejects_even_same_size_heldout_substitution(config):
    split = SplitManifest(
        dataset_sha256="a" * 64,
        seed=1729,
        audit={},
        groups={"train": [[i] for i in range(500)], "validation": [[500]], "test": [[501]]},
    )
    ids = select_queries(split, "train", 500, config.seed)
    plan = {
        "selected_query_ids": ids,
        "validation_partition_evaluated": False,
        "test_partition_evaluated": False,
    }
    assert canonical_train_ids(split, config, plan) == ids
    for changes in [
        {"selected_query_ids": [500, *ids[1:]]},
        {"test_partition_evaluated": True},
        {"validation_partition_evaluated": True},
    ]:
        with pytest.raises(DatasetError, match="canonical train500"):
            canonical_train_ids(split, config, {**plan, **changes})


def test_original_parity_namespace_is_private_and_never_reads_gold():
    # Independently authored stand-in compiled body with the same metric/oracle boundary.
    code = compile(
        "def evaluate_2stage(name, scorefn, queries, ids, dets, dits):\n"
        "    assert len(scorefn(queries[0]['query'])) == 3\n"
        "    if queries[0]['positive_ids']:\n"
        "        evaluate_ranking_ids(ids[::-1], queries[0]['positive_ids'], 10)\n"
        "    rerank_with_alpha_grid()\n",
        "synthetic_parity",
        "exec",
    )
    namespace = {}
    exec(code, namespace)
    original_globals = dict(namespace)
    reference = SimpleNamespace(original_functions=namespace)
    assert original_auto_prefix(reference, np.ones(3), [1, 2, 3], None, None, "q", None, None) == [
        3,
        2,
        1,
    ]
    assert namespace == original_globals
    with pytest.raises(AssertionError, match="judgments"):
        list(GoldAccessForbidden())


def test_train_gate_requires_positive_group_interval_and_does_not_enable_test(config):
    names = ["fast", "native_fast", "native_auto", "native_auto_same_bypass", "role_qet"]
    metrics = [
        "ndcg_at_10",
        "recall_at_10",
        "recall_at_20",
        "recall_at_100",
        "reference_prefix_ndcg_at_10",
    ]
    rows = {
        name: [{"query_id": q, **{m: 0.2 for m in metrics}} for q in range(4)] for name in names
    }
    diagnostic = [
        {
            "query_id": q,
            "role": {
                "temporal_rerank_called": False,
                "selection": {"reason": "no_target_role_cue"},
            },
        }
        for q in range(4)
    ]
    groups = {q: str(q) for q in range(4)}
    result = summarize(rows, diagnostic, groups, config)
    assert not result["train_gate_passed"]
    for row in rows["role_qet"]:
        row["ndcg_at_10"] = 0.3
    result = summarize(rows, diagnostic, groups, config)
    assert result["train_gate_passed"]
    assert not result["validation_evaluated"] and not result["test_evaluated"]


@pytest.mark.skipif(
    os.environ.get("TEMPORALPROOF_REFERENCE_TESTS") != "1",
    reason="explicit reviewed upstream-source CPU integration",
)
def test_original_kernel_parity_for_native_and_explicit_qet_and_no_shared_mutation():
    cfg = json.loads(Path("configs/experiments/m4_qet_ablation_train.json").read_text())
    reference = load_reference(
        Path("data/raw/timelyqa/5213a0c1743f23fd52a258005c3887aaf7e8cd0f"),
        Path("configs/datasets/timelyrag_reference.json"),
        cfg["reference_manifest_sha256"],
    )
    rng = np.random.default_rng(991)
    ids = list(range(100, 200))
    dits = np.array(
        [None if i % 9 == 0 else datetime(2029 + i % 7, 1, 1) for i in ids], dtype=object
    )
    dets = np.array([None] * 100, dtype=object)
    query = "2030년 입사. 2033년 6월 12일부터 시행된 정책은?"
    qit = "2035-01-01 00:00:00"
    globals_before = dict(reference.original_functions)
    for seed in range(8):
        scores = rng.integers(0, 5, 100).astype(float) if seed % 2 else rng.normal(size=100)
        native, native_info = automatic_rerank(reference, scores, ids, dits, dets, query, qit)
        original_qet, _ = reference.utils.extract_qet_from_query(
            query, qit_hint=reference.utils.to_dt(qit)
        )
        same, info = automatic_rerank_with_qet(
            reference, scores, ids, dits, dets, query, qit, query_event_time=original_qet
        )
        assert same == native
        assert {k: v for k, v in info.items() if k != "query_event_time_overridden"} == native_info
        for point in [original_qet, datetime(2033, 6, 12)]:
            hits, info = automatic_rerank_with_qet(
                reference, scores, ids, dits, dets, query, qit, query_event_time=point
            )
            expected = original_auto_prefix(
                reference, scores, ids, dits, dets, query, reference.utils.to_dt(qit), point
            )
            assert [h.doc_id for h in hits[:10]] == expected
            assert {h.doc_id for h in hits} == set(ids)
            assert info["scoring_granularity"] == native_info["scoring_granularity"]
    assert reference.original_functions == globals_before
