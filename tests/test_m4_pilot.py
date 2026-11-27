"""CPU-only train pilot guards and end-to-end fake search/scoring."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from temporalproof.data.source import ARCHIVE_SHA256, sha256_file
from temporalproof.data.splits import SplitManifest
from temporalproof.data.timelyqa import DatasetError, Document, QueryRecord
from temporalproof.evaluation.pilot import PilotConfig, collect_fast, read_rows
from temporalproof.evaluation.reranking import RerankingConfig, load_candidate_pool, run_reranking
from temporalproof.retrieval.bm25 import Hit
from temporalproof.retrieval.reranker import KOREAN_MODEL_ID, KOREAN_REVISION
from temporalproof.routing.features import PreDeepFeatures, extract_features
from temporalproof.routing.targets import build_train_outcomes, diagnostic_policies, training_groups


@pytest.fixture
def feature():
    docs = {
        0: Document(global_doc_id=0, text=" same  TEXT ", doc_insert_time="2023-01-01 00:00:00"),
        1: Document(global_doc_id=1, text="same text", doc_insert_time="2025-01-01 00:00:00"),
        2: Document(global_doc_id=2, text="different", doc_insert_time="unknown"),
    }
    return extract_features(
        "2022년 이전 2022년 2024-01-01",
        "2024-01-01 00:00:00",
        [Hit(0, 3), Hit(1, 2)],
        [Hit(0, 0.9), Hit(2, 0.7)],
        [Hit(0, 0.03), Hit(1, 0.02), Hit(2, 0.01)],
        docs,
    )


@pytest.fixture
def split():
    return SplitManifest(
        dataset_sha256=ARCHIVE_SHA256,
        seed=1729,
        groups={"train": [[0, 1]], "validation": [[2]], "test": [[3]]},
        audit={},
    )


def config():
    return PilotConfig.model_validate_json(
        Path("configs/experiments/m4_train_pilot_company_policies.json").read_text()
    )


def metric(q, score, cost):
    return {"query_id": q, "ndcg_at_10": score, "retrieval_ms": cost, "recall_at_100": 1.0}


def test_features_use_fast_signals_and_preserve_unknown_dates(feature):
    assert feature.explicit_year_count == 2
    assert feature.top10_overlap_fraction == 0.5
    assert feature.top1_agreement is True
    assert feature.bm25_relative_margin == pytest.approx(1 / 3)
    assert feature.candidate_insertion_known_fraction == pytest.approx(2 / 3)
    assert feature.candidate_after_query_fraction == 0.5
    assert feature.candidate_distinct_insertion_days == 2
    assert feature.candidate_unique_text_fraction == pytest.approx(2 / 3)
    docs = {i: Document(global_doc_id=i, text="independent", doc_insert_time=None) for i in [0, 1]}
    hits = [Hit(0, 0), Hit(1, 0)]
    unknown = extract_features("query", None, hits, hits, hits, docs)
    assert unknown.query_insertion_known is False
    assert unknown.candidate_after_query_fraction is None
    assert unknown.median_abs_insertion_gap_days is None
    assert unknown.hybrid_normalized_entropy is None


@pytest.mark.parametrize(
    "forbidden", ["gain", "gold_grade", "deep_score", "query_id", "doc_id", "incremental_rerank_ms"]
)
def test_inference_schema_rejects_outcomes_and_id_features(feature, forbidden):
    with pytest.raises(ValidationError, match="Extra inputs"):
        PreDeepFeatures.model_validate({**feature.model_dump(), forbidden: 1})


@pytest.mark.parametrize("partition", ["validation", "test"])
def test_pilot_config_rejects_held_out_partition(partition):
    value = config().model_dump()
    value["baseline"]["partition"] = partition
    with pytest.raises(ValidationError):
        PilotConfig.model_validate(value)


def test_outcomes_are_joined_by_query_and_cost_is_already_incremental(split, feature):
    rows = build_train_outcomes(
        split,
        {0: feature, 1: feature},
        [metric(0, 0.1, 80), metric(1, 0.8, 90)],
        [metric(1, 0.2, 1200), metric(0, 0.5, 1000)],
        [0.0, 0.1, 0.5],
    )
    assert rows[0].query_id == 0 and rows[0].gain == pytest.approx(0.4)
    assert rows[0].incremental_rerank_ms == 1000  # never 1000 - 80
    assert rows[0].cost_sensitive_positive == {"0.0": True, "0.1": True, "0.5": False}
    assert rows[1].cost_sensitive_positive == {"0.0": False, "0.1": False, "0.5": False}
    assert rows[0].group_id == rows[1].group_id == training_groups(split)[0]


@pytest.mark.parametrize("q", [2, 3])
def test_no_validation_or_test_outcomes(split, feature, q):
    with pytest.raises(DatasetError, match="training queries only"):
        build_train_outcomes(split, {q: feature}, [metric(q, 0.2, 80)], [metric(q, 0.8, 1000)], [0])


def test_duplicate_missing_or_changed_coverage_fail(split, feature):
    for fast, deep in [
        ([metric(0, 0.2, 80)] * 2, [metric(0, 0.8, 1000)]),
        ([metric(0, 0.2, 80)], [metric(1, 0.8, 1000)]),
        ([metric(0, 0.2, 80)], [{**metric(0, 0.8, 1000), "recall_at_100": 0.5}]),
    ]:
        with pytest.raises(DatasetError):
            build_train_outcomes(split, {0: feature}, fast, deep, [0])


def test_rules_random_and_oracle_have_distinct_information_boundaries(split, feature):
    features = {0: feature.model_copy(update={"top10_overlap_fraction": 0.1}), 1: feature}
    rows = build_train_outcomes(
        split,
        features,
        [metric(0, 0.2, 80), metric(1, 0.8, 80)],
        [metric(0, 0.8, 1000), metric(1, 0.2, 1000)],
        [0],
    )
    policies = diagnostic_policies(
        rows, features, budgets=[0.5, 1], disagreement_threshold=0.2, seed=1729
    )
    by_name = {r["name"]: r for r in policies}
    assert by_name["low_agreement"]["routed_query_ids"] == [0]
    assert by_name["low_agreement"]["ndcg_at_10"] == pytest.approx(0.8)
    assert by_name["low_agreement"]["modeled_mean_additive_stage_ms"] == 580
    assert by_name["oracle_query_budget_1"]["routed_query_ids"] == [0]
    assert by_name["oracle_query_budget_1"]["uses_outcomes_at_decision"] is True
    changed = [r.model_copy(update={"gain": -r.gain}) for r in rows]
    repeated = diagnostic_policies(
        changed, features, budgets=[0.5, 1], disagreement_threshold=0.2, seed=1729
    )
    for a, b in zip(policies, repeated, strict=True):
        if not a["uses_outcomes_at_decision"]:
            assert a["routed_query_ids"] == b["routed_query_ids"]


def test_fake_end_to_end_predeep_files_frozen_before_scoring(tmp_path, split):
    records = [
        QueryRecord(
            index=q,
            category="synthetic",
            query_ko=f"2022년 synthetic {q}",
            answer="synthetic",
            docs_ko=[
                Document(global_doc_id=q * 100 + i, text=f"independent {q}:{i}") for i in range(100)
            ],
            gold_docs_final=[0],
            gold_docs_final_global=[q * 100],
        )
        for q in range(4)
    ]
    value = config().model_dump()
    value["baseline"].update(query_limit=2, bootstrap_samples=100, warmup_queries=1)
    pilot = PilotConfig.model_validate(value)
    calls = []

    def search(text, cutoff):
        q = int(text.split()[-1])
        assert q in [0, 1]
        calls.append(q)
        hits = [Hit(q * 100 + i, float(100 - i)) for i in range(cutoff)]
        return hits, hits

    features = collect_fast(
        records,
        split,
        pilot,
        tmp_path,
        {
            "archive_sha256": ARCHIVE_SHA256,
            "split_sha256": "b" * 64,
            "source_snapshot_sha256": "c" * 64,
        },
        search,
    )
    digest = sha256_file(tmp_path / "features.jsonl")
    cfg = RerankingConfig(
        baseline=pilot.baseline.model_copy(update={"method": "fixed_cross_encoder"}),
        model_id=KOREAN_MODEL_ID,
        revision=KOREAN_REVISION,
        batch_size=16,
        max_length=512,
        candidate_manifest_sha256=sha256_file(tmp_path / "fast/manifest.json"),
    )
    pool = load_candidate_pool(tmp_path / "fast", records, split, cfg, "b" * 64)

    class Scorer:
        def score(self, query, passages):
            assert sha256_file(tmp_path / "features.jsonl") == digest
            return list(range(len(passages)))

        def profile(self):
            return {"synthetic": True}

    run_reranking(records, split, cfg, pool, Scorer(), tmp_path / "deep", {})
    outcomes = build_train_outcomes(
        split,
        features,
        read_rows(tmp_path / "fast/per_query.jsonl"),
        read_rows(tmp_path / "deep/run/per_query.jsonl"),
        pilot.cost_penalties_ndcg_per_second,
    )
    assert {r.query_id for r in outcomes} == {0, 1}
    assert all(r.gain < 0 for r in outcomes)
    assert len(calls) == 3
    assert sha256_file(tmp_path / "features.jsonl") == digest
    assert (tmp_path / "fast/qrels.tsv").read_bytes() == (
        tmp_path / "deep/run/qrels.tsv"
    ).read_bytes()


def test_tracking_logs_utility_evidence_without_raw_failure_text(tmp_path, monkeypatch):
    from temporalproof.evaluation.pilot import log_pilot
    from temporalproof.tracking import local

    uploaded = []

    class Client:
        def set_tag(self, *args):
            pass

        def log_metric(self, *args):
            pass

        def log_artifact(self, run, path, artifact_path):
            assert run == "deep" and artifact_path == "pilot"
            assert Path(path).is_file()
            uploaded.append(Path(path).name)

    for name in [
        "config.json",
        "plan.json",
        "pre_deep_checkpoint.json",
        "features.jsonl",
        "outcomes.jsonl",
    ]:
        (tmp_path / name).write_text("{}")
    (tmp_path / "failure_cases.json").write_text("untrusted raw data stays local")
    monkeypatch.setattr(local, "local_client", lambda: Client())
    monkeypatch.setattr(
        local, "log_run_directory", lambda path: "fast" if path.name == "fast" else "deep"
    )
    summary = {
        "utility": {
            "mean_gain": 0.1,
            "beneficial_count": 1,
            "harmful_count": 0,
            "unchanged_count": 0,
        }
    }
    log_pilot(tmp_path, summary)
    assert summary["mlflow_run_ids"] == {"fast": "fast", "deep": "deep"}
    assert set(uploaded) == {
        "config.json",
        "plan.json",
        "pre_deep_checkpoint.json",
        "features.jsonl",
        "outcomes.jsonl",
        "summary.json",
    }
