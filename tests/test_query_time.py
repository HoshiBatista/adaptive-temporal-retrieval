"""Independently authored date-role cases: uncertainty, abstention and isolation."""

import inspect
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from temporalproof.data.splits import SplitManifest, select_queries
from temporalproof.data.timelyqa import DatasetError, Document, QueryRecord, parse_timestamp
from temporalproof.evaluation import query_time_audit as audit
from temporalproof.retrieval.bm25 import Hit
from temporalproof.retrieval.query_time import QueryTimeSelection, analyze_query_time


@pytest.fixture
def config():
    return audit.AuditConfig.model_validate_json(
        Path("configs/experiments/m4_query_time_roles_train.json").read_text()
    )


def test_policy_target_does_not_follow_employment_date():
    for hire in ["2028년 2월 1일", "2029년 12월 25일"]:
        result = analyze_query_time(f"{hire} 입사한 직원. 2032년 6월 15일부터 시행된 정책은?")
        assert result.selected_time == datetime(2032, 6, 15)
        assert result.mentions[0].role_cues == ("employment",)
        assert result.mentions[1].role_cues == ("policy",)


def test_asof_priority_over_policy_and_no_coarse_fallthrough():
    query = "2030년 3월 1일부터 시행된 정책. {target} 기준으로 적용되는 규정은?"
    precise = analyze_query_time(query.format(target="2032년 9월 5일"))
    assert precise.selected_time == datetime(2032, 9, 5)
    coarse = analyze_query_time(query.format(target="2032년"))
    assert coarse.reason == "coarse_target_precision"
    assert coarse.selected_time is None
    assert coarse.identified_target_index == 1


@pytest.mark.parametrize(
    ("query", "reason"),
    [
        ("2030년 입사하고 2032년 4월 2일 휴직을 신청했다.", "no_target_role_cue"),
        ("2030년 규정과 2032년 규정을 비교해주세요.", "comparison_request"),
        ("2032년 3월 1일 현재, 2033년 3월 1일 현재", "multiple_target_values"),
        ("2032년 3월 1일 입사 기준", "conflicting_role_cues"),
        ("2032년 3월 1일까지 시행된 정책", "non_point_target_relation"),
        ("2032년 3월 1일 이후 기준", "non_point_target_relation"),
        ("2032년 3월 1일 이전 기준", "non_point_target_relation"),
        ("날짜가 없는 질문", "no_target_role_cue"),
    ],
)
def test_ambiguous_event_comparison_and_range_abstain(query, reason):
    result = analyze_query_time(query)
    assert result.action == "keep_fast_ranking"
    assert result.selected_time is None
    assert result.reason == reason


def test_context_stops_at_next_date_and_punctuation():
    result = analyze_query_time("2030년 2월 1일, 2032년 3월 1일부터 시행된 정책")
    assert result.mentions[0].role_cues == ()
    assert result.selected_time == datetime(2032, 3, 1)
    assert analyze_query_time("2032년 3월 1일. 시행된 정책").selected_time is None


def test_duplicate_identical_targets_are_unambiguous():
    result = analyze_query_time("2032년 3월 1일 현재, 2032년 3월 1일 기준")
    assert result.selected_time == datetime(2032, 3, 1)


@pytest.mark.parametrize(
    ("date", "precision", "lower", "upper"),
    [
        ("2032년", "year", datetime(2032, 1, 1), datetime(2033, 1, 1)),
        ("2032년 2월", "month", datetime(2032, 2, 1), datetime(2032, 3, 1)),
        ("2032-12", "month", datetime(2032, 12, 1), datetime(2033, 1, 1)),
        ("2032년 2월 29일", "day", datetime(2032, 2, 29), datetime(2032, 3, 1)),
        ("2032년 2월 29일 23시", "hour", datetime(2032, 2, 29, 23), datetime(2032, 3, 1)),
        (
            "2032-02-29T23:58:59",
            "second",
            datetime(2032, 2, 29, 23, 58, 59),
            datetime(2032, 2, 29, 23, 59),
        ),
    ],
)
def test_calendar_resolution_is_preserved(date, precision, lower, upper):
    result = analyze_query_time(f"{date} 현재")
    assert len(result.mentions) == 1
    mention = result.mentions[0]
    assert mention.precision == precision
    assert (mention.lower, mention.upper_exclusive) == (lower, upper)
    assert (result.selected_time is None) == (precision in {"year", "month"})


@pytest.mark.parametrize("date", ["2031년 2월 29일", "2032-13-05", "2032-02-29T25:15"])
def test_invalid_long_date_cannot_shrink_to_valid_partial(date):
    result = analyze_query_time(f"{date} 현재")
    assert len(result.mentions) == 1
    assert result.mentions[0].status == "invalid_calendar"
    assert result.mentions[0].lower is None
    assert result.reason == "unknown_or_invalid_target"


@pytest.mark.parametrize("date", ["3월 7일", "3월"])
def test_yearless_does_not_invent_a_year(date):
    result = analyze_query_time(f"{date} 기준")
    assert result.mentions[0].status == "missing_year"
    assert result.selected_time is None
    assert "insert" not in str(inspect.signature(analyze_query_time))


@pytest.mark.parametrize("suffix", ["Z", "+03:00", "-0500"])
@pytest.mark.parametrize("separator", [" ", ""])
def test_timezone_not_silently_discarded(suffix, separator):
    result = analyze_query_time(f"2032-02-29T12:15{suffix}{separator}현재")
    assert result.mentions[0].status == "unsupported_timezone"
    assert result.selected_time is None


def test_nfkc_offsets_and_serialization_contract():
    result = analyze_query_time("２０３２년 ２월 ２９일 현재")
    assert result.selected_time == datetime(2032, 2, 29)
    assert QueryTimeSelection.model_validate_json(result.model_dump_json()) == result
    for changes in [
        {"action": "keep_fast_ranking"},
        {"selected_index": 99},
        {"selected_time": "2040-01-01T00:00:00"},
        {"gold_docs_final": [1]},
    ]:
        with pytest.raises(ValidationError):
            QueryTimeSelection.model_validate({**result.model_dump(), **changes})
    with pytest.raises(ValidationError):
        QueryTimeSelection.model_validate(
            {
                **result.model_dump(),
                "mentions": [{**result.mentions[0].model_dump(), "start": 1}],
            }
        )


@pytest.mark.parametrize("mutation", ["validation", "test", "outcome"])
def test_audit_config_cannot_enable_heldout_or_labels(config, mutation):
    raw = config.model_dump()
    if mutation in {"validation", "test"}:
        raw["partition"] = mutation
        raw[f"{mutation}_evaluation_allowed"] = True
    else:
        raw["router_gain"] = 1.0
    with pytest.raises(ValidationError):
        audit.AuditConfig.model_validate(raw)


def toy_split():
    return SplitManifest(
        dataset_sha256="a" * 64,
        seed=1729,
        groups={
            "train": [[q] for q in range(25)] + [[25, 26]],
            "validation": [[27]],
            "test": [[28]],
        },
        audit={},
    )


def toy_records():
    return [
        QueryRecord(
            index=q,
            category="synthetic",
            query_ko="2032년 3월 1일 현재의 규정은?",
            answer="synthetic",
            query_insert_time="2031-01-01 00:00:00",
            docs_ko=[
                Document(
                    global_doc_id=q * 2 + i,
                    text=f"synthetic-{q}-{i}",
                    doc_insert_time="invalid" if i else "2033-01-01 00:00:00",
                )
                for i in range(2)
            ],
            gold_docs_final=[0],
            gold_docs_final_global=[q * 2],
        )
        for q in range(29)
    ]


def fake_reference():
    def extract(text, qit_hint):
        # Deliberately independent fixture, with observable calls for held-out isolation.
        assert "2032" in text
        return datetime(2032, 3, 1), "day"

    return SimpleNamespace(
        utils=SimpleNamespace(
            to_dt=parse_timestamp,
            extract_qet_from_query=extract,
        )
    )


def test_group_sample_deterministic_and_training_only(config):
    split = toy_split()
    cfg = config.model_copy(update={"query_count": 27, "sample_count": 20})
    selected = audit.sample_ids(split, cfg)
    reversed_split = split.model_copy(
        update={
            "groups": {
                **split.groups,
                "train": [list(reversed(g)) for g in reversed(split.groups["train"])],
            },
        }
    )
    assert selected == audit.sample_ids(reversed_split, cfg)
    assert len(selected) == len(set(selected)) == 20
    assert set(selected) <= set(split.query_ids("train"))
    assert 26 not in selected


def test_analysis_excludes_heldout_and_is_label_answer_invariant(config):
    split, records = toy_split(), toy_records()
    cfg = config.model_copy(update={"query_count": 27})
    # Held-out text would fail the fake extractor if visited.
    for record in records[-2:]:
        record.query_ko = "held-out must not be parsed"
    expected = audit.analyze_train(records, split, cfg, fake_reference())
    for record in records:
        record.answer = "different answer"
        record.gold_docs_final = [1]
        record.gold_docs_final_global = [record.docs_ko[1].global_doc_id]
    assert audit.analyze_train(records, split, cfg, fake_reference()) == expected
    assert [r["query_id"] for r in expected] == split.query_ids("train")
    assert audit.summarize(expected, split)["selected_after_query_insert_count"] == 27
    with pytest.raises(DatasetError, match="complete"):
        audit.analyze_train(records, split, config, fake_reference())


def test_cached_pool_unknown_and_future_times_are_counted_not_filtered(
    config, tmp_path, monkeypatch
):
    split, records = toy_split(), toy_records()
    cfg = config.model_copy(update={"query_count": 27, "train500_query_count": 1})
    ids = select_queries(split, "train", 1, cfg.seed)
    q = ids[0]
    plan = {
        "selected_query_ids": ids,
        "validation_partition_evaluated": False,
        "test_partition_evaluated": False,
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(audit, "verify_source", lambda *args: {"split_sha256": "a" * 64})
    monkeypatch.setattr(
        audit,
        "load_trec",
        lambda *args: {
            q: [Hit(doc_id=q * 2, score=1), Hit(doc_id=q * 2 + 1, score=0)],
        },
    )
    rows = audit.analyze_train(records, split, cfg, fake_reference())
    pools, summary = audit.audit_cached_pools(tmp_path, records, split, rows, cfg)
    assert summary["candidate_count"] == 2
    assert summary["insertion_unknown_count"] == 1
    assert summary["later_or_equal_insert_count"] == 1
    assert pools[0]["candidate_count"] == 2
    for changes in [{"selected_query_ids": [27]}, {"test_partition_evaluated": True}]:
        (tmp_path / "plan.json").write_text(json.dumps({**plan, **changes}))
        with pytest.raises(DatasetError, match="train-only"):
            audit.audit_cached_pools(tmp_path, records, split, rows, cfg)
