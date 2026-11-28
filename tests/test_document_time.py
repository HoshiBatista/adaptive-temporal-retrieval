from datetime import datetime

import pytest

from temporalproof.data.timelyqa import Document
from temporalproof.evaluation.document_time_audit import sample_passages, train_passages
from temporalproof.retrieval.document_time import analyze_document_time


@pytest.mark.parametrize(
    "text",
    [
        "본 정책은 2036년 4월 15일부터 시행합니다.",
        "2036년 4월 15일자로 발효되는 회사의 보안 지침입니다.",
        "시행일: 2036-04-15",
        "제1장 규정\n2036년 4월 15일부터 시행",
        "본 정책은 2036년 4월 15일부터 적용됩니다. 2035년 1월 1일 이후 입사한 직원입니다.",
    ],
)
def test_explicit_start_with_distinct_cue(text):
    result = analyze_document_time(text)
    assert result.start == datetime(2036, 4, 15)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("2036년 4월 15일 이후 입사한 직원의 정책", "other_dates_only"),
        ("본 규정은 2036년 4월부터 시행한다.", "non_day_start"),
        ("본 규정은 4월 15일부터 시행한다.", "unknown_start"),
        ("본 규정은 2036년 2월 30일부터 시행한다.", "unknown_start"),
        ("본 규정은 2036년 4월 15일부터 시행할 예정이다.", "other_dates_only"),
        ("개정일: 2036년 4월 15일", "other_dates_only"),
        ("본 규정은 2036년 4월 15일부터 시행되나요?", "other_dates_only"),
        ("Q: 2036년 4월 15일부터 시행된 정책인가요?", "range_or_qualified"),
        ("2036년 4월 15일부터 시행되는 정책을 검토한다.", "range_or_qualified"),
        ("시행일: 2036-04-15 ~ 2036-05-15", "range_or_qualified"),
        ("시행일: 2036-04-15–2036-05-15", "range_or_qualified"),
        ("2036년 4월 15일부터 시행되는 정책은 없다.", "range_or_qualified"),
        (
            "본 규정은 2036년 4월 15일부터 시행한다. 별도 정책은 2037년 4월 15일부터 적용한다.",
            "multiple_start_dates",
        ),
        ("등록일: 2036-04-15", "other_dates_only"),
    ],
)
def test_unresolved_start_never_becomes_effective_day(text, reason):
    result = analyze_document_time(text)
    assert result.start is None
    assert result.reason == reason
    assert result.dates


def test_sampling_drops_shared_passages_and_picks_lowest_train_id():
    from types import SimpleNamespace

    records = [
        SimpleNamespace(
            index=i,
            docs_ko=[Document(global_doc_id=j, text=text) for j, text in docs],
        )
        for i, docs in [
            (1, [(22, "local rule"), (39, "Shared rule")]),
            (2, [(9, "local  rule"), (15, "Other text")]),
            (4, [(25, "shared  rule"), (11, "held out only")]),
        ]
    ]
    docs = train_passages(records, {1, 2})
    assert [d.global_doc_id for d in docs] == [9, 15]
    assert sample_passages(docs) == sample_passages(docs[::-1])
