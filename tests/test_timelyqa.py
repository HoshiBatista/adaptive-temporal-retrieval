import io
import json
from datetime import datetime
from pathlib import Path

import pytest

from temporalproof.data.timelyqa import DatasetError, iter_records, load_records, parse_timestamp

FIXTURE = Path(__file__).parent / "fixtures" / "timelyqa_synthetic.jsonl"


def raw_record():
    return json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])


def parse(record):
    return list(iter_records(io.StringIO(json.dumps(record))))[0]


def test_observed_schema_preserves_ids_dates_and_optional_title():
    record = load_records(FIXTURE)[0]
    assert record.index == 0
    assert record.docs_ko[0].global_doc_id == 10
    assert record.query_insert_time == "2024-06-01 10:00:00"
    assert parse_timestamp(record.query_insert_time) == datetime(2024, 6, 1, 10)
    assert parse_timestamp(record.query_insert_time).tzinfo is None
    assert record.docs_ko[1].title == "Synthetic title"
    assert record.gold_docs_final == [0, 1]


@pytest.mark.parametrize("field", ["query_insert_time", "doc_insert_time"])
@pytest.mark.parametrize("value", [None, "", "absent"])
def test_missing_timestamps_stay_unknown(field, value):
    raw = raw_record()
    target = raw if field == "query_insert_time" else raw["docs_ko"][0]
    if value == "absent":
        target.pop(field)
    else:
        target[field] = value
    record = parse(raw)
    model = record if field == "query_insert_time" else record.docs_ko[0]
    assert parse_timestamp(getattr(model, field)) is None
    assert (field in model.model_fields_set) == (value != "absent")


@pytest.mark.parametrize(
    "value",
    [1700000000, True, ["2024-01-01 00:00:00"]],
)
@pytest.mark.parametrize("field", ["query_insert_time", "doc_insert_time"])
def test_non_string_timestamps_fail_without_printing_input(value, field):
    raw = raw_record()
    target = raw if field == "query_insert_time" else raw["docs_ko"][0]
    target[field] = value
    with pytest.raises(DatasetError, match=field) as exc:
        parse(raw)
    assert str(value) not in str(exc.value)


@pytest.mark.parametrize(
    "value",
    [
        "2024-02-30 00:00:00",
        "2024-6-01 00:00:00",
        "2024-01-01T00:00:00Z",
        "2020년 1월 1일 00:00:00",
    ],
)
def test_malformed_timestamps_are_preserved_and_reported_unknown(value):
    raw = raw_record()
    raw["docs_ko"][0]["doc_insert_time"] = value
    record = parse(raw)
    assert record.docs_ko[0].doc_insert_time == value
    assert parse_timestamp(value) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gold_docs_final", []),
        ("gold_docs_final_global", []),
        ("gold_docs_final", [0, 0]),
        ("gold_docs_final_global", [10, 10]),
        ("gold_docs_final", [-1]),
        ("gold_docs_final", [2]),
        ("gold_docs_final", ["0"]),
        ("gold_docs_final", [True]),
        ("gold_docs_final_global", [999]),
    ],
)
def test_invalid_or_inconsistent_labels_fail(field, value):
    raw = raw_record()
    raw[field] = value
    with pytest.raises(DatasetError):
        parse(raw)


def test_unobserved_fields_fail_instead_of_silently_changing_schema():
    raw = raw_record()
    raw["query_event_time"] = "2023-01-01 00:00:00"
    with pytest.raises(DatasetError, match="unknown_field"):
        parse(raw)


def test_invalid_json_reports_line_without_source_content():
    with pytest.raises(DatasetError, match="line 2") as exc:
        list(iter_records(io.StringIO('\n{"private-source-content":')))
    assert "private-source-content" not in str(exc.value)


def test_duplicate_document_id_in_record_fails():
    raw = raw_record()
    raw["docs_ko"][1]["global_doc_id"] = 10
    with pytest.raises(DatasetError):
        parse(raw)


def test_empty_answer_does_not_discard_retrieval_judgments():
    raw = raw_record()
    raw["answer"] = ""
    record = parse(raw)
    assert record.answer == ""
    assert record.gold_docs_final_global == [10, 11]
