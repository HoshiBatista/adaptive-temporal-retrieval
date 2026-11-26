import json
from pathlib import Path

import pytest

from temporalproof.data.audit import audit_records, main, query_group_id
from temporalproof.data.source import download_archive
from temporalproof.data.timelyqa import DatasetError, load_records

FIXTURE = Path(__file__).parent / "fixtures" / "timelyqa_synthetic.jsonl"


def test_audit_detects_duplicates_and_keeps_temporal_context_unknown():
    report = audit_records(load_records(FIXTURE), sample_size=1)
    assert report["record_count"] == 2
    assert report["document_count"] == 3
    assert report["query_texts"]["extra_occurrences"] == 1
    assert report["document_texts"]["extra_occurrences"] == 1
    assert report["query_grouping"]["group_count"] == 1
    assert report["query_grouping"]["repeated_groups_with_different_query_insert_time"] == 1
    assert report["doc_insert_time"]["unknown"] == 1
    assert report["sample"]["record_count"] == 1
    assert report["sample"]["queries_containing_hangul"] == 1
    assert "Synthetic earlier policy" not in json.dumps(report)


def test_grouping_normalizes_formatting_but_preserves_query_dates():
    assert query_group_id("Ａ  2023\nPOLICY") == query_group_id("a 2023 policy")
    assert query_group_id("2023 policy") != query_group_id("2024 policy")


@pytest.mark.parametrize("field", ["index", "global_doc_id"])
def test_cross_record_id_collisions_fail(field):
    records = load_records(FIXTURE)
    if field == "index":
        records[1].index = records[0].index
    else:
        records[1].docs_ko[0].global_doc_id = records[0].docs_ko[0].global_doc_id
    with pytest.raises(DatasetError, match="duplicate"):
        audit_records(records)


def test_empty_dataset_fails():
    with pytest.raises(DatasetError, match="no records"):
        audit_records([])


def test_audit_distinguishes_malformed_timestamps_from_missing():
    records = load_records(FIXTURE)
    records[0].docs_ko[0].doc_insert_time = "2020-!1-01 00:00:00"
    report = audit_records(records)
    assert report["doc_insert_time"]["malformed"] == 1
    assert report["doc_insert_time"]["unknown"] == 2


def test_future_gold_is_diagnostic_and_not_filtered():
    records = load_records(FIXTURE)
    records[0].query_insert_time = records[0].docs_ko[0].doc_insert_time
    report = audit_records(records)
    assert report["gold_documents_inserted_after_query"] == 1
    assert report["queries_with_gold_inserted_after_query"] == 1
    assert report["queries_with_all_gold_inserted_after_query"] == 0
    assert report["gold_document_count_per_query"] == {1: 1, 2: 1}


def test_cli_is_offline_deterministic_and_does_not_claim_benchmark_provenance(
    tmp_path, monkeypatch
):
    def network_forbidden(*args, **kwargs):
        pytest.fail("normal audit must not access network")

    monkeypatch.setattr("temporalproof.data.source.urlopen", network_forbidden)
    output = tmp_path / "report.json"
    args = ["--jsonl", str(FIXTURE), "--output", str(output)]
    assert main(args) == 0
    first = output.read_bytes()
    assert main(args) == 0
    assert output.read_bytes() == first
    assert json.loads(first)["source"]["revision"] is None


def test_missing_archive_does_not_download(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "temporalproof.data.source.urlopen", lambda *a, **kw: pytest.fail("unexpected network")
    )
    output = tmp_path / "report.json"
    assert main(["--archive", str(tmp_path / "missing.zip"), "--output", str(output)]) == 1
    assert not output.exists()


def test_corrupt_cached_archive_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "temporalproof.data.source.urlopen", lambda *a, **kw: pytest.fail("unexpected network")
    )
    archive = tmp_path / "datasets.zip"
    archive.write_bytes(b"not the original dataset")
    with pytest.raises(DatasetError, match="checksum"):
        download_archive(archive)
    assert archive.read_bytes() == b"not the original dataset"


def test_download_is_pinned_and_checked_before_publication(tmp_path, monkeypatch):
    import io

    from temporalproof.data.source import ARCHIVE_URL

    calls = []

    def fake_response(url, timeout):
        calls.append((url, timeout))
        return io.BytesIO(b"untrusted replacement archive")

    monkeypatch.setattr("temporalproof.data.source.urlopen", fake_response)
    target = tmp_path / "datasets.zip"
    with pytest.raises(DatasetError, match="checksum"):
        download_archive(target)
    assert calls == [(ARCHIVE_URL, 30)]
    assert not target.exists()


def test_oversize_download_is_not_published(tmp_path, monkeypatch):
    import io

    monkeypatch.setattr("temporalproof.data.source.MAX_ARCHIVE_BYTES", 4)
    monkeypatch.setattr("temporalproof.data.source.urlopen", lambda *a, **kw: io.BytesIO(b"12345"))
    target = tmp_path / "datasets.zip"
    with pytest.raises(DatasetError, match="size limit"):
        download_archive(target)
    assert not target.exists()


def test_verified_download_is_cached_without_repeating_network(tmp_path, monkeypatch):
    import hashlib
    import io

    content = b"synthetic-download-fixture"
    monkeypatch.setattr(
        "temporalproof.data.source.ARCHIVE_SHA256", hashlib.sha256(content).hexdigest()
    )
    monkeypatch.setattr("temporalproof.data.source.urlopen", lambda *a, **kw: io.BytesIO(content))
    target = tmp_path / "datasets.zip"
    assert download_archive(target) == target
    assert target.read_bytes() == content
    assert list(tmp_path.iterdir()) == [target]
    monkeypatch.setattr(
        "temporalproof.data.source.urlopen", lambda *a, **kw: pytest.fail("cache accessed network")
    )
    assert download_archive(target) == target


def test_cli_refuses_to_overwrite_input(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_bytes(FIXTURE.read_bytes())
    before = source.read_bytes()
    with pytest.raises(SystemExit):
        main(["--jsonl", str(source), "--output", str(source)])
    assert source.read_bytes() == before


def test_cli_schema_error_is_sanitized_and_actionable(tmp_path, capsys):
    source = tmp_path / "bad.jsonl"
    source.write_text('{"private-content": 1}', encoding="utf-8")
    assert main(["--jsonl", str(source), "--output", str(tmp_path / "out.json")]) == 1
    stderr = capsys.readouterr().err
    assert "line 1" in stderr
    assert "private-content" not in stderr
