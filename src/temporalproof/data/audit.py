"""Offline, metadata-only Company Policies audit; network requires --download."""

import argparse
import hashlib
import io
import json
import sys
import unicodedata
import zipfile
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from statistics import median
from urllib.error import URLError

from pydantic import BaseModel

from temporalproof.data.source import (
    ARCHIVE_URL,
    DEFAULT_ARCHIVE,
    MAX_MEMBER_BYTES,
    MEMBER,
    REVISION,
    download_archive,
    sha256_file,
    verify_archive,
)
from temporalproof.data.timelyqa import (
    DatasetError,
    QueryRecord,
    iter_records,
    load_records,
    parse_timestamp,
)

GROUPING_VERSION = "nfkc_casefold_whitespace_v1"


def query_group_id(text: str) -> str:
    """Keep repeated query text together even when insertion dates differ.

    This is a conservative grouping key, not a near-duplicate/version-chain audit.
    """
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _duplicate_stats(values: Sequence[str]) -> dict[str, int]:
    counts = Counter(values)
    return {
        "unique": len(counts),
        "duplicate_groups": sum(count > 1 for count in counts.values()),
        "extra_occurrences": sum(count - 1 for count in counts.values()),
    }


def _field_counts(models: Sequence[BaseModel]) -> dict[str, int]:
    return dict(sorted(Counter(key for model in models for key in model.model_fields_set).items()))


def _timestamp_stats(models: Sequence[BaseModel], field: str) -> dict[str, object]:
    values = [getattr(model, field) for model in models]
    parsed = [parse_timestamp(value) for value in values]
    known = [value for value in parsed if isinstance(value, datetime)]
    return {
        "total": len(values),
        "absent": sum(field not in model.model_fields_set for model in models),
        "unknown": len(values) - len(known),
        "malformed": sum(
            raw not in (None, "") and value is None
            for raw, value in zip(values, parsed, strict=True)
        ),
        "naive": sum(value.tzinfo is None for value in known),
        "minimum": min(known).isoformat() if known else None,
        "maximum": max(known).isoformat() if known else None,
    }


def audit_records(records: Sequence[QueryRecord], sample_size: int = 50) -> dict[str, object]:
    """Inspect all records; sample only describes the first N in file order."""
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    if not records:
        raise DatasetError("dataset contains no records")
    query_ids = [record.index for record in records]
    docs = [doc for record in records for doc in record.docs_ko]
    if len(set(query_ids)) != len(query_ids):
        raise DatasetError("duplicate query record index")
    if len({doc.global_doc_id for doc in docs}) != len(docs):
        raise DatasetError("duplicate global document ID across records")

    sample = records[:sample_size]
    sample_docs = [doc for record in sample for doc in record.docs_ko]
    groups: dict[str, list[int]] = {}
    for record in records:
        groups.setdefault(query_group_id(record.query_ko), []).append(record.index)
    query_by_id = {record.index: record for record in records}
    different_time_groups = sum(
        len({query_by_id[index].query_insert_time for index in indices}) > 1
        for indices in groups.values()
        if len(indices) > 1
    )
    doc_lengths = [len(doc.text) for doc in docs]
    # A diagnostic only: inserted-after-query does not establish invalid applicability.
    parsed_queries = {record.index: parse_timestamp(record.query_insert_time) for record in records}
    parsed_docs = {doc.global_doc_id: parse_timestamp(doc.doc_insert_time) for doc in docs}
    future_gold_by_query = [
        sum(
            doc.global_doc_id in record.gold_docs_final_global
            and parsed_docs[doc.global_doc_id] is not None
            and parsed_queries[record.index] is not None
            and parsed_docs[doc.global_doc_id] > parsed_queries[record.index]
            for doc in record.docs_ko
        )
        for record in records
    ]
    return {
        "report_version": 1,
        "record_count": len(records),
        "document_count": len(docs),
        "category_count": len({record.category for record in records}),
        "empty_answer_count": sum(not record.answer.strip() for record in records),
        "record_fields_present": _field_counts(records),
        "document_fields_present": _field_counts(docs),
        "query_texts": _duplicate_stats([record.query_ko for record in records]),
        "document_texts": _duplicate_stats([doc.text for doc in docs]),
        "query_grouping": {
            "version": GROUPING_VERSION,
            "group_count": len(groups),
            "repeated_groups_with_different_query_insert_time": different_time_groups,
            "groups": dict(sorted(groups.items())),
            "split_assignment": "not_created; near-duplicate and family audit pending",
        },
        "gold_document_count_per_query": dict(
            sorted(Counter(len(record.gold_docs_final_global) for record in records).items())
        ),
        "local_to_global_gold_mapping": "validated_zero_based_exact_set_match",
        "gold_judgment_count": sum(len(record.gold_docs_final_global) for record in records),
        "gold_documents_inserted_after_query": sum(future_gold_by_query),
        "queries_with_gold_inserted_after_query": sum(count > 0 for count in future_gold_by_query),
        "queries_with_all_gold_inserted_after_query": sum(
            count == len(record.gold_docs_final_global)
            for record, count in zip(records, future_gold_by_query, strict=True)
        ),
        "query_insert_time": _timestamp_stats(records, "query_insert_time"),
        "doc_insert_time": _timestamp_stats(docs, "doc_insert_time"),
        "explicit_query_event_time_count": 0,
        "explicit_document_event_time_count": 0,
        "event_time_policy": "absent_in_supported_schema; no extraction or fallback",
        "document_length_characters": {
            "minimum": min(doc_lengths),
            "median": median(doc_lengths),
            "maximum": max(doc_lengths),
            "total": sum(doc_lengths),
        },
        "sample": {
            "selection": "first_n_records_in_file_order",
            "record_count": len(sample),
            "document_count": len(sample_docs),
            "record_fields_present": _field_counts(sample),
            "document_fields_present": _field_counts(sample_docs),
            "queries_containing_hangul": sum(
                any("\uac00" <= char <= "\ud7a3" for char in record.query_ko) for record in sample
            ),
        },
    }


def audit_archive(path: Path, sample_size: int = 50) -> dict[str, object]:
    verify_archive(path)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo(MEMBER)
        if info.file_size > MAX_MEMBER_BYTES:
            raise DatasetError("JSONL member exceeds size limit")
        with archive.open(info) as stream, io.TextIOWrapper(stream, encoding="utf-8") as text:
            report = audit_records(list(iter_records(text)), sample_size)
        report["source"] = {
            "revision": REVISION,
            "archive_url": ARCHIVE_URL,
            "archive_sha256": sha256_file(path),
            "archive_bytes": path.stat().st_size,
            "member": MEMBER,
            "member_bytes": info.file_size,
            "archive_members": sorted(archive.namelist()),
        }
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument(
        "--archive", type=Path, help="pinned archive; default: data/raw/timelyqa/..."
    )
    inputs.add_argument(
        "--jsonl", type=Path, help="local schema-compatible JSONL; provenance unknown"
    )
    parser.add_argument(
        "--download", action="store_true", help="explicitly fetch pinned 5.8 MB ZIP"
    )
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/audits/company_policies.json")
    )
    args = parser.parse_args(argv)
    if args.sample_size < 1:
        parser.error("--sample-size must be positive")
    if args.jsonl and args.download:
        parser.error("--download cannot be combined with --jsonl")
    source = args.jsonl or args.archive or DEFAULT_ARCHIVE
    if args.output.resolve() == source.resolve() or (
        args.output.exists() and source.exists() and args.output.samefile(source)
    ):
        parser.error("--output must not overwrite the input")
    try:
        if args.download:
            download_archive(source)
        if args.jsonl:
            report = audit_records(load_records(source), args.sample_size)
            report["source"] = {"revision": None, "jsonl_sha256": sha256_file(source)}
        else:
            report = audit_archive(source, args.sample_size)
        serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    except DatasetError as exc:
        print(f"audit failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, URLError, UnicodeError, zipfile.BadZipFile, KeyError):
        print("audit failed: input unavailable or unreadable", file=sys.stderr)
        return 1
    print(f"Audited {report['record_count']} queries / {report['document_count']} documents.")
    print(f"Metadata-only report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
