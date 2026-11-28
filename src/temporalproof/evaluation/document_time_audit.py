"""Audit document rule-start cues on deduplicated training passages."""

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from temporalproof.data.source import (
    ARCHIVE_SHA256,
    DEFAULT_ARCHIVE,
    REVISION,
    load_archive_records,
    sha256_file,
)
from temporalproof.data.splits import DEFAULT_SPLIT, SplitManifest, normalize, verify_split
from temporalproof.data.timelyqa import Document, QueryRecord, parse_timestamp
from temporalproof.evaluation.baseline import source_snapshot
from temporalproof.retrieval.document_time import DocumentTime, analyze_document_time

SEED = 1729
SAMPLE_SIZE = 40
SPLIT_SHA256 = "4aab958538041f019363a8ffcc5c5d15d030ebe408d869c17063bf9c3c566fa7"


def train_passages(records: Sequence[QueryRecord], train_ids: set[int]) -> list[Document]:
    """Remove exact normalized text shared with either held-out partition."""
    held_out = {
        normalize(doc.text)
        for record in records
        if record.index not in train_ids
        for doc in record.docs_ko
    }
    unique: dict[str, Document] = {}
    for record in records:
        if record.index not in train_ids:
            continue
        for doc in record.docs_ko:
            key = normalize(doc.text)
            if key in held_out:
                continue
            if key not in unique or doc.global_doc_id < unique[key].global_doc_id:
                unique[key] = doc
    return sorted(unique.values(), key=lambda doc: doc.global_doc_id)


def sample_passages(documents: Sequence[Document]) -> list[Document]:
    return sorted(
        documents,
        key=lambda d: (
            hashlib.sha256(f"{SEED}:document-time:{d.global_doc_id}".encode()).digest(),
            d.global_doc_id,
        ),
    )[:SAMPLE_SIZE]


def summarize(documents: Sequence[Document], analyses: dict[int, DocumentTime]) -> dict:
    reasons: Counter[str] = Counter()
    mentions: Counter[str] = Counter()
    insertion: Counter[str] = Counter()
    for doc in documents:
        result = analyses[doc.global_doc_id]
        reasons[result.reason] += 1
        mentions.update(date.kind for date in result.dates)
        if result.start is None:
            continue
        inserted = parse_timestamp(doc.doc_insert_time)
        if inserted is None:
            insertion["unknown"] += 1
        elif inserted.date() == result.start.date():
            insertion["same_calendar_day"] += 1
        elif result.start < inserted:
            insertion["start_before_insertion_day"] += 1
        else:
            insertion["start_after_insertion_day"] += 1
    return {
        "passages": len(documents),
        "decision_counts": dict(reasons),
        "mention_counts": dict(mentions),
        "selected_start_vs_insertion": dict(insertion),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    split_hash = sha256_file(args.split)
    if split_hash != SPLIT_SHA256:
        parser.error("the audit requires the frozen Company Policies split")
    split = SplitManifest.model_validate_json(args.split.read_text())
    records = load_archive_records(args.archive)
    verify_split(split, records, ARCHIVE_SHA256)
    train_ids = set(split.query_ids("train"))
    documents = train_passages(records, train_ids)
    sample = sample_passages(documents)
    if len(sample) != SAMPLE_SIZE:
        parser.error("insufficient distinct training passages")
    analyses = {doc.global_doc_id: analyze_document_time(doc.text) for doc in documents}
    source = source_snapshot(Path.cwd())
    result = {
        "stage": "M4.6",
        "partition": "train",
        "revision": REVISION,
        "archive_sha256": ARCHIVE_SHA256,
        "split_sha256": split_hash,
        "source_snapshot_sha256": source["source_snapshot_sha256"],
        "script_sha256": sha256_file(Path(__file__)),
        "train_query_count": len(train_ids),
        "train_passage_occurrences": sum(len(r.docs_ko) for r in records if r.index in train_ids),
        "sample_seed": SEED,
        "sample_policy": "sha256_seed_document-time_min_id_normalized_text_exclude_heldout_v1",
        "inspection_ids": [d.global_doc_id for d in sample],
        "population": summarize(documents, analyses),
        "sample": summarize(sample, analyses),
        "frozen_extractor": {
            "module_sha256": sha256_file(Path(__file__).parents[1] / "retrieval/document_time.py"),
            "date_parser_sha256": sha256_file(
                Path(__file__).parents[1] / "retrieval/query_time.py"
            ),
            "output": "unique explicitly stated rule-start day or unknown",
            "insertion_fallback": False,
            "validity_interval_inference": False,
            "contract": "document_rule_start_v1",
        },
        "retrieval_evaluated": False,
        "validation_or_test_passages_analyzed": False,
        "neural_inference": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
