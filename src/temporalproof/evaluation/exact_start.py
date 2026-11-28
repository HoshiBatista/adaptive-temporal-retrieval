"""Compare bounded exact policy-start promotion on cached training candidates."""

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from temporalproof.data.source import (
    ARCHIVE_SHA256,
    DEFAULT_ARCHIVE,
    load_archive_records,
    sha256_file,
)
from temporalproof.data.splits import DEFAULT_SPLIT, SplitManifest, verify_split
from temporalproof.data.timelyqa import DatasetError, parse_timestamp
from temporalproof.evaluation.baseline import evaluate_query, source_snapshot
from temporalproof.evaluation.document_time_audit import SPLIT_SHA256, train_passages
from temporalproof.evaluation.pilot import read_rows
from temporalproof.evaluation.qet_ablation import (
    ALL_METRICS,
    canonical_train_ids,
    paired_comparison,
)
from temporalproof.evaluation.query_time_audit import AuditConfig
from temporalproof.evaluation.rank_blend import load_trec, verify_source
from temporalproof.retrieval.bm25 import Hit
from temporalproof.retrieval.document_time import analyze_document_time
from temporalproof.retrieval.query_time import QueryTimeSelection, analyze_query_time
from temporalproof.routing.targets import training_groups


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")
    partition: Literal["train"]
    seed: Literal[1729]
    query_count: Literal[500]
    top_k: Literal[100]
    bootstrap_samples: Literal[2000]
    promotion_rule: Literal["one_adjacent_swap_of_topmost_match_within_top20"]
    document_scope: Literal["M4.6 normalized-text representatives only"]
    query_scope: Literal["M4.4 unique policy target with day precision"]
    primary: Literal["explicit_start_vs_fast_full_qrel_ndcg10"]
    control: Literal["insertion_calendar_day_with_same_scope_and_promotion"]
    train_gate: Literal["positive_lower_bound_of_paired_group_95_interval_vs_fast"]
    train_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    document_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    query_audit_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def query_day(selection: QueryTimeSelection) -> date | None:
    index = selection.selected_index
    if (
        selection.reason == "unique_policy_precise_target"
        and index is not None
        and selection.mentions[index].precision == "day"
        and selection.selected_time is not None
    ):
        return selection.selected_time.date()
    return None


def match_ranks(hits: Sequence[Hit], target: date | None, days: Mapping[int, date]) -> list[int]:
    if target is None:
        return []
    return [rank for rank, hit in enumerate(hits, 1) if days.get(hit.doc_id) == target]


def promote(hits: Sequence[Hit], matches: Sequence[int]) -> list[Hit]:
    ranked = list(hits)
    if matches and 2 <= matches[0] <= min(20, len(ranked)):
        index = matches[0] - 1
        ranked[index - 1], ranked[index] = ranked[index], ranked[index - 1]
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/m4_exact_start_train.json")
    )
    parser.add_argument(
        "--train-run", type=Path, default=Path("artifacts/m4/company_policies_train500_v1")
    )
    parser.add_argument(
        "--query-audit", type=Path, default=Path("artifacts/m4/query_time_roles_train_v2")
    )
    parser.add_argument(
        "--document-audit",
        type=Path,
        default=Path("experiments/results/m4_document_time_audit_train_v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    config = Config.model_validate_json(args.config.read_text())
    source = verify_source(args.train_run, config.train_manifest_sha256)
    query_audit = verify_source(args.query_audit, config.query_audit_sha256)
    if sha256_file(args.document_audit) != config.document_audit_sha256:
        raise DatasetError("document audit checksum mismatch")
    document_audit = json.loads(args.document_audit.read_text())
    policy = json.loads((args.query_audit / "frozen_policy.json").read_text())
    query_config = AuditConfig.model_validate_json((args.query_audit / "config.json").read_text())
    retrieval = Path(__file__).parents[1] / "retrieval"
    frozen = document_audit["frozen_extractor"]
    for file, expected in (
        ("query_time.py", policy["query_time_module_sha256"]),
        ("query_time.py", frozen["date_parser_sha256"]),
        ("document_time.py", frozen["module_sha256"]),
    ):
        if sha256_file(retrieval / file) != expected:
            raise DatasetError("frozen extractor checksum mismatch")
    for manifest in (source, query_audit, document_audit):
        if manifest["split_sha256"] != SPLIT_SHA256 or manifest["archive_sha256"] != ARCHIVE_SHA256:
            raise DatasetError("source data provenance mismatch")
    if sha256_file(DEFAULT_SPLIT) != SPLIT_SHA256:
        raise DatasetError("split checksum mismatch")
    split = SplitManifest.model_validate_json(DEFAULT_SPLIT.read_text())
    records = load_archive_records(DEFAULT_ARCHIVE)
    verify_split(split, records, ARCHIVE_SHA256)
    ids = canonical_train_ids(split, config, json.loads((args.train_run / "plan.json").read_text()))
    baseline = load_trec(args.train_run / "fast/run.trec", ids, config.top_k)
    by_query = {record.index: record for record in records}
    audit_rows = {
        r["query_id"]: r["analysis"] for r in read_rows(args.query_audit / "query_audit.jsonl")
    }
    population = train_passages(records, set(split.query_ids("train")))
    if len(population) != document_audit["population"]["passages"]:
        raise DatasetError("document audit population differs")
    pool_ids = {hit.doc_id for hits in baseline.values() for hit in hits}
    scoped = {doc.global_doc_id: doc for doc in population if doc.global_doc_id in pool_ids}
    days = {"explicit_start": {}, "insertion_day": {}}
    for doc_id, doc in scoped.items():
        explicit = analyze_document_time(doc.text).start
        inserted = parse_timestamp(doc.doc_insert_time)
        if explicit is not None:
            days["explicit_start"][doc_id] = explicit.date()
        if inserted is not None:
            days["insertion_day"][doc_id] = inserted.date()

    counts = {name: Counter() for name in days}
    eligible = 0
    decisions = {}
    for q in ids:
        selection = analyze_query_time(
            by_query[q].query_ko,
            before_characters=query_config.before_characters,
            after_characters=query_config.after_characters,
        )
        if selection.model_dump(mode="json") != audit_rows[q]:
            raise DatasetError("query date differs from frozen audit")
        target = query_day(selection)
        if target is not None:
            eligible += 1
        decisions[q] = {}
        for name, mapping in days.items():
            ranks = match_ranks(baseline[q], target, mapping)
            updated = promote(baseline[q], ranks)
            decisions[q][name] = updated
            counts[name].update(
                {
                    "matching_occurrences_top100": len(ranks),
                    "queries_with_match_top100": int(bool(ranks)),
                    "queries_with_match_top20": int(bool(ranks) and ranks[0] <= 20),
                    "queries_already_first": int(bool(ranks) and ranks[0] == 1),
                    "queries_promoted": int(updated != baseline[q]),
                }
            )
    result = {
        "stage": "M4.7",
        "config": config.model_dump(),
        "config_sha256": sha256_file(args.config),
        "source_snapshot_sha256": source_snapshot(Path.cwd())["source_snapshot_sha256"],
        "partition": "train",
        "validation_evaluated": False,
        "test_evaluated": False,
        "coverage": {
            "eligible_queries": eligible,
            "unique_pool_documents": len(pool_ids),
            "audited_representatives_in_pools": len(scoped),
            "representative_candidate_occurrences": sum(
                hit.doc_id in scoped for hits in baseline.values() for hit in hits
            ),
            "explicit_day_documents": len(days["explicit_start"]),
            "insertion_day_documents": len(days["insertion_day"]),
            "controls": {name: dict(counter) for name, counter in counts.items()},
            "queries_with_different_explicit_and_insertion_rankings": sum(
                v["explicit_start"] != v["insertion_day"] for v in decisions.values()
            ),
        },
    }
    if not args.coverage_only:
        evaluate(result, ids, by_query, baseline, decisions, split, config, args.train_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")


def evaluate(result, ids, records, baseline, decisions, split, config, train_run):
    qrels = "".join(
        f"{q}\t0\t{d}\t1\n" for q in ids for d in sorted(records[q].gold_docs_final_global)
    )
    if qrels.encode() != (train_run / "fast/qrels.tsv").read_bytes():
        raise DatasetError("cached qrels differ from published records")
    original_rows = {row["query_id"]: row for row in read_rows(train_run / "fast/per_query.jsonl")}
    methods = ("fast", "explicit_start", "insertion_day")
    rows = {name: [] for name in methods}
    for q in ids:
        for name in methods:
            hits = baseline[q] if name == "fast" else decisions[q][name]
            if set(hits) != set(baseline[q]) or len(hits) != len(baseline[q]):
                raise DatasetError("pool or score changed")
            row = evaluate_query(records[q], hits)
            if row["recall_at_100"] != original_rows[q]["recall_at_100"]:
                raise DatasetError("candidate recall changed")
            if name == "fast" and any(row[m] != original_rows[q][m] for m in ALL_METRICS):
                raise DatasetError("baseline metric replay differs")
            rows[name].append(row)
    groups = training_groups(split)
    contrasts = [
        ("explicit_start", "fast"),
        ("insertion_day", "fast"),
        ("explicit_start", "insertion_day"),
    ]
    result["groups"] = len({groups[q] for q in ids})
    result["metrics"] = {
        name: {m: sum(row[m] for row in data) / len(data) for m in ALL_METRICS}
        for name, data in rows.items()
    }
    result["comparisons"] = {
        f"{a}_vs_{b}": paired_comparison(rows, groups, a, b, config) for a, b in contrasts
    }
    result["train_gate_passed"] = (
        result["comparisons"]["explicit_start_vs_fast"]["paired_group_bootstrap_95_interval"][0] > 0
    )
    result["explicit_start_primary_deltas"] = {
        str(a["query_id"]): a["ndcg_at_10"] - b["ndcg_at_10"]
        for a, b in zip(rows["explicit_start"], rows["fast"], strict=True)
        if a["ndcg_at_10"] != b["ndcg_at_10"]
    }


if __name__ == "__main__":
    main()
