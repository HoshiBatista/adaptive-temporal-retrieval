"""Train-only text audit and frozen prospective query-time policy; no retrieval evaluation."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from temporalproof.data.source import (
    ARCHIVE_SHA256,
    DEFAULT_ARCHIVE,
    load_archive_records,
    sha256_file,
)
from temporalproof.data.splits import DEFAULT_SPLIT, SplitManifest, select_queries, verify_split
from temporalproof.data.timelyqa import DatasetError, parse_timestamp
from temporalproof.evaluation.baseline import _write_json, source_snapshot
from temporalproof.evaluation.pilot import write_rows
from temporalproof.evaluation.rank_blend import load_trec, verify_source
from temporalproof.retrieval.query_time import RULE_VERSION, analyze_query_time
from temporalproof.retrieval.timelyrag import load_reference


class AuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    partition: Literal["train"]
    seed: int = Field(strict=True, ge=0)
    query_count: int = Field(strict=True, ge=1)
    sample_count: int = Field(strict=True, ge=20, le=50)
    sample_policy: Literal["sha256_group_then_min_query_id_v1"]
    before_characters: int = Field(strict=True, ge=1)
    after_characters: int = Field(strict=True, ge=1)
    rule_version: Literal[RULE_VERSION]
    precise_policy: Literal["unique_asof_then_policy_day_or_finer_v1"]
    abstention_action: Literal["keep_fast_ranking"]
    reference_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    train500_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    train500_query_count: int = Field(strict=True, ge=1)
    next_experiment: Literal["fixed_train500_hybrid_pool_qet_override_only"]
    validation_evaluation_allowed: Literal[False]
    test_evaluation_allowed: Literal[False]


def sample_ids(split: SplitManifest, config: AuditConfig) -> list[int]:
    """Uniform deterministic source-group sample, one minimum-ID representative each."""
    groups = sorted(
        split.groups["train"],
        key=lambda g: hashlib.sha256(
            (f"{config.seed}:date-role-audit:" + ",".join(map(str, sorted(g)))).encode()
        ).hexdigest(),
    )
    if len(groups) < config.sample_count:
        raise DatasetError("too few train groups for the predeclared inspection sample")
    return [min(g) for g in groups[: config.sample_count]]


def analyze_train(records, split, config, reference):
    ids = split.query_ids("train")
    if len(ids) != config.query_count:
        raise DatasetError("date audit requires the complete declared train partition")
    by_id = {r.index: r for r in records}
    rows = []
    for q in ids:
        record = by_id[q]
        parsed = analyze_query_time(
            record.query_ko,
            before_characters=config.before_characters,
            after_characters=config.after_characters,
        )
        qit = reference.utils.to_dt(record.query_insert_time)
        qet, granularity = reference.utils.extract_qet_from_query(record.query_ko, qit_hint=qit)
        matches = [
            i
            for i, m in enumerate(parsed.mentions)
            if m.status == "explicit"
            and m.lower == qet
            and (
                m.precision == granularity
                or (granularity == "hour" and m.precision in {"hour", "minute", "second"})
            )
        ]
        roles = sorted({role for i in matches for role in parsed.mentions[i].role_cues})
        rows.append(
            {
                "query_id": q,
                "query": record.query_ko,
                "analysis": parsed.model_dump(mode="json"),
                "reference_time": qet.isoformat() if qet is not None else None,
                "reference_granularity": granularity,
                "reference_matching_indices": matches,
                "reference_matched_role_cues": roles,
                "reference_match_status": (
                    "not_extracted"
                    if qet is None
                    else "unique_explicit_match"
                    if len(matches) == 1
                    else "multiple_matching_mentions"
                    if matches
                    else "no_explicit_match"
                ),
                "query_insert_known": qit is not None,
                "selected_after_query_insert": parsed.selected_time > qit
                if parsed.selected_time is not None and qit is not None
                else None,
            }
        )
    return rows


def summarize(rows, split):
    analyses = [r["analysis"] for r in rows]
    mentions = [m for a in analyses for m in a["mentions"]]
    selected = [r for r in rows if r["analysis"]["action"] == "use_explicit_point"]
    return {
        "query_count": len(rows),
        "source_group_count": len(split.groups["train"]),
        "queries_with_mentions": sum(bool(a["mentions"]) for a in analyses),
        "queries_with_multiple_mentions": sum(len(a["mentions"]) > 1 for a in analyses),
        "mention_count": len(mentions),
        "mention_status_counts": dict(Counter(m["status"] for m in mentions)),
        "mention_precision_counts": dict(Counter(m["precision"] for m in mentions)),
        "mention_role_cue_counts": dict(Counter(role for m in mentions for role in m["role_cues"])),
        "mentions_without_role_cue": sum(not m["role_cues"] for m in mentions),
        "mentions_with_conflicting_cues": sum(m["role_conflict"] for m in mentions),
        "selection_action_counts": dict(Counter(a["action"] for a in analyses)),
        "decision_reason_counts": dict(Counter(a["reason"] for a in analyses)),
        "reference_extracted_count": sum(r["reference_time"] is not None for r in rows),
        "reference_match_counts": dict(Counter(r["reference_match_status"] for r in rows)),
        "reference_selected_with_employment_cue_count": sum(
            "employment" in r["reference_matched_role_cues"] for r in rows
        ),
        "reference_selected_with_only_employment_cue_count": sum(
            r["reference_matched_role_cues"] == ["employment"] for r in rows
        ),
        "selected_reference_point_agreement_count": sum(
            r["reference_time"] == r["analysis"]["selected_time"] for r in selected
        ),
        "selected_reference_point_difference_count": sum(
            r["reference_time"] != r["analysis"]["selected_time"] for r in selected
        ),
        "selected_after_query_insert_count": sum(
            r["selected_after_query_insert"] is True for r in selected
        ),
        "scope": (
            "heuristic coverage and agreement only; "
            "no temporal-role accuracy/gold or retrieval score"
        ),
        "role_cues_are_gold": False,
        "validation_queries_analyzed": False,
        "test_queries_analyzed": False,
        "new_neural_inference": False,
        "new_router_fit": False,
    }


def audit_cached_pools(root, records, split, rows, config):
    manifest = verify_source(root, config.train500_manifest_sha256)
    ids = select_queries(split, "train", config.train500_query_count, config.seed)
    plan = json.loads((root / "plan.json").read_text())
    if (
        plan["selected_query_ids"] != ids
        or plan["validation_partition_evaluated"]
        or plan["test_partition_evaluated"]
    ):
        raise DatasetError("cached candidate pool is not the canonical train-only source")
    hits = load_trec(root / "fast/run.trec", ids, 100)
    by_query = {r.index: r for r in records}
    documents = {d.global_doc_id: d for r in records for d in r.docs_ko}
    analysis = {r["query_id"]: r for r in rows}
    pools = []
    for q in ids:
        qit = parse_timestamp(by_query[q].query_insert_time)
        times = [parse_timestamp(documents[h.doc_id].doc_insert_time) for h in hits[q]]
        pools.append(
            {
                "query_id": q,
                "candidate_count": len(times),
                "insertion_unknown_count": sum(t is None for t in times),
                "later_or_equal_insert_count": sum(
                    t is not None and qit is not None and t >= qit for t in times
                ),
                "document_event_metadata_count": 0,
                "selected_explicit_query_anchor": analysis[q]["analysis"]["action"]
                == "use_explicit_point",
            }
        )
    return pools, {
        "query_count": len(ids),
        "candidate_count": sum(r["candidate_count"] for r in pools),
        "insertion_unknown_count": sum(r["insertion_unknown_count"] for r in pools),
        "later_or_equal_insert_count": sum(r["later_or_equal_insert_count"] for r in pools),
        "explicit_document_event_metadata_count": 0,
        "queries_with_selected_precise_anchor": sum(
            r["selected_explicit_query_anchor"] for r in pools
        ),
        "source_manifest_sha256": config.train500_manifest_sha256,
        "source_split_sha256": manifest["split_sha256"],
        "scope": (
            "cached train500 candidate timestamps only; no gold/gain/utility used; no reranking"
        ),
    }


def policy_contract(config):
    return {
        "version": 1,
        "rule_version": RULE_VERSION,
        "policy": config.precise_policy,
        "query_time_module_sha256": sha256_file(
            Path(__file__).parents[1] / "retrieval/query_time.py"
        ),
        "context_characters": {
            "before": config.before_characters,
            "after": config.after_characters,
        },
        "selection": (
            "unique explicit as_of before policy; "
            "conflicting/comparative/coarse/invalid/missing-year => bypass"
        ),
        "canonicalization": (
            "full calendar date uses naive midnight; "
            "explicit finer time preserved; no UTC/validity claim"
        ),
        "coarse_dates": (
            "keep calendar resolution extent; never force year/month to January/first-day QET"
        ),
        "abstention": (
            "preserve original fast ranking; do not pass None to an insertion-fallback reranker"
        ),
        "next_train_only_ablation": {
            "source_train_manifest_sha256": config.train500_manifest_sha256,
            "queries": "canonical train500 seed1729, same frozen hybrid top100 pool and scores",
            "controls": ["unchanged fast ranking", "native reference auto-alpha"],
            "candidate": "replace only QET with selected explicit point; bypass on abstention",
            "hold_fixed": [
                "original query text/granularity helpers",
                "document insertion/event fallback",
                "alpha/gate coefficients",
                "native NumPy tie policy",
                "published qrels",
                "top100 candidate pool",
            ],
            "primary_metric": "full-qrel nDCG@10 paired by source group; reference-prefix separate",
            "new_model_or_grid_search": False,
            "oracle_alpha": False,
            "gate": "train-only comparison and native parity before any validation invocation",
        },
        "validation_invoked": False,
        "test_invoked": False,
        "retrieval_experiment_executed": False,
        "accuracy_claim": "no benchmark temporal-role gold available; cue coverage is not accuracy",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/m4_query_time_roles_train.json")
    )
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--reference-source", type=Path, default=DEFAULT_ARCHIVE.parent)
    parser.add_argument(
        "--reference-manifest", type=Path, default=Path("configs/datasets/timelyrag_reference.json")
    )
    parser.add_argument(
        "--train-run", type=Path, default=Path("artifacts/m4/company_policies_train500_v1")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        config = AuditConfig.model_validate_json(args.config.read_text())
        if args.output.exists():
            raise DatasetError("output exists; choose a new audit directory")
        records = load_archive_records(args.archive)
        split = SplitManifest.model_validate_json(args.split.read_text())
        verify_split(split, records, ARCHIVE_SHA256)
        reference = load_reference(
            args.reference_source, args.reference_manifest, config.reference_manifest_sha256
        )
        root = args.output
        root.mkdir(parents=True, exist_ok=False)
        provenance = {
            **source_snapshot(Path.cwd()),
            "archive_sha256": ARCHIVE_SHA256,
            "split_sha256": sha256_file(args.split),
            "config_file_sha256": sha256_file(args.config),
            "reference_manifest_sha256": config.reference_manifest_sha256,
        }
        inspected = sample_ids(split, config)
        _write_json(root / "config.json", config.model_dump())
        _write_json(
            root / "plan.json",
            {
                "config": config.model_dump(),
                "provenance": provenance,
                "selected_train_query_ids": split.query_ids("train"),
                "inspection_sample_ids": inspected,
                "sampling": (
                    "40 groups by fixed hash, minimum query ID representative; no outcomes/labels"
                ),
                "scope": (
                    "all and only train queries analyzed for date roles; "
                    "source grouping verification is dataset-wide"
                ),
            },
        )
        rows = analyze_train(records, split, config, reference)
        summary = summarize(rows, split)
        pools, summary["candidate_pool_coverage"] = audit_cached_pools(
            args.train_run, records, split, rows, config
        )
        if summary["candidate_pool_coverage"]["source_split_sha256"] != provenance["split_sha256"]:
            raise DatasetError("cached pools use a different split file")
        train_ids = set(split.query_ids("train"))
        train_docs = [d for r in records if r.index in train_ids for d in r.docs_ko]
        summary["train_associated_passages"] = {
            "count": len(train_docs),
            "unknown_insertion_count": sum(
                parse_timestamp(d.doc_insert_time) is None for d in train_docs
            ),
            "explicit_event_metadata_count": 0,
        }
        write_rows(root / "query_audit.jsonl", rows)
        write_rows(root / "pool_audit.jsonl", pools)
        by_id = {r["query_id"]: r for r in rows}
        _write_json(root / "inspection40.json", [by_id[q] for q in inspected])
        _write_json(root / "summary.json", summary)
        _write_json(root / "frozen_policy.json", policy_contract(config))
        _write_json(
            root / "manifest.json",
            {
                **provenance,
                "files_sha256": {
                    p.name: sha256_file(p) for p in sorted(root.iterdir()) if p.is_file()
                },
            },
        )
    except (OSError, ValueError, KeyError) as exc:
        print(f"query-time audit failed: {exc}; source evidence retained", file=sys.stderr)
        return 1
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
