"""Verify exact cached experiment replay and export aggregate-only evidence."""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from temporalproof.data.source import sha256_file
from temporalproof.evaluation.baseline import _write_json, source_snapshot
from temporalproof.evaluation.pilot import read_rows
from temporalproof.evaluation.rank_blend import verify_source
from temporalproof.routing.cross_validation import grouped_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument(
        "--validation-run", type=Path, default=Path("artifacts/m4/router_validation495_v2")
    )
    parser.add_argument("--measured-on", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("aggregate exists; choose a new path")
    a = verify_source(args.run, sha256_file(args.run / "manifest.json"))
    b = verify_source(args.replay, sha256_file(args.replay / "manifest.json"))
    if a != b:
        raise ValueError("cached experiment replay manifest differs")
    for name in a["files_sha256"]:
        if (args.run / name).read_bytes() != (args.replay / name).read_bytes():
            raise ValueError("cached experiment file replay differs")
    if a["source_snapshot_sha256"] != source_snapshot(Path.cwd())["source_snapshot_sha256"]:
        raise ValueError("experiment source differs from implementation")
    checkpoint = json.loads((args.run / "train_checkpoint.json").read_text())
    for name, digest in checkpoint.items():
        if sha256_file(args.run / name) != digest:
            raise ValueError("train selection checkpoint differs")
    summary = json.loads((args.run / "summary.json").read_text())
    config = json.loads((args.run / "config.json").read_text())
    verify_source(args.validation_run, config["validation_manifest_sha256"])
    # Compare the same frozen routes before/after blend, without reselecting routes/weight.
    old_fast = {r["query_id"]: r for r in read_rows(args.validation_run / "fast/per_query.jsonl")}
    old_deep = {
        r["query_id"]: r for r in read_rows(args.validation_run / "deep/run/per_query.jsonl")
    }
    from temporalproof.evaluation.rank_blend import load_trec, metrics
    from temporalproof.retrieval.rank_blend import blend_ranks

    ids = list(old_fast)
    fast = load_trec(args.validation_run / "fast/run.trec", ids, config["top_k"])
    deep = load_trec(args.validation_run / "deep/run/run.trec", ids, config["top_k"])
    qrels = {}
    for line in (args.validation_run / "fast/qrels.tsv").read_text().splitlines():
        q, _, doc, _ = line.split()
        qrels.setdefault(int(q), set()).add(int(doc))
    weight = summary["validation"]["selected_weight"]
    quality = {
        q: metrics(blend_ranks(fast[q], deep[q], weight, config["rank_offset"]), qrels[q])
        for q in ids
    }
    plan = json.loads((args.validation_run / "plan.json").read_text())
    groups = [plan["query_groups"][str(q)] for q in ids]
    decisions = json.loads((args.validation_run / "decisions.json").read_text())
    comparisons = []
    for name, selected in [
        ("all_queries", set(ids)),
        *[
            (kind, set(decisions[f"{kind}_budget_0.1"]))
            for kind in ["logistic", "agreement", "random"]
        ],
    ]:
        delta = [
            quality[q]["ndcg_at_10"] - old_deep[q]["ndcg_at_10"] if q in selected else 0.0
            for q in ids
        ]
        comparisons.append(
            {
                "scope": name,
                "mean_ndcg_delta_blend_vs_original_deep": float(np.mean(delta)),
                "paired_group_bootstrap_95_interval": grouped_interval(
                    delta, groups, config["seed"], config["bootstrap_samples"]
                ),
                "secondary_diagnostic": True,
            }
        )
    fold_rows = json.loads((args.run / "fold_selection.json").read_text())
    public = {
        "measured_on": str(args.measured_on),
        "config": config,
        "source_snapshot_sha256": a["source_snapshot_sha256"],
        "source_files_sha256": a["source_files_sha256"],
        "manifest_sha256": sha256_file(args.run / "manifest.json"),
        "train_manifest_sha256": a["train_manifest_sha256"],
        "validation_manifest_sha256": a["validation_manifest_sha256"],
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "selection_checkpoint_sha256": sha256_file(args.run / "train_checkpoint.json"),
        "summary": summary,
        "folds": [
            {
                "fold": r["fold"],
                "selected_weight": r["selected_weight"],
                "selection_query_count": len(r["selection_query_ids"]),
                "held_out_query_count": len(r["held_out_query_ids"]),
                "training_mean_by_weight": r["training_mean_by_weight"],
            }
            for r in fold_rows
        ],
        "secondary_paired_comparisons": comparisons,
        "exact_replay_all_files_including_manifest": True,
        "limitations": [
            "hypothesis designed after negative development validation; test untouched",
            "only one train-selected blend weight evaluated on validation",
            "transfer keeps original routes/model, not retrained for the new utility",
            "primary is fold-selected all-query blend, not best-looking transfer policy",
            "conditional group intervals do not redo weight selection",
            "positive blend weights still require complete expensive reranking",
            "source stage costs omit unmeasured new fusion/router overhead",
            "unjudged is not established irrelevant; no new gold labels",
            "pool oracle is an offline diagnostic, not an inference feature",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, public)
    print(json.dumps({"selected_weight": weight, "comparisons": comparisons}, sort_keys=True))


if __name__ == "__main__":
    main()
