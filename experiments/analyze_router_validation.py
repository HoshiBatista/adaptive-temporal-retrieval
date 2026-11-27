"""Verify frozen validation evidence/replay and export only aggregate metadata."""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from temporalproof.data.source import sha256_file
from temporalproof.data.splits import DEFAULT_SPLIT, SplitManifest
from temporalproof.evaluation.baseline import _write_json, source_snapshot
from temporalproof.evaluation.pilot import read_rows
from temporalproof.evaluation.router_validation import ValidationConfig, evaluate, load_frozen_model


def verify_files(root, manifest, field):
    for name, digest in manifest[field].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or sha256_file(path) != digest:
            raise ValueError("evidence checksum/path mismatch")


def trec_rows(path):
    rows = {}
    for line in path.read_text().splitlines():
        q, _, doc, rank, score, _ = line.split()
        rows.setdefault(int(q), []).append((int(doc), int(rank), score))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument(
        "--training-run", type=Path, default=Path("artifacts/m4/router_train500_groupcv_v1")
    )
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument(
        "--old-fast",
        type=Path,
        default=Path("artifacts/m2/company_policies_validation100_verified_v1/hybrid_rrf"),
    )
    parser.add_argument(
        "--old-deep",
        type=Path,
        default=Path("artifacts/m3/korean_company_policies_validation100_v1/run"),
    )
    parser.add_argument("--interrupted-run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measured-on", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("aggregate exists; choose a new path")
    root = args.run
    config = ValidationConfig.model_validate_json((root / "config.json").read_text())
    split = SplitManifest.model_validate_json(args.split.read_text())
    load_frozen_model(args.training_run, split, config)
    manifest = json.loads((root / "manifest.json").read_text())
    verify_files(root, manifest, "files_sha256")
    if manifest["source_snapshot_sha256"] != source_snapshot(Path.cwd())["source_snapshot_sha256"]:
        raise ValueError("source differs from measurement")
    summary = json.loads((root / "summary.json").read_text())
    if evaluate(root, split, config) != summary:
        raise ValueError("frozen decisions/quality/interval replay differs")
    children = {}
    overlap = {}
    for kind, current, old in [
        ("fast", root / "fast", args.old_fast),
        ("deep", root / "deep/run", args.old_deep),
    ]:
        child = json.loads((current / "manifest.json").read_text())
        verify_files(current, child, "output_sha256")
        previous = json.loads((old / "manifest.json").read_text())
        verify_files(old, previous, "output_sha256")
        now = trec_rows(current / "run.trec")
        earlier = trec_rows(old / "run.trec")
        if len(earlier) != 100 or not earlier.keys() <= now.keys():
            raise ValueError("missing original validation100 overlap")
        if any(earlier[q] != now[q] for q in earlier):
            raise ValueError("original validation100 rankings/scores differ")
        qrels = {int(line.split()[0]): [] for line in (old / "qrels.tsv").read_text().splitlines()}
        for path, dest in [(old / "qrels.tsv", qrels)]:
            for line in path.read_text().splitlines():
                dest[int(line.split()[0])].append(line)
        new_qrels = {q: [] for q in qrels}
        for line in (current / "qrels.tsv").read_text().splitlines():
            q = int(line.split()[0])
            if q in new_qrels:
                new_qrels[q].append(line)
        if qrels != new_qrels:
            raise ValueError("original validation100 qrels differ")
        overlap[kind] = {
            "query_count": len(earlier),
            "rankings_scores_qrels_identical": True,
            "old_manifest_sha256": sha256_file(old / "manifest.json"),
        }
        children[kind] = {
            "manifest_sha256": sha256_file(current / "manifest.json"),
            "summary": json.loads((current / "summary.json").read_text()),
            "profile": child.get("encoder" if kind == "fast" else "reranker"),
            "encoder_cache": child.get("encoder_cache"),
            "environment": {
                key: child[key]
                for key in [
                    "python",
                    "platform",
                    "logical_cpu_count",
                    "thread_environment",
                    "packages",
                ]
            },
        }
    timings = [r["feature_ms"] for r in read_rows(root / "feature_timing.jsonl")]
    numeric_features = {r["query_id"]: r["features"] for r in read_rows(root / "features.jsonl")}
    fast = {r["query_id"]: r for r in read_rows(root / "fast/per_query.jsonl")}
    deep = {r["query_id"]: r for r in read_rows(root / "deep/run/per_query.jsonl")}
    cases = [
        {
            "query_id": q,
            "fast_ndcg_at_10": fast[q]["ndcg_at_10"],
            "deep_ndcg_at_10": deep[q]["ndcg_at_10"],
            "gain": deep[q]["ndcg_at_10"] - fast[q]["ndcg_at_10"],
        }
        for q in fast
    ]
    slices = {}
    for name, ids in {
        "low_agreement": [
            q
            for q, f in numeric_features.items()
            if f["top10_overlap_fraction"] < config.disagreement_threshold
        ],
        "explicit_year": [q for q, f in numeric_features.items() if f["explicit_year_count"] > 0],
    }.items():
        slices[name] = {
            "count": len(ids),
            "mean_deep_gain": float(
                np.mean([deep[q]["ndcg_at_10"] - fast[q]["ndcg_at_10"] for q in ids])
            )
            if ids
            else None,
        }
    interrupted = None
    if args.interrupted_run is not None:
        attempt = args.interrupted_run
        for name in ["config.json", "frozen_model.json", "features.jsonl", "decisions.json"]:
            if (attempt / name).read_bytes() != (root / name).read_bytes():
                raise ValueError(
                    "serialization-only restart changed frozen inference inputs/decisions"
                )
        earlier_predictions = read_rows(attempt / "predictions.jsonl")
        current_predictions = read_rows(root / "predictions.jsonl")
        for rows in [earlier_predictions, current_predictions]:
            for row in rows:
                row.pop("router_predict_ms")
        if earlier_predictions != current_predictions:
            raise ValueError("restart changed frozen router scores")
        interrupted = {
            "reason": "stopped before full deep completion to fix NumPy integer JSON serialization",
            "pre_deep_checkpoint_sha256": sha256_file(attempt / "pre_deep_checkpoint.json"),
            "features_model_config_decisions_scores_identical": True,
            "no_complete_deep_or_policy_quality_claim": True,
        }
    router_times = [r["router_predict_ms"] for r in read_rows(root / "predictions.jsonl")]
    result = {
        "measured_on": str(args.measured_on),
        "analysis_script_sha256": sha256_file(Path(__file__)),
        "interrupted_attempt": interrupted,
        "config": config.model_dump(mode="json"),
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "source_files_sha256": manifest["source_files_sha256"],
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "training_manifest_sha256": config.training_manifest_sha256,
        "final_model_sha256": config.final_model_sha256,
        "archive_sha256": manifest["archive_sha256"],
        "split_sha256": manifest["split_sha256"],
        "packages": manifest["packages"],
        "summary": summary,
        "stages": children,
        "original_validation100_overlap": overlap,
        "feature_timing_mean_ms": float(np.mean(timings)),
        "router_prediction_timing_ms": {
            "mean": float(np.mean(router_times)),
            "p50": float(np.quantile(router_times, 0.5)),
            "p95": float(np.quantile(router_times, 0.95)),
            "scope": (
                "warm single-query numeric features and parsed JSON model; "
                "no parsing/feature extraction/HTTP"
            ),
        },
        "between_models_allocated_bytes": json.loads(
            (root / "pre_deep_checkpoint.json").read_text()
        )["between_models_allocated_bytes"],
        "diagnostic_slices": slices,
        "largest_deep_regressions": sorted(cases, key=lambda r: (r["gain"], r["query_id"]))[:5],
        "largest_deep_gains": sorted(cases, key=lambda r: (-r["gain"], r["query_id"]))[:5],
        "verification": "all evidence checksums, exact CPU decisions/summary/interval replay; "
        "original validation100 exact GPU scores/ranks/qrels; no second full GPU run",
        "limitations": [
            "development validation includes model-selection validation100",
            "single train500 model; no validation fitting/threshold tuning/test",
            "class-weighted scores are not calibrated probabilities",
            "matched query counts are not matched measured wall-clock budgets",
            "offline cohort quota; modeled means are not live service p95",
            "insertion-time/text proxies are not valid time/version-chain gold",
            "dataset redistribution terms unresolved; raw records/weights omitted",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.output, result)
    print(json.dumps({"primary": summary["primary"], "overlap": overlap}, sort_keys=True))


if __name__ == "__main__":
    main()
