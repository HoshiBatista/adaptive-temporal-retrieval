"""Reviewed, hash-pinned upstream helpers and a qrel-free automatic-alpha adapter.

Original upstream files stay outside git. No pipeline, model wrapper or downloads run.
"""

import ast
import importlib.util
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType

import numpy as np

from temporalproof.data.source import REVISION, sha256_file
from temporalproof.data.timelyqa import DatasetError
from temporalproof.retrieval.bm25 import Hit

CONFIG_NAMES = {
    "K",
    "STAGE1_TOPN",
    "ALPHA_GRID",
    "TIME_STD_MAX",
    "TEXT_STD_MAX",
    "MISS_MARGIN",
    "GAP_MAX",
}
MODULE_NAMES = (
    "config",
    "time_utils",
    "time_scores",
    "alpha",
    "metrics",
    "io_utils",
    "TimelyQABench_datasets",
)


@dataclass
class ReferenceCode:
    config: ModuleType
    utils: ModuleType
    scores: ModuleType
    alpha: ModuleType
    metrics: ModuleType
    datasets: ModuleType
    original_functions: dict
    manifest: dict


def verify_reference(directory: Path, manifest_path: Path, expected_sha256: str) -> dict:
    if sha256_file(manifest_path) != expected_sha256:
        raise DatasetError("reference manifest differs from predeclared checksum")
    manifest = json.loads(manifest_path.read_text())
    expected_names = {
        "alpha.py",
        "time_scores.py",
        "pipeline.py",
        "io_utils.py",
        "requirements.txt",
        "config.py",
        "metrics.py",
        "retrieval.py",
        "time_utils.py",
        "TimelyQABench_datasets.py",
    }
    if manifest["revision"] != REVISION or set(manifest["files"]) != expected_names:
        raise DatasetError("unexpected reference revision/files")
    for name, spec in manifest["files"].items():
        path = directory / name
        if (
            not path.resolve().is_relative_to(directory.resolve())
            or path.stat().st_size != spec["bytes"]
            or sha256_file(path) != spec["sha256"]
        ):
            raise DatasetError("reviewed upstream source checksum/path mismatch")
    return manifest


def load_reference(directory: Path, manifest_path: Path, expected_sha256: str) -> ReferenceCode:
    """Execute unchanged reviewed helpers, isolating their absolute import names.

    Only the original numeric assignments from config.py and the original two free
    functions from retrieval.py are compiled. Pipeline/I/O setup, FlagEmbedding imports
    and the unused RetrieverBundle constructor are excluded without modifying files.
    """
    manifest = verify_reference(directory, manifest_path, expected_sha256)
    backup = {name: sys.modules.get(name) for name in MODULE_NAMES}
    loaded = {}
    try:
        config = ModuleType("config")
        tree = ast.parse((directory / "config.py").read_text())
        constants = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and all(isinstance(t, ast.Name) and t.id in CONFIG_NAMES for t in node.targets)
        ]
        if {t.id for node in constants for t in node.targets} != CONFIG_NAMES:
            raise DatasetError("reviewed numeric config assignments missing")
        exec(
            compile(
                ast.Module(body=constants, type_ignores=[]), str(directory / "config.py"), "exec"
            ),
            config.__dict__,
        )
        sys.modules["config"] = config
        loaded["config"] = config
        for name in MODULE_NAMES[1:]:
            spec = importlib.util.spec_from_file_location(name, directory / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        namespace = {
            "np": np,
            "tqdm": lambda seq, **kwargs: seq,
            **{n: getattr(config, n) for n in ["K", "STAGE1_TOPN", "ALPHA_GRID"]},
        }
        for name in ["time_utils", "time_scores", "alpha", "metrics"]:
            namespace.update(
                {k: v for k, v in loaded[name].__dict__.items() if not k.startswith("_")}
            )
        # Both function bodies are unchanged AST nodes from the pinned source, for parity only.
        functions = [
            node
            for node in ast.parse((directory / "retrieval.py").read_text()).body
            if isinstance(node, ast.FunctionDef)
        ]
        if {node.name for node in functions} != {"rerank_with_alpha_grid", "evaluate_2stage"}:
            raise DatasetError("reviewed upstream retrieval functions differ")
        exec(
            compile(
                ast.Module(body=functions, type_ignores=[]), str(directory / "retrieval.py"), "exec"
            ),
            namespace,
        )
    finally:
        for name, module in backup.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return ReferenceCode(
        config,
        loaded["time_utils"],
        loaded["time_scores"],
        loaded["alpha"],
        loaded["metrics"],
        loaded["TimelyQABench_datasets"],
        namespace,
        manifest,
    )


def native_top_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if (
        scores.ndim != 1
        or not len(scores)
        or not np.isfinite(scores).all()
        or isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 1
    ):
        raise ValueError("invalid finite reference score vector/cutoff")
    return np.argsort(scores)[::-1][:top_k]


def automatic_rerank(
    reference: ReferenceCode,
    scores: np.ndarray,
    document_ids: list[int],
    document_insert_times: np.ndarray,
    document_event_times: np.ndarray,
    query: str,
    query_insert_time: str | None,
) -> tuple[list[Hit], dict]:
    """Unchanged native extraction/fallback; no gold or outcomes accepted."""
    return _automatic_rerank(
        reference,
        scores,
        document_ids,
        document_insert_times,
        document_event_times,
        query,
        query_insert_time,
    )


def automatic_rerank_with_qet(
    reference: ReferenceCode,
    scores: np.ndarray,
    document_ids: list[int],
    document_insert_times: np.ndarray,
    document_event_times: np.ndarray,
    query: str,
    query_insert_time: str | None,
    *,
    query_event_time: datetime,
) -> tuple[list[Hit], dict]:
    """Override QET only; original text/granularity helpers and coefficients unchanged."""
    if not isinstance(query_event_time, datetime) or query_event_time.tzinfo is not None:
        raise ValueError("QET override requires an explicit naive datetime")
    hits, info = _automatic_rerank(
        reference,
        scores,
        document_ids,
        document_insert_times,
        document_event_times,
        query,
        query_insert_time,
        qet_override=query_event_time,
    )
    return hits, {**info, "query_event_time_overridden": True}


def _automatic_rerank(
    reference: ReferenceCode,
    scores: np.ndarray,
    document_ids: list[int],
    document_insert_times: np.ndarray,
    document_event_times: np.ndarray,
    query: str,
    query_insert_time: str | None,
    *,
    qet_override: datetime | None = None,
) -> tuple[list[Hit], dict]:
    """Shared native kernel; explicit override does not replace other time inputs."""
    if (
        len(document_ids) != len(set(document_ids))
        or len(scores) != len(document_ids)
        or len(document_insert_times) != len(scores)
        or len(document_event_times) != len(scores)
    ):
        raise ValueError("unaligned corpus scores/IDs/timestamps")
    utils, time_scores = reference.utils, reference.scores
    candidates = native_top_indices(scores, reference.config.STAGE1_TOPN)
    qit = utils.to_dt(query_insert_time)
    qet, extracted_granularity = utils.extract_qet_from_query(query, qit_hint=qit)
    if qet_override is not None:
        qet = qet_override
    qg = utils.infer_gran_from_query_text(query)
    dg = utils.infer_gran_from_docs(
        document_insert_times, document_event_times, candidates, threshold=0.7
    )
    granularity = utils.stabilize_gran(utils.combine_granularity(qg, dg), qg, dg)
    delta = time_scores.compute_single_delta(
        qit, qet, document_insert_times, document_event_times, candidates, granularity
    )
    normalized = time_scores.minmax_normalize(delta)
    affinity = time_scores.time_affinity_from_delta(normalized, gamma=2.0)
    penalty = time_scores.build_delta_penalty(normalized, candidates)
    gap = utils.compute_gap_avg(
        candidates, document_event_times, document_insert_times, granularity
    )
    raw = scores[candidates].astype(float)
    base = (raw - raw.min()) / (np.ptp(raw) + 1e-9)
    # These availability terms intentionally include upstream's insertion fallbacks.
    known_insert = np.array([document_insert_times[i] is not None for i in candidates])
    known_event = np.array([document_event_times[i] is not None for i in candidates])
    known_event_or_insert = np.array(
        [
            document_event_times[i] is not None or document_insert_times[i] is not None
            for i in candidates
        ]
    )
    ratios = {
        "QET_DET": float(np.mean(known_event_or_insert & (qet is not None or qit is not None))),
        "QIT_DIT": float(np.mean(known_insert & (qit is not None))),
        "QIT_DET": float(np.mean(known_event & (qit is not None))),
        "QET_DIT": float(np.mean(known_insert & (qet is not None))),
    }
    signals = {
        "qet_exists": float(qet is not None),
        "gran_id": float({"year": 0, "month": 1, "day": 2, "hour": 3}.get(granularity, 1)),
        "gap_avg": float(gap),
    }
    gate = reference.alpha.compute_time_gate(signals, ratios, penalty)
    margin = time_scores.per_candidate_margin_weights(base, power=1.0)
    alpha, _ = reference.alpha.choose_alpha_from_signals(
        qet_exists=signals["qet_exists"],
        gran_id=signals["gran_id"],
        gap_avg=signals["gap_avg"],
        valid_ratio_terms=ratios,
        base_scores_norm=base,
        delta_pen=penalty,
        mix_finite_ratio=float(np.mean(np.isfinite(normalized))),
        alpha_grid=reference.config.ALPHA_GRID,
    )
    mixing = np.where(np.isfinite(affinity) & (affinity > 0), (1 - alpha) * gate * margin, 0.0)
    final = (1 - mixing) * base + mixing * affinity
    order = native_top_indices(final, len(candidates))
    hits = [Hit(document_ids[candidates[i]], float(final[i])) for i in order]
    return hits, {
        "alpha": float(alpha),
        "gate": float(gate),
        "query_event_time_extracted": qet is not None,
        "query_event_time_derived": qet.isoformat() if qet is not None else None,
        "extracted_granularity": extracted_granularity,
        "scoring_granularity": granularity,
        "unknown_temporal_distance_count": int((~np.isfinite(delta)).sum()),
        "later_or_equal_insert_candidate_count": sum(
            qit is not None
            and document_insert_times[i] is not None
            and document_insert_times[i] >= qit
            for i in candidates
        ),
        "hard_filter_applied": False,
        "gold_used_for_alpha": False,
    }
