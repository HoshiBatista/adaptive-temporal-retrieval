"""M3 integrity/ranking tests use synthetic CPU data and fake model inference."""

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from temporalproof.data.source import ARCHIVE_SHA256, sha256_file
from temporalproof.data.splits import SplitManifest
from temporalproof.data.timelyqa import DatasetError, Document, QueryRecord
from temporalproof.evaluation.baseline import BaselineConfig, run_baseline
from temporalproof.evaluation.reranking import RerankingConfig, load_candidate_pool, run_reranking
from temporalproof.retrieval.bm25 import Hit
from temporalproof.retrieval.reranker import (
    MODEL_ID,
    REVISION,
    fetch_model,
    rank_scores,
    verify_model,
)


@pytest.fixture
def evidence(tmp_path):
    records = [
        QueryRecord(
            index=q,
            category="synthetic",
            query_ko="same synthetic query" if q < 2 else "TEST",
            answer="synthetic",
            docs_ko=[
                Document(global_doc_id=q * 100 + i, text=f"independent synthetic passage {q}:{i}")
                for i in range(100)
            ],
            gold_docs_final=[0],
            gold_docs_final_global=[q * 100],
        )
        for q in range(3)
    ]
    split = SplitManifest(
        dataset_sha256=ARCHIVE_SHA256,
        seed=1729,
        groups={"train": [], "validation": [[0, 1]], "test": [[2]]},
        audit={},
    )
    baseline = BaselineConfig.model_validate(
        {
            **json.loads(Path("configs/experiments/bm25_company_policies.json").read_text()),
            "method": "hybrid_rrf",
            "query_limit": 2,
            "bootstrap_samples": 100,
            "warmup_queries": 1,
        }
    )
    source = tmp_path / "source"
    run_baseline(
        records,
        split,
        baseline,
        source,
        {
            "archive_sha256": ARCHIVE_SHA256,
            "split_sha256": "b" * 64,
            "source_snapshot_sha256": "c" * 64,
        },
        query_search=lambda q, *_: [Hit(q * 100 + i, float(100 - i)) for i in range(100)],
    )
    config = RerankingConfig(
        baseline=baseline.model_copy(update={"method": "fixed_cross_encoder"}),
        model_id=MODEL_ID,
        revision=REVISION,
        batch_size=16,
        max_length=512,
        candidate_manifest_sha256=sha256_file(source / "manifest.json"),
    )
    return records, split, config, source


def load(evidence):
    records, split, config, source = evidence
    return load_candidate_pool(source, records, split, config, "b" * 64)


def repin(evidence, filename, content):
    _, _, config, source = evidence
    (source / filename).write_text(content)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["output_sha256"][filename] = sha256_file(source / filename)
    (source / "manifest.json").write_text(json.dumps(manifest))
    config.candidate_manifest_sha256 = sha256_file(source / "manifest.json")


def test_id_join_handles_duplicate_queries_preserves_coverage_and_never_scores_test(
    evidence, tmp_path
):
    records, split, config, source = evidence
    pool = load(evidence)

    class FakeScorer:
        calls = []

        def score(self, query, passages):
            assert query == "same synthetic query" and len(passages) == 100
            self.calls.append(passages)
            return np.arange(100, dtype=float)

        def profile(self):
            return {"synthetic": True}

    scorer = FakeScorer()
    result = run_reranking(records, split, config, pool, scorer, tmp_path / "out", {})
    assert len(scorer.calls) == 3  # one warmup, two selected queries; test not scored
    assert result["paired_deltas_vs_hybrid"]["recall_at_100"]["mean_delta"] == 0
    assert result["paired_deltas_vs_hybrid"]["ndcg_at_10"]["regressed"] == 2
    assert result["router_training_targets_generated"] is False
    assert (tmp_path / "out/run/qrels.tsv").read_bytes() == (source / "qrels.tsv").read_bytes()
    ranked = (tmp_path / "out/run/run.trec").read_text().splitlines()
    for q in [0, 1]:
        rows = [line.split() for line in ranked if line.split()[0] == str(q)]
        assert {int(row[2]) for row in rows} == set(range(q * 100, q * 100 + 100))
        assert int(rows[0][2]) == q * 100 + 99
    with pytest.raises(FileExistsError):
        run_reranking(records, split, config, pool, scorer, tmp_path / "out", {})


@pytest.mark.parametrize("filename", ["run.trec", "qrels.tsv", "per_query.jsonl", "manifest.json"])
def test_source_tampering_fails_before_scoring(evidence, filename):
    source = evidence[3]
    with (source / filename).open("a") as stream:
        stream.write("tampered")
    with pytest.raises(DatasetError, match="checksum"):
        load(evidence)


@pytest.mark.parametrize("mutation", ["duplicate", "unknown", "rank", "nan", "short"])
def test_structurally_bad_pool_rejected_even_if_rehashed(evidence, mutation):
    lines = (evidence[3] / "run.trec").read_text().splitlines()
    columns = lines[1].split()
    if mutation == "duplicate":
        columns[2] = lines[0].split()[2]
    elif mutation == "unknown":
        columns[2] = "9999"
    elif mutation == "rank":
        columns[3] = "99"
    elif mutation == "nan":
        columns[4] = "nan"
    if mutation == "short":
        lines.pop()
    else:
        lines[1] = " ".join(columns)
    repin(evidence, "run.trec", "\n".join(lines) + "\n")
    with pytest.raises(DatasetError):
        load(evidence)


def test_split_and_selection_mismatch_rejected(evidence):
    records, split, config, source = evidence
    with pytest.raises(DatasetError, match="provenance"):
        load_candidate_pool(source, records, split, config, "a" * 64)
    config.baseline = config.baseline.model_copy(update={"seed": 2})
    with pytest.raises(DatasetError, match="config differs"):
        load(evidence)


def test_rehashed_changed_qrels_rejected(evidence):
    repin(evidence, "qrels.tsv", "0\t0\t99\t1\n")
    with pytest.raises(DatasetError, match="judgments"):
        load(evidence)


def test_scores_have_stable_ties_and_reject_missing_nonfinite_or_duplicate_values():
    assert [hit.doc_id for hit in rank_scores([20, 10, 30], [2, 2, 0])] == [10, 20, 30]
    for ids, scores in [
        ([1, 1], [0, 1]),
        ([1, 2], [1]),
        ([1], [np.nan]),
        ([1], [np.inf]),
        ([1, 2], [[1], [2]]),
    ]:
        with pytest.raises(ValueError):
            rank_scores(ids, scores)


def test_checked_downloads_are_explicit_and_do_not_overwrite_corruption(tmp_path, monkeypatch):
    from temporalproof.retrieval import reranker

    payload = b"independent synthetic weights"
    monkeypatch.setattr(
        reranker,
        "ASSETS",
        {"model.safetensors": (len(payload), hashlib.sha256(payload).hexdigest())},
    )
    with pytest.raises(DatasetError):
        verify_model(tmp_path)
    monkeypatch.setattr(reranker, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    fetch_model(tmp_path)
    monkeypatch.setattr(reranker, "urlopen", lambda *a, **kw: pytest.fail("unexpected network"))
    fetch_model(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(DatasetError, match="not replaced"):
        fetch_model(tmp_path)


def test_model_asset_manifest_matches_code():
    from temporalproof.retrieval.reranker import ASSETS

    manifest = json.loads(Path("configs/models/mmarco_minilm.json").read_text())
    assert manifest["model_id"] == MODEL_ID and manifest["revision"] == REVISION
    assert manifest["assets"] == {name: list(asset) for name, asset in ASSETS.items()}
