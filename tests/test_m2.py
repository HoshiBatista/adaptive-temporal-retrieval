import io
import json
from pathlib import Path

import numpy as np
import pytest

from temporalproof.data.splits import SplitManifest
from temporalproof.data.timelyqa import DatasetError, Document, load_records
from temporalproof.evaluation.baseline import BaselineConfig, run_baseline
from temporalproof.evaluation.comparison import paired_comparisons
from temporalproof.retrieval.bm25 import Hit
from temporalproof.retrieval.dense import DenseIndex, corpus_cache, normalized_matrix
from temporalproof.retrieval.encoder import fetch_model, verify_model
from temporalproof.retrieval.fusion import (
    insertion_features,
    insertion_time_rerank,
    reciprocal_rank_fusion,
)


def test_rrf_rewards_agreement_and_deterministically_breaks_ties():
    result = reciprocal_rank_fusion([[Hit(10, 10), Hit(20, 9)], [Hit(20, 100), Hit(10, 1)]], 2)
    assert [hit.doc_id for hit in result] == [10, 20]
    assert result[0].score == pytest.approx(1 / 61 + 1 / 62)
    with pytest.raises(ValueError, match="duplicate"):
        reciprocal_rank_fusion([[Hit(10, 1), Hit(10, 0)]], 2)


@pytest.mark.parametrize("value", [None, "", "2022-!1-01 00:00:00"])
def test_unknown_times_remain_unknown(value):
    features = insertion_features("2023-01-01 00:00:00", value)
    assert features.known is False
    assert features.affinity is None
    assert features.inserted_after_query is None


def test_insertion_boundaries_and_symmetric_gap_do_not_assert_validity():
    anchor = "2023-01-01 00:00:00"
    equal = insertion_features(anchor, anchor)
    assert equal.signed_gap_days == 0
    assert equal.inserted_after_query is False
    assert equal.affinity == 1
    past = insertion_features(anchor, "2022-01-01 00:00:00")
    future = insertion_features(anchor, "2024-01-01 00:00:00")
    assert past.affinity == future.affinity == 0.5
    assert future.inserted_after_query is True
    assert past.inserted_after_query is False


def test_temporal_rerank_keeps_exact_candidate_set_and_unknown_base_order():
    hits = [Hit(10, 0.03), Hit(20, 0.029), Hit(30, 0.028)]
    original = [hit.doc_id for hit in hits]
    assert [hit.doc_id for hit in insertion_time_rerank(hits, None, {})] == original
    ranked = insertion_time_rerank(
        hits,
        "2023-01-01 00:00:00",
        {30: "2023-01-01 00:00:00"},
        weight=2,
    )
    assert ranked[0].doc_id == 30
    assert set(hit.doc_id for hit in ranked) == set(original)


def test_faiss_cosine_and_boundary_ties_are_stable():
    pytest.importorskip("faiss")
    index = DenseIndex(np.array([30, 10, 20]), np.array([[1, 0], [2, 0], [0, 1]]))
    assert [hit.doc_id for hit in index.search(np.array([1, 0]), 1)] == [10]
    assert [hit.doc_id for hit in index.search(np.array([1, 0]), 3)] == [10, 30, 20]
    with pytest.raises(ValueError, match="dimension"):
        index.search(np.array([1, 0, 0]), 3)


@pytest.mark.parametrize("value", [np.array([[0.0, 0.0]]), np.array([[np.nan, 1.0]])])
def test_bad_embedding_values_fail(value):
    with pytest.raises(ValueError):
        normalized_matrix(value)


def test_cache_reuses_exact_input_and_rejects_modified_files(tmp_path):
    pytest.importorskip("faiss")

    class FakeEncoder:
        inputs_at_max_length = 0
        calls = 0

        def profile(self):
            return {"model_id": "synthetic", "dimension": 2}

        def encode(self, texts, *, kind):
            assert kind == "passage"
            self.calls += 1
            return np.array([[1, 0], [0, 1]], dtype=np.float32)

    encoder = FakeEncoder()
    docs = [Document(global_doc_id=i, text=f"synthetic passage {i}") for i in range(2)]
    first, manifest = corpus_cache(docs, encoder, tmp_path)
    _, second = corpus_cache(list(reversed(docs)), encoder, tmp_path)
    assert encoder.calls == 1 and second["cache_hit"] is True
    assert first.index.ntotal == 2
    target = tmp_path / manifest["cache_key"] / "ids.npy"
    target.write_bytes(b"corrupted")
    with pytest.raises(DatasetError, match="checksum"):
        corpus_cache(docs, encoder, tmp_path)


def test_model_download_is_explicit_checked_and_not_replaced(tmp_path, monkeypatch):
    import hashlib

    from temporalproof.retrieval import encoder

    content = b"synthetic model asset"
    monkeypatch.setattr(
        encoder, "ASSETS", {"asset": (len(content), hashlib.sha256(content).hexdigest())}
    )
    monkeypatch.setattr(encoder, "urlopen", lambda *a, **kw: io.BytesIO(content))
    fetch_model(tmp_path)
    verify_model(tmp_path)
    monkeypatch.setattr(encoder, "urlopen", lambda *a, **kw: pytest.fail("unexpected network"))
    fetch_model(tmp_path)
    (tmp_path / "asset").write_bytes(b"modified")
    with pytest.raises(DatasetError):
        fetch_model(tmp_path)


def test_paired_analysis_joins_by_id_and_records_regressions():
    def row(query_id, score):
        return {
            "query_id": query_id,
            **{
                name: score
                for name in ["ndcg_at_10", "recall_at_10", "recall_at_20", "recall_at_100"]
            },
        }

    first = [row(0, 0), row(1, 1)]
    other = [row(1, 0.5), row(0, 1)]
    result = paired_comparisons(first, other, 1729, 100)["ndcg_at_10"]
    assert result["mean_delta"] == 0.25
    assert result["improved"] == result["regressed"] == 1
    with pytest.raises(ValueError, match="matching"):
        paired_comparisons(first, other[:1], 1729, 100)


def test_common_evaluator_accepts_label_free_search_and_saves_method(tmp_path):
    root = Path(__file__).parents[1]
    records = load_records(root / "tests/fixtures/timelyqa_synthetic.jsonl")
    config = BaselineConfig.model_validate(
        {
            **json.loads((root / "configs/experiments/bm25_company_policies.json").read_text()),
            "method": "dense",
            "query_limit": 2,
            "bootstrap_samples": 100,
        }
    )
    split = SplitManifest(
        dataset_sha256="a" * 64,
        seed=1729,
        groups={"train": [], "validation": [[0, 1]], "test": []},
        audit={},
    )

    def search(text, cutoff, inserted):
        assert isinstance(text, str) and (inserted is None or isinstance(inserted, str))
        return [Hit(10, 1), Hit(11, 0.9), Hit(12, 0.8)]

    run_baseline(records, split, config, tmp_path / "run", {}, search=search)
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["config"]["method"] == "dense"
    assert all(
        line.endswith("dense") for line in (tmp_path / "run/run.trec").read_text().splitlines()
    )
