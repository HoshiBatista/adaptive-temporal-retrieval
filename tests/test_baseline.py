import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from temporalproof.data.splits import SplitManifest
from temporalproof.data.timelyqa import load_records
from temporalproof.evaluation.baseline import BaselineConfig, evaluate_query, run_baseline
from temporalproof.evaluation.metrics import reference_prefix_ndcg_at_k
from temporalproof.retrieval.bm25 import Hit

FIXTURE = Path(__file__).parent / "fixtures" / "timelyqa_synthetic.jsonl"
CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / "bm25_company_policies.json"


def test_reference_formula_is_named_separately_and_differs_on_missing_gold():
    record = load_records(FIXTURE)[0]
    result = evaluate_query(record, [Hit(10, 1.0), Hit(99, 0.0)])
    assert result["ndcg_at_10"] == pytest.approx(0.6131471927654584)
    assert result["reference_prefix_ndcg_at_10"] == 1.0
    assert result["recall_at_100"] == 0.5
    assert reference_prefix_ndcg_at_k([], {10}, 10) == 0.0


def test_final_test_partition_is_not_available_to_m1_runner():
    config = json.loads(CONFIG.read_text())
    config["partition"] = "test"
    with pytest.raises(ValidationError):
        BaselineConfig.model_validate(config)


def test_end_to_end_cpu_exports_replayable_rankings_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "temporalproof.data.source.urlopen", lambda *a, **kw: pytest.fail("unexpected download")
    )
    records = load_records(FIXTURE)
    split = SplitManifest(
        dataset_sha256="a" * 64,
        seed=1729,
        groups={"train": [], "validation": [[0, 1]], "test": []},
        audit={},
    )
    config = BaselineConfig.model_validate(
        {
            **json.loads(CONFIG.read_text()),
            "query_limit": 2,
            "bootstrap_samples": 100,
        }
    )
    first, second = tmp_path / "first", tmp_path / "second"
    result = run_baseline(records, split, config, first, {"fixture": True})
    run_baseline(list(reversed(records)), split, config, second, {"fixture": True})
    assert result["query_count"] == 2
    for name in ["run.trec", "qrels.tsv", "config.json"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["test_partition_evaluated"] is False
    assert manifest["corpus_document_count"] == 3
    assert set(manifest["output_sha256"]) == {
        "config.json",
        "failure_cases.json",
        "per_query.jsonl",
        "qrels.tsv",
        "run.trec",
        "summary.json",
    }
    per_query = [json.loads(line) for line in (first / "per_query.jsonl").read_text().splitlines()]
    assert all("retrieval_ms" in row for row in per_query)
    assert len((first / "run.trec").read_text().splitlines()) == 6
    with pytest.raises(FileExistsError):
        run_baseline(records, split, config, first, {})
