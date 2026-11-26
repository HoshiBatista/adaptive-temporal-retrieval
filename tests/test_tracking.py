import json
import os
from datetime import date
from pathlib import Path

import pytest

from temporalproof.data.source import sha256_file
from temporalproof.data.splits import SplitManifest
from temporalproof.data.timelyqa import load_records
from temporalproof.evaluation.baseline import BaselineConfig, run_baseline
from temporalproof.tracking.local import (
    EXPERIMENT,
    RUN_FILES,
    TrackingError,
    local_client,
    log_run_directory,
    server_command,
    validate_run,
)

ROOT = Path(__file__).parents[1]


@pytest.fixture
def saved_run(tmp_path):
    records = load_records(ROOT / "tests/fixtures/timelyqa_synthetic.jsonl")
    config = BaselineConfig.model_validate(
        {
            **json.loads((ROOT / "configs/experiments/bm25_company_policies.json").read_text()),
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
    provenance = {
        "source_snapshot_sha256": "a" * 64,
        "git_commit": "old-source-commit",
        "git_dirty": True,
        "archive_sha256": "b" * 64,
        "split_sha256": "c" * 64,
    }
    directory = tmp_path / "run"
    run_baseline(records, split, config, directory, provenance)
    return directory


def test_server_uses_loopback_and_explicit_absolute_storage(tmp_path):
    command = server_command(tmp_path, 5012)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert command[command.index("--port") + 1] == "5012"
    assert str(tmp_path / "mlflow.db") in command[command.index("--backend-store-uri") + 1]
    assert "--no-serve-artifacts" in command


def test_corrupt_evidence_is_rejected_before_database_creation(saved_run, tmp_path):
    (saved_run / "run.trec").write_text("changed result", encoding="utf-8")
    store = tmp_path / "store"
    with pytest.raises(TrackingError, match="checksum"):
        log_run_directory(saved_run, store=store)
    assert not store.exists()


def test_symlinked_artifact_outside_run_is_rejected(saved_run, tmp_path):
    outside = tmp_path / "outside"
    target = saved_run / "qrels.tsv"
    outside.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(outside)
    with pytest.raises(TrackingError, match="outside"):
        validate_run(saved_run)


def test_nonfinite_summary_is_rejected(saved_run):
    summary_path = saved_run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["retrieval_timing_ms"]["mean"] = float("nan")
    summary_path.write_text(json.dumps(summary))
    manifest_path = saved_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"]["summary.json"] = sha256_file(summary_path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(TrackingError, match="validate"):
        validate_run(saved_run)


def test_real_local_store_logs_once_preserves_evidence_and_disables_remote_defaults(
    saved_run, tmp_path, monkeypatch
):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.invalid")
    monkeypatch.setenv("MLFLOW_REGISTRY_URI", "https://example.invalid")
    pytest.importorskip("mlflow")
    store = tmp_path / "store"
    before = {path.name: path.read_bytes() for path in saved_run.iterdir()}
    run_id = log_run_directory(saved_run, store=store, measured_on=date(2026, 11, 26))
    assert log_run_directory(saved_run, store=store, measured_on=date(2026, 11, 26)) == run_id
    client = local_client(store)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    assert len(client.search_runs([experiment.experiment_id])) == 1
    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["temporalproof.source_git_commit"] == "old-source-commit"
    assert run.data.tags["temporalproof.source_git_dirty"] == "true"
    assert run.data.tags["temporalproof.record_type"] == "historical_import"
    assert run.data.tags["temporalproof.measured_on"] == "2026-11-26"
    assert (
        run.data.metrics["ndcg_at_10"]
        == json.loads(before["summary.json"])["metrics"]["ndcg_at_10"]["mean"]
    )
    assert {
        Path(item.path).name for item in client.list_artifacts(run_id, "evidence")
    } == RUN_FILES | {"manifest.json"}
    assert {path.name: path.read_bytes() for path in saved_run.iterdir()} == before
    assert os.environ["MLFLOW_DISABLE_TELEMETRY"] == "true"
    assert client.tracking_uri.startswith("sqlite:////")
    with pytest.raises(TrackingError, match="different historical"):
        log_run_directory(saved_run, store=store, measured_on=date(2025, 1, 1))


@pytest.mark.parametrize("failure", [TrackingError, OSError])
def test_cli_keeps_completed_bm25_results_when_explicit_tracking_fails(
    tmp_path, monkeypatch, capsys, failure
):
    import temporalproof.evaluation.baseline as baseline

    records = load_records(ROOT / "tests/fixtures/timelyqa_synthetic.jsonl")
    split = SplitManifest(
        dataset_sha256="a" * 64,
        seed=1729,
        groups={"train": [], "validation": [[0, 1]], "test": []},
        audit={},
    )
    split_path = tmp_path / "split.json"
    split_path.write_text(split.model_dump_json())
    config = json.loads((ROOT / "configs/experiments/bm25_company_policies.json").read_text())
    config.update(query_limit=2, bootstrap_samples=100)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    monkeypatch.setattr(baseline, "load_archive_records", lambda path: records)
    monkeypatch.setattr(baseline, "verify_split", lambda *args: None)

    def fail(*args, **kwargs):
        raise failure("synthetic unavailable store")

    monkeypatch.setattr("temporalproof.tracking.local.log_run_directory", fail)
    output = tmp_path / "run"
    assert (
        baseline.main(
            [
                "--config",
                str(config_path),
                "--split",
                str(split_path),
                "--output",
                str(output),
                "--mlflow",
            ]
        )
        == 1
    )
    assert (output / "manifest.json").exists()
    assert "BM25 completed" in capsys.readouterr().err


def test_missing_optional_package_has_an_install_instruction(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def without_mlflow(name, *args, **kwargs):
        if name == "mlflow":
            raise ImportError("synthetic missing optional dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_mlflow)
    with pytest.raises(TrackingError, match="uv sync --frozen --extra tracking"):
        local_client(tmp_path / "unused-store")
    assert not (tmp_path / "unused-store").exists()
