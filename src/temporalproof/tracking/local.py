"""Optional local SQLite tracking and a loopback-only MLflow UI."""

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from temporalproof.data.source import sha256_file

REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_STORE = REPOSITORY / "artifacts" / "mlflow"
EXPERIMENT = "temporalproof-company-policies"
RUN_FILES = {"config.json", "summary.json", "per_query.jsonl", "run.trec", "qrels.tsv"}
VERIFIED_FILES = RUN_FILES | {"failure_cases.json"}


class TrackingError(ValueError):
    """Actionable tracking failure without source content or credentials."""


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    config: dict[str, Any]
    output_sha256: dict[str, str]
    selected_query_ids: list[int]
    corpus_document_count: int = Field(ge=1)
    source_snapshot_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    git_commit: str | None
    git_dirty: bool | None
    archive_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    split_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class MetricSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    mean: float
    query_bootstrap_95_percent_interval: tuple[float, float]


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    query_count: int = Field(ge=1)
    metrics: dict[str, MetricSummary]
    index_build_ms: float = Field(ge=0)
    retrieval_timing_ms: "TimingSummary"


class TimingSummary(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    mean: float = Field(ge=0)
    p50: float = Field(ge=0)
    p95: float = Field(ge=0)
    scope: str


def local_client(store: Path = DEFAULT_STORE):
    """Explicit SQLite URI ignores remote MLFLOW_TRACKING_URI settings."""
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    try:
        from mlflow import MlflowClient
    except ImportError:
        raise TrackingError(
            "install tracking with: uv sync --frozen --extra tracking --group dev"
        ) from None
    store = store.resolve()
    store.mkdir(parents=True, exist_ok=True)
    uri = "sqlite:///" + (store / "mlflow.db").as_posix()
    try:
        return MlflowClient(tracking_uri=uri, registry_uri=uri)
    except Exception as exc:
        raise TrackingError("cannot initialize local MLflow SQLite store") from exc


def validate_run(directory: Path) -> tuple[RunManifest, RunSummary, str]:
    """Check immutable file evidence before creating a database entry."""
    try:
        manifest = RunManifest.model_validate_json((directory / "manifest.json").read_bytes())
        if set(manifest.output_sha256) != VERIFIED_FILES:
            raise TrackingError("unexpected run artifact list")
        for name, expected in manifest.output_sha256.items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()):
                raise TrackingError("run artifact points outside the run directory")
            if sha256_file(path) != expected:
                raise TrackingError("run artifact checksum mismatch; source files were not changed")
        summary = RunSummary.model_validate_json((directory / "summary.json").read_bytes())
        if summary.query_count != len(manifest.selected_query_ids):
            raise TrackingError("query count differs from run manifest")
        fingerprint = sha256_file(directory / "manifest.json")
    except (OSError, ValidationError) as exc:
        raise TrackingError("cannot read or validate the saved run manifest and artifacts") from exc
    return manifest, summary, fingerprint


def log_run_directory(
    directory: Path, *, store: Path = DEFAULT_STORE, measured_on: date | None = None
) -> str:
    """Import a complete run once; historical measurements keep their original evidence.

    MLflow start/end timestamps describe this logging operation. `measured_on` is
    explicitly a historical measurement date, not a fabricated original clock time.
    """
    manifest, summary, fingerprint = validate_run(directory)
    client = local_client(store)
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        artifact_root = (store.resolve() / "artifacts" / EXPERIMENT).as_uri()
        experiment_id = client.create_experiment(EXPERIMENT, artifact_location=artifact_root)
    else:
        experiment_id = experiment.experiment_id
        expected_root = (store.resolve() / "artifacts" / EXPERIMENT).as_uri()
        if experiment.artifact_location != expected_root:
            raise TrackingError("existing experiment uses a different artifact location")
    previous = client.search_runs(
        [experiment_id],
        filter_string=(
            f"tags.`temporalproof.fingerprint` = '{fingerprint}' AND attributes.status = 'FINISHED'"
        ),
        max_results=1,
    )
    if previous:
        if (
            measured_on
            and previous[0].data.tags.get("temporalproof.measured_on") != measured_on.isoformat()
        ):
            raise TrackingError("existing record has a different historical measurement date")
        return previous[0].info.run_id
    tags = {
        "mlflow.runName": str(manifest.config.get("name", "bm25"))
        + (" [historical import]" if measured_on else ""),
        "temporalproof.fingerprint": fingerprint,
        "temporalproof.record_type": "historical_import" if measured_on else "completed_run",
        "temporalproof.measured_on": (measured_on or date.today()).isoformat(),
        "temporalproof.source_git_commit": manifest.git_commit or "unknown",
        "temporalproof.source_git_dirty": str(manifest.git_dirty).lower(),
        "temporalproof.source_snapshot_sha256": manifest.source_snapshot_sha256,
        "temporalproof.archive_sha256": manifest.archive_sha256,
        "temporalproof.split_sha256": manifest.split_sha256,
        "temporalproof.timing_scope": summary.retrieval_timing_ms.scope,
    }
    run_id = client.create_run(experiment_id, tags=tags).info.run_id
    try:
        for key, value in manifest.config.items():
            client.log_param(run_id, key, value)
        client.log_param(run_id, "corpus_document_count", manifest.corpus_document_count)
        client.log_param(run_id, "query_count", summary.query_count)
        for key, value in manifest.model_extra.get("packages", {}).items():
            client.log_param(run_id, f"package.{key}", value)
        for key, value in summary.metrics.items():
            client.log_metric(run_id, key, value.mean)
            lower, upper = value.query_bootstrap_95_percent_interval
            client.log_metric(run_id, f"{key}.bootstrap_lower", lower)
            client.log_metric(run_id, f"{key}.bootstrap_upper", upper)
        client.log_metric(run_id, "index_build_ms", summary.index_build_ms)
        for key in ["mean", "p50", "p95"]:
            client.log_metric(
                run_id, f"retrieval_ms.{key}", getattr(summary.retrieval_timing_ms, key)
            )
        for name in sorted(RUN_FILES | {"manifest.json"}):
            client.log_artifact(run_id, str(directory / name), artifact_path="evidence")
        client.set_terminated(run_id, status="FINISHED")
    except Exception as exc:
        client.set_terminated(run_id, status="FAILED")
        raise TrackingError(
            "MLflow logging failed; original file results remain available"
        ) from exc
    return run_id


def server_command(store: Path, port: int = 5000) -> list[str]:
    store = store.resolve()
    return [
        sys.executable,
        "-m",
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        "1",
        "--backend-store-uri",
        "sqlite:///" + (store / "mlflow.db").as_posix(),
        "--default-artifact-root",
        (store / "artifacts").as_uri(),
        "--no-serve-artifacts",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    server = actions.add_parser("server", help="start the local UI; stop with Ctrl+C")
    server.add_argument("--port", type=int, default=5000)
    importer = actions.add_parser(
        "import-run", help="log an existing, hash-verified historical run"
    )
    importer.add_argument("--run-dir", type=Path, required=True)
    importer.add_argument("--measured-on", type=date.fromisoformat, required=True)
    args = parser.parse_args(argv)
    if args.action == "server" and not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        if args.action == "server":
            local_client()
            return subprocess.call(server_command(DEFAULT_STORE, args.port), env=os.environ.copy())
        run_id = log_run_directory(args.run_dir, measured_on=args.measured_on)
    except (OSError, TrackingError) as exc:
        print(
            f"MLflow operation failed: {exc}",
            file=sys.stderr,
        )
        return 1
    print(f"Local MLflow run: {run_id}")
    print("Open http://127.0.0.1:5000 after starting temporalproof-mlflow server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
