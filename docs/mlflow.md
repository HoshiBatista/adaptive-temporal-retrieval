# Local MLflow tracking

The optional `tracking` extra pins **MLflow 3.18.0**. Research runs retain their JSON/TREC
files; MLflow adds a local comparison UI and a copy of the evidence. No remote service or
account is involved. The common run-file format supports M1 BM25 and M2 comparisons.

## Install and open

From the repository root:

```bash
uv sync --frozen --extra tracking --group dev
uv run --frozen --extra tracking temporalproof-mlflow server
```

Open <http://127.0.0.1:5000>. The server runs in the foreground; stop it with Ctrl+C.
After a WSL restart, run the server command again. If another application uses the port,
choose `temporalproof-mlflow server --port 5001` and open that port instead.

The wrapper binds only to `127.0.0.1`, uses one server worker, and explicitly selects
the project's SQLite database and local artifact paths. It disables MLflow SDK/UI usage
telemetry before importing/starting MLflow. Remote tracking environment variables do not
redirect the wrapper's explicitly constructed local clients.

## Storage

- Metadata: `artifacts/mlflow/mlflow.db`.
- Artifact copies: `artifacts/mlflow/artifacts/temporalproof-company-policies/`.
- Original research output: the chosen `artifacts/bm25/<run>/` directory.

All these paths are ignored by Git. Store paths are resolved against the installed
editable project's location, independent of the shell's current directory. Artifact
locations in SQLite are absolute file URIs; do not assume a copied database is portable
to a different checkout path. Original file manifests remain available for reimport into
a new store. Do not delete `artifacts/mlflow/` when cleaning benchmark scratch outputs.

Logging writes directly to local SQLite/files, so it works while the UI server is stopped.
The UI reads the same database. This is a single-machine setup; it does not configure
network sharing or cloud storage. MLflow creates/manages its own database schema; no
TemporalProof application database/migrations are introduced.

## Log a new BM25 run

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --extra tracking temporalproof-bm25 --mlflow \
  --output artifacts/bm25/my_tracked_validation_run
```

Use a new output directory each time. Without `--mlflow`, BM25 does not load MLflow or
write a tracking record. Include `--extra tracking` in `uv run` commands while using
this environment's tracking tools: uv reconciles optional dependencies with the extras
requested by each command. Core audit/BM25/tests still support a base installation.

Logging happens **after** evaluation and file output, outside measured retrieval timing.
The experiment is `temporalproof-company-policies`. Logged items:

- Config parameters, corpus/query counts and original package versions.
- nDCG/Recall means and conditional query-bootstrap interval bounds.
- Index-build time and sequential retrieval mean/p50/p95, with timing scope.
- Dataset/split/source hashes, original Git commit and dirty-state tags.
- Config, manifest, summary, per-query metrics, TREC ranking and qrels under `evidence/`.
  Raw failure-example text is not copied.

M2 uses `temporalproof-compare --mlflow` with `--extra tracking --extra dense`. Each method
has a distinct `method` parameter/run name and the manifest artifact retains encoder,
cache, fusion and insertion-policy provenance. Compare only matching split/query IDs.

If explicit tracking fails, the command returns nonzero and states that BM25 file output
is complete. Fix the local tracking issue and import the existing directory; rerunning
retrieval is unnecessary. No automatic logging or external model/LLM service is enabled.

## Import an existing measurement

```bash
uv run --frozen --extra tracking temporalproof-mlflow import-run \
  --run-dir artifacts/bm25/company_policies_validation100_v1 \
  --measured-on 2026-11-26
```

The importer requires complete original run files and verifies their recorded SHA-256
values before writing. It preserves all source results and their original provenance.
The MLflow run is tagged `historical_import`, with the supplied original measurement date.
MLflow's start/end clock timestamps describe **import time**; no original clock time is
fabricated. Repeating a successful import returns the same run ID. A conflicting date
for an already imported fingerprint is rejected. Incomplete/failed exports can be retried.

## Verified setup, 2026-11-26

- Imported M1 validation100: `2bc4266fc7564093a8915c585d1155fd`, nDCG@10
  0.17595877026340895, preserving the original source snapshot/dirty state.
- A separate **three-query train smoke run** exercised `temporalproof-bm25 --mlflow`:
  `3438643655224623a76a0247c9c70b21`. It is an integration check, not a new benchmark.
- Both records are FINISHED and expose six evidence artifacts each. Root UI, `/health`,
  HTTP experiment/run listing and browser artifact download returned successful responses.
- SQLite tests cover local settings despite remote environment variables, import
  idempotence, provenance/metric preservation, checksum and symlink rejection, nonfinite
  timing rejection, missing optional dependency and preservation of completed BM25 files.
- MLflow's dependencies emit SQLAlchemy/Starlette deprecation warnings; observed tracking,
  UI and artifact checks succeeded. No dependency source was patched.

Filter by `params.partition = 'validation'` before comparing validation runs; the train
smoke is not comparable with the M1 validation result. Compare future dense/hybrid runs
only on the same corpus, split, selected query IDs and metric protocol.

## Sources

- [MLflow server and local storage configuration](https://mlflow.org/docs/latest/self-hosting/architecture/tracking-server/)
- [MLflow usage-telemetry opt-out](https://mlflow.org/docs/latest/community/usage-tracking/)

M3 supports `temporalproof-rerank --mlflow` with the same extras and evidence format.
Its `retrieval_ms.*` metrics mean **incremental reranking only**; inspect the timing-scope
tag/summary before comparing latency. Model assets/settings, candidate source hashes and
CPU/GPU profile are in `evidence/manifest.json`. The initial negative MiniLM result is
FINISHED as `1419b5ef303b4611a38e40a9f61cdfa9`; file-only replay is not another query sample.

Korean M3 follow-up: `temporalproof-rerank --config
configs/experiments/m3_korean_company_policies.json --mlflow --output <new-directory>`.
Run `144b746d2199493fba60acfa5c130f05` is FINISHED, nDCG@10 0.163721 and incremental
mean 1,148.100 ms. It uses the same candidate pool as MiniLM; six evidence files and
exact-score replay were verified. The default config remains the original MiniLM one.
Both negative comparisons against hybrid remain visible.

M4 pilot: `temporalproof-pilot --mlflow --output <new-directory>` logs completed standard
fast/deep runs tagged `M4_train_only_pilot`. Initial fast ID is
`66198b1aee61450a804c7b33e6363776`; deep is `001ba53181f041c49df1ca9c88ca8b63`.
Deep `retrieval_ms.*` remains incremental reranking only; fast includes feature extraction.
`pilot/` stores an explicit allowlist of config/plan/checkpoint/features/outcomes/summary/
manifest; local raw failure text is excluded. `pilot/analysis` includes the checked
aggregate and static PNG/SVG quality–cost figure with provenance. The curve uses modeled
stage costs and an explicitly labeled training oracle bound, not a fitted/live router.
