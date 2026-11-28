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

Train500 collection logs remain separate from the immutable source collection:
fast `f33af6d892db41d098858d92b0c010a0`, deep `0498406345ce4eb48db41c563688f4b7`.
The collector used clean `30945ff`, and tracking IDs are stored in a local sidecar.

`temporalproof-router-cv --pilot-run <frozen-train500-directory> --output <new-directory>
--mlflow` logs a separate grouped-OOF run, **`a47d56cea741452d82e64501cf051aa0`**.
The `routing` extra pins sklearn 1.9.1; keep tracking/dense/routing extras together in the
shared GPU environment. Router logging validates seven explicit output hashes, checks
local artifact location and uploads them plus manifest under `router/`. It is idempotent
by manifest fingerprint. `analysis/` contains the checked aggregate/PNG/SVG/provenance.
Model parameters/OOF scores are local research artifacts, never public benchmark content.
Primary metrics describe grouped OOF training and modeled stage sums, not final test,
calibrated probabilities or actual service p95. See [report](m4_router_groupcv_report.md).


Frozen development validation uses the separate Python module and analysis/plot/log
commands in the [validation report](m4_router_validation_report.md). FINISHED runs:
fast `536e5d8412a84faeb8faf98ac84b78d1`, deep `fa31b63560844d8581b7c68b7bae53eb`,
frozen router `d3258e57ad3a43709832aa94b3a1c499`. The router run has explicit metadata
artifacts under `frozen_router/` and checked aggregate/PNG/SVG under `analysis/`, with
no model parameters or raw failure text. Repeated logging returns the same three IDs;
the immutable collection manifest excludes its later tracking sidecar. Metrics remain
**development validation / modeled cost**, with no refit or final test. Primary nDCG@10
0.190746 regresses against hybrid 0.198972; preserve this negative result.


M4.2 cached failure analysis/rank blending logs to FINISHED run
**`93c50c8dd83842caba35022102083b74`** through `experiments/log_rank_blend.py`.
See the [report](m4_rankblend_report.md) for original/replay/aggregate/plot/log commands.
`rankblend/` contains seven explicit metadata files; `analysis/` contains the aggregate
and PNG/SVG. Raw inspection/query/passage text, qrels, candidate scores and model parameters
are excluded. A repeated log returned the same ID. The v2 figure only clarifies tick
labels; both figure versions preserve the same measurements. Primary is grouped
train-fold-selected all-query blend, not the best validation transfer policy. Source
stage costs omit unmeasured new overhead; no new GPU timing or final-test score is logged.


M4.3 native TimelyRAG automatic code comparison is FINISHED as
**`e4bcabb2988340e49c9d976d3b27d8b5`**, with six standard stage runs listed in the
[reference report](m4_timelyrag_reference_report.md). `experiments/log_timelyrag_reference.py`
validates full evidence hashes and the local artifact root. Five explicit metadata files
plus aggregate/PNG/SVG are logged; no full score matrices, source text, failure text or
weights. Repeated logging returns the same seven IDs; initial prototype runs remain
separate. Auto-stage `retrieval_ms.*` is **incremental temporal CPU work only**, not fast
retrieval or live end-to-end p95. Native ties/reference-prefix metrics and E5-backbone
substitution are explicit; no paper whole-benchmark or oracle score is claimed.
