# TemporalProof — verified project status and agent handoff

Last verified: **2026-11-26**. Only executed work is recorded below.

## Current state

- **M0:** reproducible Company Policies data audit, strict typed JSONL adapter, synthetic
  CPU fixture and metric/schema/audit tests implemented and run.
- **M1:** frozen grouped split and independent CPU BM25 implemented; 81 CPU tests pass.
  Two runs on the same 100 validation queries / full 30,000-passage corpus have identical
  rankings, qrels and quality metrics. nDCG@10 0.175959, Recall@10 0.230000, Recall@100
  0.393333. These are development-subset measurements; test is not evaluated.
- **Local tracking:** optional MLflow 3.18.0 SQLite/artifact store, loopback UI, idempotent
  historical import and BM25 `--mlflow` are implemented. Current suite: **89 tests pass**.
  M1 validation100 and a separate three-query train integration smoke are FINISHED in
  the local store. No new validation comparison or final-test evaluation was performed.
- **Environment:** `pyproject.toml`, `uv.lock`, Python >=3.11 declared; validation executed
  on Linux x86-64, CPython 3.13.11. Actual locked versions include Pydantic 2.14.4,
  pytest 9.2.0, Ruff 0.16.10, rank-bm25 0.2.2 and NumPy 2.5.5.
- **Research results:** BM25 validation-subset baseline only; no dense/hybrid/reranker/router
  comparison, final test evaluation or official whole-benchmark reproduction.
- **Application:** no API, database, UI, generation or ChangeImpact implementation.
- **Data:** pinned 5.8 MB archive in ignored `data/raw/`; derived reports in ignored
  `artifacts/`. No source benchmark content is committed; test data is independently authored.
- **Next milestone:** M2 compact dense/hybrid retrieval on the same fixed validation IDs.

## Verified work log

### 2026-11-26 — Local MLflow setup

- Goal: provide local experiment persistence/UI and integrate existing BM25 output after
  the user chose local hosting. No remote tracking service, model or GPU job introduced.
- Files changed: optional dependency/CLI in `pyproject.toml`, `uv.lock`, `.gitignore`,
  `src/temporalproof/tracking/`, BM25 CLI flag, `tests/test_tracking.py`, `docs/mlflow.md`,
  README, architecture decisions and this handoff; M1 report updated to link tracking.
- Commands executed:

  ```bash
  uv sync --extra tracking --group dev
  uv run --frozen --extra tracking ruff format .
  uv run --frozen --extra tracking ruff check .
  uv run --frozen --extra tracking pytest -q
  uv run --frozen --extra tracking temporalproof-mlflow import-run --run-dir artifacts/bm25/company_policies_validation100_v1 --measured-on 2026-11-26
  uv run --frozen --extra tracking temporalproof-mlflow server
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --frozen --extra tracking temporalproof-bm25 --mlflow --config artifacts/configs/bm25_mlflow_train_smoke_v1.json --output artifacts/bm25/mlflow_train_smoke_v1
  ```

- Validation: **89 tests passed**, Ruff checks passed. Initial unused-import/line-length
  findings corrected. One third-party SQLAlchemy deprecation warning remains; server
  startup also reports a third-party Starlette deprecation warning. Functional checks pass.
- Tests use tiny independent fixtures and temporary SQLite stores; no benchmark download,
  model inference or UI server is required. Missing tracking extra and failed tracking
  preserve core usability/file results. Normal base-environment sync was not rerun while
  the live server is using its installed optional dependencies.
- Live checks: HTTP root and `/health` return 200; HTTP run listing sees two FINISHED
  records; each has six evidence artifacts. `/get-artifact` serves the original M1
  summary successfully. SDK telemetry client is disabled; wrapper disables UI telemetry.
- Existing M1 import ID: `2bc4266fc7564093a8915c585d1155fd`; source commit/dirty-state,
  source snapshot and metric 0.17595877026340895 retained. Reimport does not duplicate.
- New CLI integration smoke ID: `3438643655224623a76a0247c9c70b21`, three train queries
  over the full corpus. This is a functionality check, not a comparable benchmark score.
- State: ignored `artifacts/mlflow/mlflow.db` and local file artifacts. The UI is started
  at `http://127.0.0.1:5000`; restart the documented command after WSL/session shutdown.
  No autostart service or remote access was configured. No commits/pushes in this task.
- Next: M2 dense/hybrid comparisons can use the established local tracking setup while
  retaining immutable file output, the original split and fixed validation selection.

### 2026-11-26 — M1 / frozen split and CPU BM25 baseline

- Goal: establish a deterministic, query-grouped sparse retrieval baseline and first
  failure analysis without changing official data/labels or evaluating final test queries.
- Files changed: `pyproject.toml`, `uv.lock`, archive loader in `data/source.py`, new
  `data/splits.py`, `retrieval/bm25.py`, `evaluation/baseline.py`, reference-prefix metric
  diagnostic, three new test modules, `configs/experiments/bm25_company_policies.json`,
  `data/splits/company_policies_v1.json`, aggregate `experiments/results/` JSON,
  `docs/m1_bm25_report.md`, README and relevant protocol/card/roadmap/decision/status docs.
- Exact commands executed:

  ```bash
  uv sync --group dev
  uv run --frozen ruff format .
  uv run --frozen ruff check .
  uv run --frozen pytest -q
  uv run --frozen ruff format --check .
  uv run --frozen temporalproof-prepare-split
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --frozen temporalproof-bm25 --output artifacts/bm25/company_policies_validation100_v1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --frozen temporalproof-bm25 --output artifacts/bm25/company_policies_validation100_replay_v1
  cmp artifacts/bm25/company_policies_validation100_v1/run.trec artifacts/bm25/company_policies_validation100_replay_v1/run.trec
  cmp artifacts/bm25/company_policies_validation100_v1/qrels.tsv artifacts/bm25/company_policies_validation100_replay_v1/qrels.tsv
  cmp artifacts/bm25/company_policies_validation100_v1/config.json artifacts/bm25/company_policies_validation100_replay_v1/config.json
  uv sync --frozen --group dev
  git diff --check
  ```

- Validation: **81 tests pass**; lint and format checks pass. Initial lint checks found
  an unused import and overlong lines, corrected before benchmark execution. Existing
  M0 tests still pass. Split regeneration succeeds without changing the frozen file.
- Replay validation: a Python comparison of `run.trec`, `qrels.tsv`, copied config and
  quality summaries found exact equality. Timing differences are retained, not hidden.
  A final checksum check matched generated files, aggregate results, the frozen split
  and the current implementation snapshot to the recorded run manifest.
- Split audit: 2,891 connected groups, largest 20; 2,058 train / 495 validation / 447 test
  query records. Query/template/near-query and exact shared-passage links use no qrels.
  Frozen split SHA-256: `4aab958538041f019363a8ffcc5c5d15d030ebe408d869c17063bf9c3c566fa7`.
  True semantic families/near-duplicate passages are not exhaustively audited.
- Scope: seed 1729; 100 hash-selected validation queries spanning 98 groups; top 100
  candidates from all 30,000 original passages; raw whitespace tokenization; no temporal
  filtering, text deduplication, neural models, tuning or final-test evaluation.
- Quality: qrel-based nDCG@10 0.1759587703; Recall@10 0.23; Recall@20 0.2816666667;
  Recall@100 0.3933333333. Separately named reference prefix-nDCG diagnostic 0.2427621397.
  Conditional query-bootstrap intervals and limits are in the report/results JSON.
- Failure observations: 88/100 queries lack complete gold in top 10; 45/100 have no
  gold in top 100. Fixed worst-five inspection found scenario/cohort/date confusions;
  these are observations, not relabeled temporal gold. Full text stays ignored locally.
- Timing scope: sequential warm CPU queries on AMD Ryzen 5 7500F, 12 visible logical CPUs,
  WSL2, Python 3.13.11, thread env variables=1. Initial mean/p50/p95: 70.205/70.288/101.995
  ms; replay: 69.003/69.707/97.380 ms. Not API/load-test timing. GPU and peak memory unmeasured.
- Provenance: pinned TimelyRAG revision/checksum unchanged; Git parent `ff7322d` with dirty
  working tree and exact implementation/dependency hashes recorded in run manifests.
- Not run: final test, dense/hybrid, neural reranker/router, MLflow, model downloads,
  GPU or load tests. No commits/pushes performed as part of M1.
- Next task: M2 dense/hybrid candidate comparison, with unchanged split/selection and
  per-query analysis; retain untouched test and existing temporal/license caveats.

### 2026-11-26 — M0 / Company Policies audit foundation

- Goal: turn the initial repository/dataset inspection into a repeatable, CPU-only audit.
- Files changed: `pyproject.toml`, `uv.lock`, `.gitignore`, package files under
  `src/temporalproof/data/` and `src/temporalproof/evaluation/`, three test modules and
  `tests/fixtures/timelyqa_synthetic.jsonl`, `configs/datasets/timelyqa_company_policies.json`,
  `docs/dataset_card.md`, `docs/benchmark_protocol.md`, README and agent handoff/protocol/roadmap/decisions.
- Source revision: `kaist-dmlab/TimelyRAG@5213a0c1743f23fd52a258005c3887aaf7e8cd0f`.
  Archive SHA-256: `a006481ffdf723d4b15554da2c621ebdfefd59b78ad37c58a633b3b71de247ad`.
- Setup command executed: `uv sync --group dev`.
- Frozen setup rechecked successfully: `uv sync --frozen --group dev`.
- Validation commands executed:

  ```bash
  uv run --frozen ruff format .
  uv run --frozen ruff check .
  uv run --frozen pytest -q
  uv run --frozen ruff format --check .
  /usr/bin/time -v uv run --frozen temporalproof-audit --sample-size 50
  uv run --frozen temporalproof-audit --download --archive data/raw/download_smoke/datasets.zip --output artifacts/audits/download_smoke.json
  uv run --frozen temporalproof-audit --output artifacts/audits/repeat.json
  cmp artifacts/audits/company_policies.json artifacts/audits/repeat.json
  cmp artifacts/audits/company_policies.json artifacts/audits/download_smoke.json
  git check-ignore data/raw/timelyqa/5213a0c1743f23fd52a258005c3887aaf7e8cd0f/datasets.zip
  git diff --check
  ```

- Results: **61 tests passed**; Ruff lint/format checks passed. Fresh opt-in download,
  cached offline audit and repeated audit succeeded; JSON reports were byte-identical.
  Full audit counted 3,000 queries / 30,000 unique document IDs. The first 50 records
  (500 documents) provide a deterministic schema sample.
- Audit observations: 2,976 distinct query texts, 27,199 distinct document texts;
  no local/global gold mapping disagreements; 7,533 binary positive judgments;
  one empty answer; ten optional document titles; no explicit event-time fields;
  47 unparseable document insertion strings; 1,253 later-inserted positives across
  911 queries (all positives are later-inserted in 146 queries).
- Early checks that failed, now resolved: unknown JSON keys appeared in error locations;
  sanitizer now redacts them and its regression test passes. Initial strict timestamp
  validation and nonempty-answer assumptions stopped the real audit; the adapter now
  preserves malformed timestamp strings, reports unknown parsed times, and accepts the
  observed empty answer without discarding qrels. Initial Ruff line-length findings and
  the final diff's README trailing-whitespace finding fixed.
- Research decisions: ADR-007 preserves raw time uncertainty; ADR-008 separates qrel-based
  metrics from reference reproduction. Reference source files remain unchanged.
- Not run: BM25, any model download/training, final train/test evaluation, GPU checks,
  API/load tests, Python 3.11 execution or clean-machine setup.

## Open questions / constraints

1. **License:** inspected pinned repository/archive contain no explicit license file;
   GitHub license metadata was null. The paper describes a permissive research release
   without specifying precise terms in the inspected statement. Do not redistribute
   benchmark text/reference code on the basis of the project's MIT license.
2. **Temporal evaluation:** insertion time is not valid time. No event-time/validity labels
   are supplied. Strict known-at filtering conflicts with later-inserted positives;
   keep original labels intact and define any temporal slice separately.
3. **Splits:** M1 supplies a custom frozen grouping/split; it is not an author-provided
   split. Query templates/near queries and exact shared passages are grouped, but residual
   semantic-family and near-passage leakage risk remains. Full unlabeled corpus is shared;
   this is query-held-out evaluation, not corpus/domain/temporal separation.
4. **Official comparison:** reference nDCG normalizes over retrieved positives; reference
   best-alpha uses query qrels. Neither can silently replace the main qrel-based protocol.
5. **Models / transfer:** Korean support and actual memory/latency are not measured.
   TEMPO integration and any judge API are not implemented.

## Recommended next task

Implement M2: add a pinned compact encoder suitable for Korean and hybrid candidate
retrieval, using the same frozen validation selection/corpus. Compare paired query deltas,
regressions and candidate recall before reranking. Use the implemented local MLflow and
preserve portable run manifests. Do not tune on or evaluate final test yet. Original
qrels and temporal uncertainty remain unchanged. See `docs/m1_bm25_report.md`.
