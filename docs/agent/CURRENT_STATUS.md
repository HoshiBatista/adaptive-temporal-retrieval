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
  historical import and `--mlflow` are implemented. M1 validation100, a separate train
  integration smoke, M2 methods and both M3 models are FINISHED in the local store. The final M2
  records are tagged `environment_lock_verified`; test remains unevaluated.
- **M2:** pinned multilingual E5 GPU encoder, FAISS cosine index/cache, RRF fusion and
  insertion-time diagnostic implemented. On the original validation100, nDCG@10 is
  BM25 0.175959 / dense 0.150042 / hybrid 0.207685 / hybrid+insertion 0.203661.
  Paired intervals contain zero; no superiority claim.
- **M3:** fixed compact MiniLM cross-encoder, checked frozen hybrid candidates, complete
  pair-score output and GPU/CPU-stage profiling implemented. Two runs have byte-identical
  rankings/scores, qrels/configs and metrics. nDCG@10 **0.109982**, below hybrid 0.207685;
  7 query gains / 33 regressions, mean incremental rerank **149.703 ms**, p95 204.211 ms.
  Recall@100 stays 0.415 per unchanged candidates. Negative result retained locally.
- **M3 Korean follow-up:** pinned bge-reranker-v2-m3-ko reaches nDCG@10 **0.163721**,
  above MiniLM but below hybrid; 11 gains/28 regressions versus hybrid, mean incremental
  rerank **1,148.100 ms**, p95 **1,500.089 ms**. Two GPU runs have identical scores and
  metrics; source/candidate/qrel integrity verified. **124 CPU tests pass**. No default
  model switch, router training, further model sweep or test evaluation.
- **Environment:** `pyproject.toml`, `uv.lock`, Python >=3.11 declared; validation executed
  on Linux x86-64, CPython 3.13.11. Actual locked versions include Pydantic 2.14.4,
  pytest 9.2.0, Ruff 0.16.10, rank-bm25 0.2.2 and NumPy 2.5.5.
- **Research results:** M1/M2 validation-subset baselines and paired analysis only;
  fixed reranker comparison now available; no router, final test or official whole-benchmark reproduction.
- **Application:** no API, database, UI, generation or ChangeImpact implementation.
- **Data:** pinned 5.8 MB archive in ignored `data/raw/`; derived reports in ignored
  `artifacts/`. No source benchmark content is committed; test data is independently authored.
- **Next increment:** M4 predeclared small train-only utility pilot. Determine whether
  useful fast/deep gains are predictable at the measured cost before learning a router;
  hybrid remains the fast/default reference.

## Verified work log

### 2026-11-26 — M3 follow-up / pinned Korean alternative

- Goal: check one explicitly Korean-optimized reranker on the identical frozen M2 pool
  after MiniLM's regression. Existing M2/M3 working-tree changes and evidence preserved.
- Authorization: user approved starting the proposed alternative-model check; earlier
  GPU-download authorization remains applicable. Downloaded ~2.29 GB pinned local assets;
  no new dependency, paid API, remote push or training run introduced.
- Model: dragonkue/bge-reranker-v2-m3-ko revision
  `2aca5884ecac490192af9ebd86836d9073d826cd`; author declares Korean optimization and
  Apache-2.0. Complete fine-tuning data undisclosed, so no independent decontamination
  claim. Six asset sizes/hashes and pinned card hash recorded in a model manifest.
- Files: small ModelSpec registry/shared offline loader in `retrieval/reranker.py`;
  model/revision-pair config validation and model-specific default directory in the
  existing reranking CLI; separate Korean model/experiment configs, seven added CPU
  cases in `tests/test_korean_reranker.py`, report/aggregate and documentation updates.
- Commands executed:

  ```bash
  PYTHONPATH=src .venv/bin/python -m temporalproof.retrieval.reranker --model-id dragonkue/bge-reranker-v2-m3-ko --download
  uv run --frozen --extra tracking --extra dense ruff format .
  uv run --frozen --extra tracking --extra dense ruff check .
  uv run --frozen --extra tracking --extra dense pytest -q
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 uv run --frozen --extra tracking --extra dense temporalproof-rerank --config configs/experiments/m3_korean_company_policies.json --mlflow --output artifacts/m3/korean_company_policies_validation100_v1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 uv run --frozen --extra tracking --extra dense temporalproof-rerank --config configs/experiments/m3_korean_company_policies.json --output artifacts/m3/korean_company_policies_validation100_replay_v1
  ```

- Validation: **124 CPU tests pass**, Ruff lint/format and diff checks pass. New tests
  cover supported/unsupported models and revision pairing, unchanged experiment protocol,
  manifest consistency, checks before GPU/network use, model-specific pinned downloads
  and corruption cleanup. No network/model/GPU inference in normal tests. Existing
  third-party MLflow/SQLAlchemy warning remains. Both CLI help commands checked.
- Independent synthetic Korean GPU smoke ranks a Seoul/South-Korea passage above a
  Paris/France passage (raw logits 12.5105 / -6.8893). Loader has no missing, mismatched
  or unexpected weights. This checks operation/direction, not domain quality.
- Comparability: exact original M2 manifest, pool hash, query order, qrels, cutoffs,
  seed 1729, FP32/eager/TF32-off, batch 16, max 512, longest-first, concurrency 1.
  No tokenizer/model mixing: both tokenizers are pinned and differences disclosed.
  Three selected-query warmups; 10,300 pairs including warmup, none at the length limit.
- Quality: nDCG@10 **0.1637214591**, Recall@10 **0.2**, Recall@20 **0.245**,
  Recall@100 **0.415**; candidate coverage unchanged per query, 43 queries with no gold
  in pool. Versus hybrid: paired nDCG **−0.043963**, conditional interval
  **[−0.086258, −0.004264]**, 11 gains/28 regressions/61 unchanged. Versus MiniLM:
  **+0.053740**, interval **[0.024141, 0.084094]**, 21 gains/5 regressions/74 unchanged.
  Conditional development-sample intervals do not model group dependence or establish
  unseen-domain behavior. No significance/generalization claim beyond that scope.
- Error inspection: 2351/2763/2487/2313 regress, 2059/5/1214 improve. Both policy
  scenario/date confusions and closely matching unjudged alternatives appear; no labels
  changed. Report uses ID-level outcomes and paraphrases, not redistributed raw text.
- Resources: initial/replay mean **1,148.100 / 1,148.535 ms**, p95
  **1,500.089 / 1,499.717 ms**; CPU tokenization mean **20.653 / 20.621 ms**,
  synchronized forward/transfers **1,126.857 / 1,127.344 ms**. Same RTX 5060 Ti/Ryzen
  host and thread environment as M3; encoder unloaded. Both peaks **2,676.298 MiB
  allocated / 3,832 MiB reserved** in PyTorch. No combined pipeline/service/total-VRAM claim.
- Replay: byte-identical all 10,000 scores/rankings, qrels and config; matching metrics,
  candidate hashes and paired deltas. Current source matches both new run snapshots.
  Original BM25/M2/MiniLM evidence retained. MLflow **144b746d2199493fba60acfa5c130f05**
  FINISHED with six evidence artifacts verified. Aggregate includes comparisons against
  hybrid and MiniLM and replay telemetry.
- Decision: neither measured reranker becomes the default always-deep path. No further
  model sweep. Next: small predeclared train-only fast/deep utility pilot before learning
  a router. Validation outcomes are not training targets or inference features.
- Not run: train pilot/targets, router learning, final test/TEMPO, CPU-only inference,
  mixed precision/quantization, live full pipeline, simultaneous-model VRAM or load tests.
  No commit or push performed.


### 2026-11-26 — M3 / fixed reranker, negative result and GPU profiling

- Goal: measure a fixed deep ranking path's incremental quality/cost over the original
  M2 hybrid top-100 before adaptive routing. Existing M2 working-tree changes preserved.
- Authorization: prior user GPU-download approval remains applicable; ~493 MB pinned
  reranker assets downloaded. No training, paid API, remote push or test evaluation.
- New files: `retrieval/reranker.py`, `evaluation/reranking.py`, `tests/test_m3.py`, pinned
  model/experiment configs, `docs/m3_reranker_report.md` and aggregate M3 results. Common
  evaluator adds a label-free ID-aware callback and explicit development warmup override;
  CLI entrypoints, README/protocol/roadmap/decision/tracking docs updated.
- Commands executed:

  ```bash
  PYTHONPATH=src .venv/bin/python -m temporalproof.retrieval.reranker --download
  uv sync --extra tracking --extra dense --group dev
  uv run --frozen --extra tracking --extra dense ruff format .
  uv run --frozen --extra tracking --extra dense ruff check .
  uv run --frozen --extra tracking --extra dense pytest -q
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 uv run --frozen --extra tracking --extra dense temporalproof-rerank --mlflow --output artifacts/m3/company_policies_validation100_v1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 uv run --frozen --extra tracking --extra dense temporalproof-rerank --output artifacts/m3/company_policies_validation100_replay_v1
  ```

- Validation: **117 CPU tests pass**, Ruff lint/format and diff checks pass. Tests use
  synthetic CPU data/fake inference; cover frozen-source hashes, ID/rank/score/cutoff
  errors, qrel/config/split drift, deterministic ties, explicit verified downloads and
  duplicate-query-text ID joins with an untouched synthetic test query. Existing
  third-party SQLAlchemy deprecation warning remains.
- Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 revision
  `1427fd652930e4ba29e8149678df786c240d8825`, model card Apache-2.0. Korean is absent from
  documented supervised mMARCO languages; this is transfer evaluation. FP32/eager,
  TF32 off, batch 16, max 512, dynamic padding/longest-first, raw scalar logit/ID ordering.
  Loader has no missing/mismatched learned weights and verifies legacy position buffer.
  Separate Korean smoke ranks the synthetic relevant passage higher; no benchmark claim.
- Frozen source: original verified M2 hybrid manifest
  `fef933fc67bb78de87ff0c5bcfcb4ba7260dcbe83ba0b53e2185b5e83b7d51dc`; input run
  `8df0d79f34035189c5aa385ea57794b18aa10cee8cd2db51a4d13f7f3700d6ab`. All source outputs,
  original split/query selection, 100 candidates/query and qrels verified before scoring.
  Cross-encoder receives only query/passage text. All 10,000 pair scores saved locally.
- Result: nDCG@10 **0.1099817658**, Recall@10 **0.135833**, Recall@20 **0.169167**,
  Recall@100 **0.415**, 43 queries without gold in pool. Paired nDCG delta **−0.097703**,
  conditional interval **[−0.144189, −0.056641]**, 7 gains/33 regressions/60 unchanged.
  No generalization/significance claim beyond the stated conditional development sample.
- Error inspection: IDs 2351/2313/2763 regress; 1868/758/200 improve. Some unjudged
  top hits closely match query scenarios; others confuse policy conditions. No qrels or
  source data changed. No pair reached 512 tokens (10,300 including warmup).
- Initial/replay rerank mean **149.703 / 147.124 ms**, p95 **204.211 / 200.628 ms**;
  CPU tokenization mean **20.179 / 19.811 ms**, synchronized forward/transfers mean
  **129.142 / 126.947 ms**. Concurrency 1, three explicit selected-query warmups excluded
  from latency, thread environment=1 on the same RTX 5060 Ti/Ryzen 5 7500F host.
- Both runs peak **784.868 MiB allocated / 1,540 MiB reserved** in PyTorch's allocator.
  Encoder not loaded concurrently. These are rerank-only stage timings and allocator
  telemetry, not live end-to-end p95, service throughput or total process VRAM/RSS.
- Replay: byte-identical rankings including scores, qrels/config; identical quality,
  candidate pool hashes and paired deltas. Final source snapshot matches run manifests.
  M1/M2 original evidence unchanged. Local MLflow run
  `1419b5ef303b4611a38e40a9f61cdfa9` FINISHED, six evidence artifacts verified; UI health OK.
- Not run: CPU-only inference, alternative reranker, training/router labels, final test,
  TEMPO, live hybrid+reranker pipeline/load tests. No commit/push performed.
- Next gate: retain negative MiniLM baseline and check another explicitly Korean-capable
  reranker on identical candidates before M4. Future router targets must be train-only;
  these validation outcomes must never become router-training labels.

### 2026-11-26 — M2 / GPU dense, hybrid and insertion-time baselines

- Goal: compare lexical/semantic/hybrid retrieval and a documented temporal metadata
  heuristic under the unchanged M1 corpus, split and validation selection.
- Authorization: user explicitly approved GPU model/dependency downloads ("для гпу все").
  No model training, paid APIs, remote tracking or final-test evaluation was used.
- Files changed: `pyproject.toml`, `uv.lock`; common evaluator injection/method metadata;
  `retrieval/encoder.py`, `dense.py`, `fusion.py`, `evaluation/comparison.py`, `tests/test_m2.py`;
  M2 config, model asset manifest, aggregate results and report; README and agent/protocol docs.
- Commands executed:

  ```bash
  uv sync --extra tracking --extra dense --group dev
  PYTHONPATH=src python3 -m temporalproof.retrieval.encoder --download
  uv run --frozen --extra tracking --extra dense ruff format .
  uv run --frozen --extra tracking --extra dense ruff check .
  uv run --frozen --extra tracking --extra dense pytest -q
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 uv run --frozen --extra tracking --extra dense temporalproof-compare --mlflow --output artifacts/m2/company_policies_validation100_v1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 uv run --frozen --extra tracking --extra dense temporalproof-compare --output artifacts/m2/company_policies_validation100_replay_v1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 uv run --frozen --extra tracking --extra dense temporalproof-compare --mlflow --output artifacts/m2/company_policies_validation100_verified_v1
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 uv run --frozen --extra tracking --extra dense temporalproof-compare --output artifacts/m2/company_policies_validation100_verified_replay_v1
  ```

- Validation: **102 CPU tests pass**, Ruff checks pass. Initial syntax/import/line-length
  findings corrected before measurements. New tests cover FAISS tie/ID alignment, bad
  vectors, explicit checked downloads, cache integrity/reuse, rank fusion, missing dates,
  equal/past/future insertion boundaries, candidate parity, paired query joins/regressions
  and the common label-free retrieval callback. Unit tests use no model/GPU inference.
- GPU smoke: two independent Korean strings encoded to finite unit vectors `(2,384)`.
  Model load has no missing/mismatched learned parameters; a legacy position-ID buffer
  was checked against its generated default, without changing original weights.
- Model: intfloat/multilingual-e5-small@614241f622f53c4eeff9890bdc4f31cfecc418b3,
  MIT card, six pinned assets (~493 MB), FP32, eager attention, max 512, batch 16, prefixes
  and masked-mean/L2 pooling. PyTorch 2.10.0 / Transformers 5.5.0 / FAISS CPU 1.16.0.
- Hardware: RTX 5060 Ti 16 GB, CUDA 12.8; CPU Ryzen 5 7500F / WSL2 / Python 3.13.11.
  Sequential concurrency 1, thread environment=1; query GPU work is synchronized.
- Scope: all 30,000 original passages, original 100 validation IDs/split SHA unchanged;
  top 100 per method, fixed RRF offset 60 and insertion bonus 0.1. The temporal variant
  is a symmetric insertion-gap diagnostic, not event-time/validity inference or filtering.
- Quality: BM25 nDCG@10 0.1759587703 / dense 0.1500424340 / hybrid 0.2076845949 /
  insertion 0.2036611501. Recall@100 respectively 0.393333 / 0.361667 / 0.415 / 0.415.
  Hybrid's paired nDCG delta +0.031726 has conditional interval [-0.008300, 0.071907];
  24 queries improve, 17 regress. Insertion versus hybrid delta -0.004023: 7 improve,
  9 regress, 84 unchanged. No significance/generalization claim is made.
- Initial/replay rankings including scores, qrels, copied configs and quality summaries
  match exactly. M1 BM25 quality and selected IDs match exactly. Temporal/hybrid candidate
  sets match for every query. Full texts/qrels remain ignored locally; unchanged ID-level
  judgments can penalize useful unjudged text, as shown by case 1214.
- Evidence: `docs/m2_retrieval_report.md` and aggregate results include source manifests,
  conditional intervals, resources, replay scope and positive/negative examples. A final
  cache hardening adds the environment lock hash so dependency changes cannot silently
  reuse embeddings. Final verification/replay also produced byte-identical rankings,
  scores, qrels/configs and metrics; their source snapshot and environment lock match the
  delivered implementation. Earlier check runs are retained, not hidden or pooled as
  additional independent queries.
- Final resources: embedding/cache build 34.076 s; FAISS build 93.680 ms; peak PyTorch
  allocator 923.195 MiB allocated / 1,518 MiB reserved, not total-process VRAM. Sequential
  mean times BM25/dense/hybrid/insertion: 77.755 / 11.913 / 88.590 / 90.572 ms; replay
  variation and exact scope are in the report. No host peak RSS or service throughput claim.
- Final MLflow run IDs: BM25 `7f78491db25643d09fde0f681a17b125`, dense
  `f66fcec374824e9095c600cfa229d9e6`, hybrid `ffb70dac7f2a4977acc4074d5a82cf16`,
  insertion `bcaae473e89d401587302b2e9800a626`.
- Not run: final test, domain/TEMPO transfer, full official TimelyRAG reproduction,
  neural reranker/router, service load tests or total-process peak VRAM/RSS.
- Next: freeze a candidate pool and measure a compact fixed multilingual reranker's
  incremental quality/time before generating training-fold router labels.

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
5. **Models / transfer:** one Korean-capable E5 baseline is measured locally; cross-domain
   transfer and full-process resource measurements remain unverified. A compact multilingual
   MiniLM reranker regresses on Korean transfer. A separately pinned Korean BGE follow-up
   is now measured and also regresses versus hybrid; neither is adopted as default deep
   path. Its complete fine-tuning corpus is undisclosed; decontamination is not established.
   TEMPO integration and any judge API are not implemented.

## Recommended next task

The Korean follow-up is measured and also regresses against hybrid. Next implement a
small deterministic train-only feasibility pilot: freeze train candidates and fast/deep
outcomes, inspect gain/cost and pool-coverage distributions, and decide whether available
pre-deep features justify learning routing policies. Predeclare the pilot size/config;
retain negative outcomes. Validation results must never become router-training labels
or inference features. Keep final test untouched. See `docs/m3_korean_reranker_report.md`.
