# Milestone plan and task acceptance gates

This is a **suggested order**, not a promise that all work is done or that it takes a fixed number of days. For each milestone, implement and test the smallest end-to-end slice. Update `CURRENT_STATUS.md` with verified facts.

## M0 — Repository and dataset audit (start here)

**Objective:** verify the data and benchmark before constructing a large system.

Tasks:
- [x] Inspect actual file tree, dependency manager, README and git state.
- [x] Confirm official TimelyRAG repository and exact dataset download instructions.
- [x] Check dataset license/terms, access, filenames, field schema and split metadata.
- [x] Sample records and detect time-field ambiguity, duplicate queries and missing judgments.
- [x] Write `docs/dataset_card.md` with provenance, sample stats, and split conventions.
- [x] Add CPU-only deterministic schema/metric unit tests and a tiny fixture.

Verified 2026-11-26: Company Policies audit runs over all 3,000 queries / 30,000 documents;
61 CPU tests pass. Checking terms found no explicit license file; exact rights remain
unresolved. The research gate is conditional: no event-time labels or frozen split are
available, and later-inserted positives prevent unqualified known-at claims. See the
[data card](../dataset_card.md) and [benchmark protocol](../benchmark_protocol.md).

**Acceptance:** no invented dataset fields, reproducible audit command, tests genuinely run; any external access blocker documented.

## M1 — Reference metrics and sparse baseline

Tasks:
- [x] Implement relevance-grade mapping and tested nDCG@10/Recall@K functions.
- [x] Build small BM25 index and evaluate held-out queries.
- [x] Save per-query run and qrel files; record subset selection and seed.
- [x] Create a first error analysis report including temporal hard cases.

Verified 2026-11-26: library CPU BM25 evaluated 100 deterministic validation queries
against all 30,000 Company Policies passages; test was not evaluated. The frozen grouped
split, config and aggregate provenance are included in the repository; full runs/qrels and
failure text stay ignored. Replay rankings and quality metrics match exactly; 81 CPU tests
pass. See [M1 report](../m1_bm25_report.md). Failure categories are exploratory observations,
not new temporal gold labels; timings are sequential local query timings, not load tests.

**Acceptance:** baseline metrics computed using an explicitly documented protocol; no query/pair leakage; CPU smoke command works.

## M2 — Dense and temporal retrieval

Tasks:
- [x] Choose/pin a compact embedding model and index version.
- [x] Compute dense candidate rankings on exactly the same evaluation split.
- [x] Add hybrid fusion and explicit temporal filtering/ranking features.
- [x] Compare B1/B2/B3/B4 with confidence intervals or per-query deltas.

Verified 2026-11-26: multilingual-e5-small GPU FP32, FAISS exact cosine, fixed RRF and
soft insertion-time ranking evaluated on the original 100 validation IDs. The temporal
variant is an insertion-metadata diagnostic, not validity inference or an official
TimelyRAG reproduction. Dense mean quality regressed; hybrid's mean gain has a paired
interval containing zero; the insertion bonus did not improve hybrid mean nDCG. Original
qrels/test remain untouched. See [M2 report](../m2_retrieval_report.md) for replay, hardware,
resource scope and failure cases. 102 CPU tests pass without model inference/downloads.

**Acceptance:** model provenance + repeatable datasets; time-boundary test cases; report includes both improvements and regressions.

## M3 — Deep reranker and profiling

Tasks:
- [x] Implement fixed top-K cross-encoder reranking.
- [x] Cache fixed candidate pools and output scores.
- [x] Profile CPU/GPU latency at controlled batch size and concurrency.
- [x] Record GPU peak and trade-off against the fast model.

**Acceptance:** deep reranker gives a measurable basis for training the router; no unsupported GPU capacity claims.

Verified M3: frozen hybrid top-100 + pinned multilingual MiniLM; 117 CPU tests, two
reproducible GPU runs and local MLflow. nDCG@10 drops 0.207685 → 0.109982 with mean
incremental rerank 149.703 ms. CPU tokenization/GPU-forward-transfer stages are measured;
CPU-only inference and live end-to-end timing are not. See [M3 report](../m3_reranker_report.md).

**Korean follow-up checked:** pinned bge-reranker-v2-m3-ko on the identical pool reaches
nDCG@10 0.163721 (hybrid 0.207685), 11 gains/28 regressions, mean incremental cost
1,148.100 ms. Two GPU runs reproduce all scores exactly; **124 CPU tests pass**.
Neither reranker is adopted as the default deep path. The first M4 train-only utility
pilot is now measured below, before learning a route policy.
See [follow-up report](../m3_korean_reranker_report.md).

## M4 — Main research contribution: adaptive router

Tasks:
- [x] Compute **training-only** fast/deep gains and measured-cost targets (initial 100-query pilot).
- [ ] Implement pre-deep routing features (time intent, rank disagreement, margin/entropy, version diversity).
- [x] Compare fixed/rule/learned logistic routing (grouped OOF train500 and frozen development validation495).
- [ ] Evaluate quality–cost Pareto curves on an untouched test split.
- [ ] Run feature ablations, temporal-hard slices and a domain-transfer trial if feasible.

Initial pilot verified: 100 train queries / 100 source groups, 15 positive gains,
18 regressions, 67 unchanged. Strict 15-field pre-deep schema and separate train targets
implemented; dates/text repetition are proxies, not actual version chains. Fixed rules,
seeded random budgets and a clearly offline oracle bound are measured descriptively.
No fitted router or held-out routing evaluation yet. 140 CPU tests, exact-score/feature
replay and local MLflow verified. See [M4 pilot](../m4_train_pilot_report.md).
Next increment measured: 500 queries / 493 groups, five group folds, train-fold-only
imputation/scaling/logistic fits. Primary budget routes 49 queries: nDCG@10 0.198929 versus
hybrid 0.197016 and matched agreement control 0.204155; all primary contrast intervals
include zero. Two feature sets/three cost penalties are retained; no learned superiority.
157 CPU tests, exact CPU model/OOF replay and local MLflow verified. Primary all-selected-
train JSON model was saved before validation. See [grouped router report](../m4_router_groupcv_report.md).

Frozen development validation verified: all 495 queries / 462 groups; primary routes
49 and reaches nDCG@10 0.190746 versus hybrid 0.198972 and agreement 0.198817. Primary
minus hybrid group interval [−0.016980, −0.000566] is below zero; matched-control
intervals include zero. Full deep reaches 0.148561. No validation fit/targets/threshold
tuning, no test evaluation. Exact CPU score/decision/metric replay and original
validation100 GPU rankings/scores/qrels verified; 167 CPU tests and local MLflow pass.
Retain the negative result and hybrid reference. Next inspect failures to predeclare a
train-only experiment, rather than select a new validation operating point. See the
[validation report](../m4_router_validation_report.md).

M4.2 measured: cached failure audit separates absent candidates from harmful permutations;
predeclared rank-blend weights use train-fold selection only. Final .20 is frozen before
validation. OOF nDCG0.196354 versus fast0.197016; validation0.196351 versus fast0.198972,
both intervals include zero. Old-route transfer0.199371 is secondary, not a new learned
policy or primary switch. Exact CPU replay,188 tests and local MLflow verified. No new
neural inference/test. Next prioritize the official TimelyRAG baseline on a matched
protocol before further changed-utility router fits; keep reference/full-qrel metric
columns and automatic-alpha/oracle outcomes distinct. Candidate coverage remains separate.
See [report](../m4_rankblend_report.md).

M4.3 measured: pinned native auto-alpha code on validation495, BM25 reproduction and
CUDA E5/hybrid transfers. All1485 real original auto top10s match; two GPU runs preserve
score matrices/ranks/alpha/quality. Native full-qrel means BM25 .154621→.146812, E5
.141065→.131546 (intervals include zero), hybrid .198335→.173203 (negative interval).
Reference-prefix and native tie policies are explicit; no whole-paper/BGE-M3 claim.
206 tests with source integration; default202 pass/4 optional skips. No benchmark oracle,
new fit or final test. Next audit derived event-role/fallback policy on train. See the
[reference report](../m4_timelyrag_reference_report.md).

**Acceptance:** no oracle features at inference; held-out evaluation with reproducible scripts and a meaningful conclusion (positive or negative).

## M5 — Minimal production-style service

Tasks:
- [ ] Add version-aware ingestion with stable document/version/passage IDs.
- [ ] Provide FastAPI endpoints for search, versions and evidence provenance.
- [ ] Persist required state with tests and migration(s) where appropriate.
- [ ] Add request validation, structured error handling and basic logging.
- [ ] Package local components in Docker Compose if it simplifies reproducibility.

**Acceptance:** end-to-end integration: ingest two revisions → search as of each date → see distinct cited evidence.

## M6 — Product features and validation

Tasks:
- [ ] Add minimal React interface with search, timeline and citation view.
- [ ] Implement conservative ChangeImpact: changed cited passage → `review_required`.
- [ ] Add unit, integration, contract and one E2E test for revisions and history.
- [ ] Perform load tests: fast-only, mixed routing, reads during ingestion; separately profile optional generation.
- [ ] Add CI, Prometheus metrics/structured logs, documentation and demo fixtures.

**Acceptance:** deterministic demo, real test results, observed p50/p95/RPS in a clearly described environment; no assertion of automatic factual invalidation.

## M7 — Thesis evidence and open-source polish

Tasks:
- [ ] Freeze experimental protocol/configs and record dataset/model versions.
- [ ] Prepare comparison tables, Pareto plots, ablation, error taxonomy and honest limitations.
- [ ] Add setup documentation, sample environment template, security/licensing review and architecture diagram.
- [ ] Validate clean clone/setup instructions on a fresh environment where feasible.

**Acceptance:** reproducible end-to-end research story and usable service demonstration; no invented results.

## Research gates (stop/go)

- **After M0:** do the labels and time metadata actually permit the intended ranking experiment? If no, raise this with the project owner before changing the thesis scope.
- **After M3:** is there enough quality difference and runtime difference between fast/deep paths to make adaptive routing meaningful? If no, analyze another compact reranker or pivot within the same retrieval problem.
- **After M4:** does the learned router beat a simple policy at a defensible operating point? If no, publish the negative result and narrow the claimed contribution.

## Scope discipline

Keep M0–M4 ahead of heavy frontend work. Optional Rust/Go/large LLM components are not prerequisites. Do not “complete” milestones by adding placeholder folders or TODO-only modules.
