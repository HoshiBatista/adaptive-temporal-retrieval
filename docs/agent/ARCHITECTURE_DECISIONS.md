# Architecture constraints and decision log

This document records **planned architectural boundaries**, not implementations. Agents must inspect the repository before asserting any component exists.

## Core design principles

1. **Retrieval quality is independent of generation quality.** Evaluate ranked evidence without requiring an LLM API.
2. **Time is a first-class query constraint**, not just a recency boost.
3. **Provenance survives every transformation:** corpus document ID → version ID → passage/chunk ID → retrieved evidence → saved answer reference.
4. **Fast path and deep path share candidates and metadata.** Avoid comparing different candidate pools unless that is the explicitly studied variable.
5. **Optimize only after measuring.** Build a CPU-friendly baseline before adding a Rust gateway, GPU reranking, async pipelines or a web dashboard.
6. **Degrade gracefully.** If timestamps are missing, return unknown applicability; if a reranker is offline, use the fast path; if answer generation is disabled, still return ranked evidence.

## Logical components (planned)

```text
Versioned source documents
         |
  Data loader / metadata normalizer
         |
 [Document + version + passage IDs]
         |
 +-------+---------------+
 |                       |
BM25 index             Vector index
 |                       |
 +----- hybrid retrieval-+
                |
     fast semantic-temporal features
                |
       lightweight route policy
             /        \
        fast path    deep reranker
             \        /
       ordered evidence + citations
                |
  optional grounded response generator
                |
 answer → cited passage references
                |
 source revision → ChangeImpact review flag
```

## Proposed package modules

| Module | Responsibilities | Do not put here |
|---|---|---|
| `src/temporalproof/data/` | benchmark loaders, manifests, validation | retriever training business logic |
| `src/temporalproof/temporal/` | temporal filters, validity intervals, version semantics | HTTP requests and DB side effects in pure functions |
| `src/temporalproof/retrieval/` | sparse, dense, hybrid candidate retrieval | answer generation |
| `src/temporalproof/reranking/` | fixed fast/deep rankers, features | query gold judgments at inference |
| `src/temporalproof/routing/` | route prediction, routing budgets, routing features | access to gold/qrels or post-deep inference |
| `src/temporalproof/evaluation/` | exact benchmark protocols and metrics | live user-data writes |
| `src/temporalproof/provenance/` | evidence refs and persistent mappings | irreversible rewriting of original revisions |
| `src/temporalproof/changeimpact/` | review-required notification on changed cited evidence | assertions that the original answer is false |
| `src/temporalproof/api/` | typed service routes, validation | ML experiment training loops |

Use the README's proposed structure as a **guide**; create folders only when concrete code needs them.

## Interfaces to decide before implementation

Data record (conceptual; adapt to verified benchmark):

```text
DocumentVersion {document_id, version_id, published_at?, valid_from?, valid_until?, passages[]}
Passage {passage_id, version_id, text, source_uri?, section_id?}
SearchQuery {query_id?, text, as_of?, time_range?, requested_time_mode?}
SearchHit {passage_id, version_id, fast_score, deep_score?, temporal_status, evidence_uri}
RouterDecision {path: FAST|DEEP, reason_codes[], features_version, policy_version}
SavedAnswer {answer_id, question, answer_text?, evidence_refs[], time_intent, created_at}
ChangeImpactEvent {answer_id, changed_evidence_refs[], status: REVIEW_REQUIRED|REVIEWED}
```

`valid_from`/`published_at` are optional only when unknown. Never synthesize fake values for missing dates without a policy and an audit trail. Use fixed rules for interval boundaries.

## Data and storage boundaries

- **Raw benchmarks:** immutable, gitignored, with source manifest/checksum.
- **Derived data:** versioned preprocess outputs, feature caches and candidate lists; deterministic rerun supported.
- **Metadata/product state:** PostgreSQL when persistence is necessary; use minimal test fixtures before introducing a database.
- **Vector search:** FAISS *or* pgvector for MVP, not both without evidence.
- **Lexical search:** library-based BM25 first; consider Tantivy/Rust only after a measurement.
- **Monitoring:** structured logs and request IDs first; Prometheus/Grafana for the integrated service.

## Decisions (proposed; review when new evidence arrives)

| ID | Status | Decision | Rationale | Revisit trigger |
|---|---|---|---|---|
| ADR-001 | PROVISIONAL | Python-first, CPU-testable research package | Iteration speed; baseline clarity | Measured CPU bottleneck |
| ADR-002 | PROVISIONAL | TimelyQABench primary; TEMPO external validation | Existing relevance judgments | License/schema incompatibility |
| ADR-003 | PROVISIONAL | Simple learned router before neural policy | Lower overfitting risk and cost | Measured failure at fixed routing budget |
| ADR-004 | PROVISIONAL | ChangeImpact is dependency-based review notification | Avoids need for own staleness labels | A separate validated staleness benchmark becomes available |
| ADR-005 | PROVISIONAL | Optional generator, not part of required research metrics | Prevents LLM confounders | A reliable answer QA benchmark is verified |
| ADR-006 | PROVISIONAL | Rust gateway deferred | Avoids premature complexity | Stable Python benchmark reveals a service bottleneck |

### ADR-007: Preserve raw benchmark timestamps and audit uncertainty
Date: 2026-11-26
Status: ACCEPTED for the M0 adapter

Company Policies contains 47 insertion strings outside the exact reference-format policy,
no explicit event-time fields, and positive evidence inserted after query issuance. The
typed adapter retains raw strings; parsing returns unknown for missing or malformed
values, and the audit reports malformed counts separately. It does not repair values,
derive event times, infer validity intervals or filter judgments. Unknown schema fields
fail explicitly, so a different revision/domain needs an observed-schema review. Input
errors omit source values and unknown field names. Revisit on a documented derived
normalization or an expanded benchmark adapter; raw inputs stay immutable.

### ADR-008: Separate qrel-based metrics from reference reproduction
Date: 2026-11-26
Status: ACCEPTED for initial evaluation

Binary nDCG uses ideal DCG from the full positive set; Recall uses the full positive-set
denominator. The reference normalizes nDCG using the retrieved prefix, and its best-alpha
selection uses qrels per query. Preserve reference files unchanged and label any future
reproduction separately; best-alpha is an offline diagnostic. Tests cover the missing-gold
normalization counterexample. See [`benchmark_protocol.md`](../benchmark_protocol.md).

### ADR-009: Freeze connected query groups before retrieval evaluation
Date: 2026-11-26
Status: ACCEPTED for Company Policies v1

The split joins normalized exact query text, number-masked templates, near templates
(character-trigram set Jaccard >=0.9) and any normalized exact passage shared across query
records. All associated passages participate without consulting qrels. Groups are assigned
by a seeded SHA-256 bucket rule (70/15/15 target proportions). The ID-only manifest is
recomputed and checked before evaluation; different outputs need a new version. This
controls observed query/exact-passage reuse, without claiming exhaustive semantic-family
separation. No corpus deduplication or label rewriting occurs. M1 evaluates a fixed
100-query validation selection and refuses test configs; test is reserved for later frozen
comparisons. See [`m1_bm25_report.md`](../m1_bm25_report.md).

### ADR-010: Independent CPU BM25 and file-based run manifests
Date: 2026-11-26
Status: ACCEPTED for M1

Use rank-bm25 0.2.2 with explicit default Okapi parameters and raw `str.split()` to match
the reference's sparse preprocessing, with deterministic score-descending/ID-ascending
ties. Search the entire domain, not per-query attached candidates. Save run/qrel files,
per-query metrics, reference-formula diagnostic, fixed-rule failure examples, configuration,
source/dependency hashes and timing scope. Existing outputs cannot be overwritten.
MLflow is deferred until multi-model comparisons in M2; manifests remain the portable
record even when tracking is added. Raw text/qrels remain ignored while exact data terms
are unresolved. Revisit tokenization/models on train/validation only.

### ADR-011: Optional local SQLite MLflow with immutable file evidence
Date: 2026-11-26
Status: ACCEPTED at the user's explicit request after M1

The `tracking` extra pins MLflow 3.18.0. Explicit low-level clients write to a project-local
SQLite database and file artifact store under ignored `artifacts/mlflow/`, independent
of the UI process or remote tracking environment variables. A wrapper serves that store
on loopback only with a single worker and disabled SDK/UI telemetry. MLflow owns its
schema; application persistence remains unimplemented.

BM25 tracking is opt-in and runs after evaluation/output, outside retrieval timing.
File manifests remain the portable evidence. Import validates all original output hashes,
copies an explicit artifact allowlist, preserves original Git/source/package provenance,
and records historical measurement date separately from MLflow import clock timestamps.
Finished imports are idempotent by manifest fingerprint; different historical dates are
rejected. Tracking failure leaves completed research files intact and returns an error
for explicit tracking requests. Local raw failure text is not duplicated into MLflow.
This implements the tracking previously deferred in ADR-010; no remote account or model
execution is required. See [`mlflow.md`](../mlflow.md) for actual commands and scope.

### ADR-012: Pinned offline GPU E5 and exact FAISS cosine index
Date: 2026-11-26
Status: ACCEPTED for M2 after explicit GPU-download authorization

Use multilingual-e5-small revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` (MIT model
card), with checksum-verified safetensors/tokenizer assets and no remote code. The fetch
command is the sole model-download path; inference uses local files only and refuses
silent CPU fallback. Follow the author's query/passage prefixes, masked-mean pooling and
L2 normalization. Initial experiment is FP32/eager/TF32-off on one CUDA GPU, max 512 tokens.

FAISS CPU 1.16.0 IndexFlatIP provides exact cosine retrieval of normalized embeddings;
full scores are ordered by score then ID to resolve cutoff ties. Cache identity covers
corpus IDs/text, model/config/assets, encoder/cache source hashes, device, FAISS and the
environment lock hash. Cache arrays use non-pickle NumPy formats and verified checksums.
This chooses an existing index rather than writing a new search engine; an approximate
index or another precision/model requires a separately recorded experiment.

### ADR-013: Common evaluation and conservative insertion-time diagnostic
Date: 2026-11-26
Status: ACCEPTED for M2

Reuse the evaluator through a retrieval function receiving only query text, cutoff and
query insertion metadata; qrels remain exclusively in evaluation. Method IDs distinguish
BM25, dense, equal-weight RRF (offset 60) and RRF with fixed insertion-affinity bonus 0.1.
Hybrid initially fuses two top-100 lists and returns 100; the insertion variant reranks
exactly those IDs. Signed insertion gaps/after-query flags are metadata features, not
validity labels. Missing dates stay unknown and add no bonus; no hard filter or event-time
inference is introduced. Later-inserted positives and official labels stay unchanged.

Compare fixed validation IDs with paired per-query deltas/bootstrap intervals, retain
regressions and record sequential timing separately from offline embedding/index costs.
All four local MLflow entries use the same manifest format as M1. The observed dense and
time-bonus regressions are retained; no parameters are retuned to manufacture gains.
Freeze one shared candidate pool before any M3 fast/deep reranking comparison.

### ADR-014: Fixed frozen-pool cross-encoder before adaptive routing
Date: 2026-11-26
Status: ACCEPTED for M3 infrastructure; MiniLM not adopted as the deep path

Freeze the original hybrid top-100 IDs/order/scores and pin its complete source manifest.
Reject data/split/config/qrel drift or malformed candidates before model work. Extend the
common evaluator with an optional ID-aware callback so identical query texts cannot
misjoin candidate pools; no relevance labels reach retrieval/scoring callbacks. Default
warmup remains train; frozen runs explicitly select and record a development warmup ID.

Use mmarco-mMiniLMv2-L12-H384-v1@1427fd652930e4ba29e8149678df786c240d8825 as one compact
fixed baseline: six checksum-verified local assets, safetensors, no remote code, CUDA,
FP32/eager/TF32-off, max 512, batch 16. Korean is not a documented supervised mMARCO
language; report transfer limitations rather than promising Korean quality. Rank all
original pairs by raw relevance logit/ID and retain every score as immutable file evidence.

Measure synchronized incremental reranking and CPU preprocessing/forward-transfer wall
components with the encoder unloaded. Separate these from earlier fast retrieval timings;
no live end-to-end p95 or simultaneous-model VRAM claim follows. Two GPU runs produced
identical scores but worse mean quality than hybrid, with seven query-level gains and
33 regressions. Retain the negative result and MLflow evidence. Check another explicitly
Korean-capable reranker on the same pool before M4; do not train a router on validation
outcomes. See [M3 report](../m3_reranker_report.md).

### ADR-015: Explicit pinned Korean alternative in the existing reranking protocol
Date: 2026-11-26
Status: ACCEPTED for the M3 follow-up

After the negative MiniLM result, add one preselected Korean alternative,
`dragonkue/bge-reranker-v2-m3-ko@2aca5884ecac490192af9ebd86836d9073d826cd`.
The author declares Korean optimization and Apache-2.0; complete fine-tuning data is
undisclosed, so benchmark decontamination is not independently established. Treat its
quality as an experiment, not a guaranteed improvement. Keep the original negative run.

A small explicit ModelSpec registry shares the existing offline XLM-R classification
loader, scoring, checked fetch and telemetry. Config validation rejects an unsupported
model or a mismatched model/revision; the configured model selects the asset directory.
Both models use verified safetensors/local tokenizers, no remote code or extra libraries.
Expected position capacity is model-specific (514 / 8194); evaluation remains capped at
512 tokens for both. Compare identical hybrid pools/qrels and FP32/eager/TF32-off,
batch 16, concurrency 1. Model-specific tokenizers and capacity are disclosed.

No search over weights/budgets, router training, final test or data relabeling is part
of this increment. Measure the larger model's own incremental cost and allocator peaks;
separate model runs do not establish simultaneous residency or live pipeline latency.

Verified outcome: nDCG@10 0.163721 versus hybrid 0.207685 and MiniLM 0.109982;
11 gains/28 regressions against hybrid, mean incremental cost 1,148.100 ms. Replay
reproduces all scores exactly; 124 CPU tests pass. Neither model is adopted as the
always-deep path. Next: predeclared train-only utility pilot before learning a router.
See [Korean follow-up](../m3_korean_reranker_report.md).

### ADR-016: Train-only pilot with checkpointed pre-deep features
Date: 2026-11-27
Status: ACCEPTED for the first M4 increment

Predeclare 100 train queries with the existing split/hash selection, unchanged hybrid
and Korean cross-encoder, feature definitions, lambda penalties and diagnostic controls
before collecting outcomes. Do not fit a router or evaluate held-out labels yet. Save
source/group/selection/config provenance before GPU work, and checkpoint feature/fast
manifest hashes before deep inference. Keep inference features and outcome targets in
separate files; strict feature schemas reject outcome/ID columns. Train-only target
construction refuses validation/test IDs or mismatched joins/candidate coverage.

Reuse the common evaluation and frozen-pool runner. Fast latency includes pre-deep feature
extraction; deep latency is already incremental, so do not subtract fast time again.
Targets use gain > lambda * incremental seconds for fixed 0/0.01/0.05 nDCG/second penalties.
Fixed disagreement/year rules and seeded random query budgets are descriptive controls.
Outcome-aware positive-gain routing is a clearly marked offline training oracle bound,
never an inference feature or learned-policy result. Policy cost sums are modeled stage
costs, not live pipeline p95. Date/text diversity are proxies, not true version chains.

Models run sequentially. Verify encoder model collection by weak reference before loading
the reranker; a zero allocator counter is not a valid removal requirement because runtime
workspaces can persist. Retain residual allocation telemetry. Initial fast-only attempt
stopped on that overly strict check and remains untouched; subsequent runs record the
corrected guard and their own source snapshots. See [PyTorch 2.10 CUDA semantics](https://docs.pytorch.org/docs/2.10/notes/cuda.html#cublas-workspaces).

Local MLflow retains standard fast/deep evidence and an explicit allowlist of pilot
features/outcomes/config/summary/checkpoint/manifest. Raw failure text is not uploaded.
No new dependencies, training run, raw-data redistribution or held-out evaluation is
introduced. See [pilot protocol/report](../m4_train_pilot_report.md).

### ADR-017: Frozen train500 collection and fold-local logistic routing
Date: 2026-11-27
Status: ACCEPTED for the first learned-router experiment; advantage unestablished

Run a predeclared 500-query GPU collection from a clean detached `30945ff` to prevent
router development from changing the collection's source snapshot. Keep immutable data
and router provenance separate, with exact archive/split/config/code/checkpoint/group
checks on loading. The same numerical packages are retained; explicit `routing` extra
pins sklearn 1.9.1. No reranker training or new model download is introduced.

Use five shuffled GroupKFold splits, seed 1729, fold-local median imputation plus missing
indicators/empty-column preservation, scaling and balanced LBFGS logistic C=1. Fixed
full/no-explicit-temporal feature sets and penalties 0/0.01/0.05 are predeclared; primary
is full/0 at a 10% query cap. Rank quota scores only inside each held-out fold: global OOF
ranking could indirectly depend on held-out labels through other models. Decisions never
receive gold, actual rerank cost or IDs as learned features. Random/agreement controls
match query counts per fold. Outcome-derived targets remain offline train artifacts.

Store validated version-1 JSON preprocessing/linear parameters, with independent numerical
checks against sklearn, instead of pickle. A single-class fold has an explicit recorded
constant fallback. An all-selected-train primary model is saved for later validation;
only cross-fitted scores count as OOF evaluation. Group intervals condition on existing
OOF fits without refitting; stage-cost sums/CPU microbenchmarks are not live service p95.

500 queries / 493 groups produce primary nDCG@10 0.198929 with 49 deep calls versus
hybrid 0.197016 and agreement-budget 0.204155. Primary contrast intervals include zero;
there is no learned-quality superiority claim. Exact CPU model/score replay and 157 CPU
tests are verified. Retain all sensitivity results and freeze policies for validation
before considering a new method. See [report](../m4_router_groupcv_report.md).

## Decision entry template

```text
### ADR-XXX: <short title>
Date: YYYY-MM-DD
Status: PROPOSED | ACCEPTED | SUPERSEDED
Context:
Options considered:
Decision:
Evidence / benchmark / issue:
Consequences and reversal plan:
```

Do not mark provisional decisions accepted solely because a code agent likes a particular technology.
