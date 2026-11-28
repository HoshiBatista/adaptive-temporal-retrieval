# Research protocol — Adaptive Semantic–Temporal Reranking (ASTR)

**Status:** M0–M3, utility pilots, grouped-OOF logistic routing and frozen development validation are measured.
Measured validation-subset results are in [`docs/m1_bm25_report.md`](../m1_bm25_report.md);
no final test or official whole-benchmark scores exist. Observed schema and limitations are recorded in
[`docs/dataset_card.md`](../dataset_card.md); initial metric decisions are in
[`docs/benchmark_protocol.md`](../benchmark_protocol.md). In particular, Company Policies
has no explicit event-time fields, 47 unparseable document insertion strings and 1,253
later-inserted positive documents. Do not assume clean validity intervals or enforce a
known-at filter against unchanged qrels without a separate documented experiment.
M2's [dense/hybrid report](../m2_retrieval_report.md) retains the fixed validation selection,
reports paired deltas including regressions and tests insertion metadata only. M3 adds a
fixed cross-encoder on frozen candidates and measured incremental cost; see the
[M3 report](../m3_reranker_report.md). Frozen train500 routing on all 495 development validation queries regresses against hybrid:
0.190746 versus 0.198972 nDCG@10, paired group interval below zero. No validation
refit/threshold tuning; see [validation report](../m4_router_validation_report.md). The set
includes prior model-selection validation100. Final test, valid-time applicability and
independent transfer evaluation remain pending.

## 1. Primary research question

Can a learned router that uses query-time and initial retrieval signals achieve a better retrieval quality–compute trade-off than fixed neural reranking for an evolving knowledge base?

Two related questions:

- Do temporal features improve retrieval compared with semantic-only rankers?
- Can a cheap decision policy identify requests where deep reranking adds meaningful nDCG/Recall gains?

Null hypothesis: routing does **not** outperform a simple fixed policy at matched cost/quality. Negative findings are acceptable if documented.

## 2. Dataset inventory (verify before use)

| Source | Intended role | What must be checked before relying on it |
|---|---|---|
| [TimelyRAG / TimelyQABench](https://github.com/kaist-dmlab/TimelyRAG) | Primary training and evaluation of time-aware retrieval | Official splits, relevance judgments, queries grouped by ID, event vs ingestion timestamps, dataset license, reference scripts |
| [TEMPO](https://github.com/tempo-bench/Tempo) | Optional independent/transfer evaluation for challenging temporal retrieval | Download access, exact corpus/query schema, license, golden passages/steps, metric definitions, model/LLM judge dependencies |
| [TimelyRAG paper](https://aclanthology.org/2026.findings-emnlp.398/) | Related work / method comparison | Compare on same data/eval; historical published scores alone are not apples-to-apples baselines |

**Never assume a dataset is public-domain or commercially usable just because it is on GitHub.** Confirm terms before redistributing records or weights. Keep raw datasets outside source control.

### M0 data audit output

Write `docs/dataset_card.md` (or update the existing file) with:
- source, URL, download date, source commit/revision, checksum, licensing notes;
- data sizes verified by counting records and files;
- **observed** field names and types, timestamps/time zones and missing-value rates;
- query and document IDs, relevance grading and whether labels support document or passage retrieval;
- split policy, duplicate overlap/temporal adjacency risks;
- sample record structure with personal/sensitive content removed;
- expected memory/disk requirements for the chosen subset.

Do not silently convert event time to publication time or infer missing `valid_until` dates without documenting assumptions.

## 3. Initial experiment sequence

| ID | System | Trained? | Mandatory? |
|---|---|---|---|
| B0 | Dataset-provided reference scoring and metric smoke tests | No | Yes |
| B1 | BM25 sparse retrieval | No | Yes |
| B2 | Compact dense embeddings + vector similarity | No / optional fine-tune | Yes |
| B3 | Hybrid sparse+dense fusion | No | Yes |
| B4 | Explicit temporal filtering / feature boost | No | Yes |
| B5 | Fixed deep cross-encoder rerank over top-K | Optional | Yes after B1–B4 |
| B6 | Official TimelyRAG method | Depends on implementation | Reproduce when feasible; report parity gaps |
| A1 | Rule-based adaptive route | No | Yes |
| A2 | Learned route: logistic regression or CatBoost | Yes | Main contribution |
| A3 | Ablations of time features, uncertainty, version diversity and router inputs | Yes | Yes |

Start on a small deterministic query/corpus subset. Do not report subset metrics as official whole-benchmark scores. Never perform train-time selection using the final test set.

## 4. Offline router target construction

For each **training** query, evaluate a frozen fast system and the frozen deep reranker on the same candidate pool. Let:

- `gain = metric(deep_ranking, gold_train_qrels) - metric(fast_ranking, gold_train_qrels)`;
- `extra_cost = measured_deep_latency - fast_latency` (with a documented profiling protocol);
- a candidate training target be `gain > lambda * extra_cost`, where `lambda` is selected only on training/validation folds.

**Critical:** such targets may depend on gold judgments and post-hoc deep scores **during offline training**. **Online inference router features must depend exclusively on raw query/candidate metadata and fast-stage signals**. Do not include `gain`, `deep score`, gold grade, correct rank, or future version data as inference features.

Use out-of-fold model predictions for any learned feature or preselection model that could otherwise memorize training labels. Build route targets only within the proper split. Cache all per-query fast and deep outputs with versioned manifests.

## 5. Temporal relevance safeguards

- Differentiate requested **valid-at/event time** from **known-at/ingestion time** when defined. These may be two distinct axes.
- Specify whether boundary intervals are `[valid_from, valid_until)` or another convention, and test both endpoints.
- Maintain stable doc version IDs and preserve earlier revisions.
- For multi-period queries, evaluate whether each period is covered; do not merge evidence as if all periods were simultaneous.
- For missing time metadata, represent `unknown`, not an inferred true label.
- Check whether the benchmark's “positive documents” actually validate temporal applicability rather than just topical relevance.

## 6. Split and leakage policy

- Prefer benchmark-author splits if they suit the task. Otherwise split by **query/group**, not by random query–document row.
- Audit repeated question templates, duplicated content, same version chains and nearly identical revisions across train/test.
- If testing historical knowledge, enforce `published_at <= query_known_at` when the task models information available *at that time* (unless benchmark intentionally defines retrospectively known effective dates).
- Use a single held-out test set exactly once for the final model comparison; iterate on train/validation.
- Feature design, threshold selection, reranker top-K and router budget are fixed before final test evaluation.

## 7. Metrics and reporting

**Retrieval:** nDCG@10, Recall@K (define positive grades), MRR where relevant. Group by query, not by pair. Include temporal-hard-query slices, if identifiable without manual gold creation.

**Routing:** achieved nDCG vs mean reranker cost and p50/p95 latency, percentage routed to deep path, recall of queries that genuinely gain from deep reranking (diagnostic only). Include quality–budget curves.

**Operations:** corpus indexing duration, storage size, peak VRAM, p95 under fixed concurrency, request failures, throughput, and model memory footprints. Separate ingestion time, query runtime, and optional generation.

**Statistics:** per-query paired deltas; bootstrap 95% confidence intervals; same random seeds, hardware and corpus/index settings whenever possible.

**Results policy:** use `TBD` until measured. Save exact configs, package/model versions, source revisions, qrel definitions, query identifiers, random seeds, hardware descriptions and raw per-query outputs. No cherry-picking impressive cases.

## 8. Ablation questions

1. Does temporal metadata help without the learned router?
2. Does BM25+dense agreement predict reranker usefulness?
3. How much does version diversity contribute?
4. Can a simple threshold compete with CatBoost?
5. Does the route policy transfer between domains?
6. Are efficiency gains preserved when measured as **real p95 latency** rather than model FLOPs?

## 9. Failure / fallback criteria

- If dataset access or labels do not support the proposal: preserve the audit and revise the research question with the thesis owner before engineering an invented replacement dataset.
- If the learned router does not beat a simple policy: report negative results; focus on the independently evaluated semantic–temporal retrieval method and error taxonomy.
- If the temporal features are not reliable: use an explicit metadata coverage report and ablate them; never infer ground-truth validity silently.
- If remote APIs/LLM judges are unavailable: use reproducible document-level retrieval metrics; do not fabricate answer-quality outcomes.

## M3 measured development result and routing gate

The fixed MiniLM cross-encoder on frozen hybrid validation100 candidates regresses in
mean nDCG@10 (0.207685 → 0.109982), with 7 gains/33 regressions and ~150 ms incremental
rerank cost. Preserve this negative baseline. The multilingual backbone alone does not
establish Korean relevance quality; Korean was absent from documented supervised mMARCO
languages. The separately pinned Korean BGE follow-up on identical candidates reaches
0.163721, still below hybrid, with 11 gains/28 regressions and ~1.15 s mean incremental
cost. Both negative always-deep comparisons are retained. The resulting first train-only
utility pilot is measured below; no learned policy has been fitted yet. These validation
outcomes are never router-training labels. Score caching and allocator/stage telemetry do not establish live pipeline p95,
concurrent model residency or service throughput. See [M3 report](../m3_reranker_report.md).

The [Korean follow-up report](../m3_korean_reranker_report.md) documents paired comparisons,
exact-score replay, model-specific tokenizers and the undisclosed fine-tuning-data caveat.
No independent benchmark-decontamination claim is made. No further model/weight sweep,
final test or routing features derived from these deep outcomes were introduced.

## First M4 train-only pilot measured

The predeclared 100-query train pilot now has checkpointed pre-deep features and separate
outcome/cost targets, with exact-score/feature replay. Its 15 positive gains are not a
held-out routing result. Untrained rules/random budgets are descriptive controls; an
outcome-aware upper bound is explicitly non-deployable. Cost penalties use incremental
rerank seconds directly; modeled stage sums do not establish live pipeline latency.
See [pilot report](../m4_train_pilot_report.md). Next use a larger predeclared train sample
with grouped out-of-fold feature preprocessing/model fitting before comparing on validation.
Do not treat earlier validation outcomes or query/document IDs as router input features.

## First learned router: train500 grouped OOF

The predeclared 500-query collection uses the clean committed collector `30945ff` while
router development has a separate source snapshot. Loader checks pins/hashes/canonical
train groups and outcomes; no held-out labels enter training. Five group folds independently
fit median imputation, missing indicators, scaling and balanced logistic regression (C=1).
No hyperparameter/threshold search occurs. Full/no-explicit-temporal feature sets and
0/0.01/0.05 penalties are predeclared sensitivity variants, with full/0/10%-query-cap primary.

Quota decisions are ranked within the held-out fold, not globally across OOF scores:
other fold models may have seen this fold's labels. Controls have identical fold query
caps, not exact time budgets. Cached JSON inference excludes outcomes/IDs; safe parameter
states reproduce sklearn predictions without pickle. Raw gain/cost remains offline only.

Primary OOF nDCG@10 0.198929 versus hybrid 0.197016 and agreement-budget 0.204155 gives
no established gain; all primary paired group intervals contain zero. Group bootstraps
condition on fixed OOF predictions and do not refit. AUC is secondary and scores are not
calibrated probabilities. Original negative results/secondary regressions are preserved.
An all-500-selected-train primary model is saved for future validation, not evaluated on
its own training predictions as OOF. Next freeze the learned/control policies for validation;
final test remains untouched. See [report](../m4_router_groupcv_report.md).


## M4.2: Conservative rank blending after negative development validation

This hypothesis was explicitly motivated by earlier validation failures, not independent
confirmation. Fix reciprocal offset60 and weights0/.05/.10/.20/.30 before experiments;
select each held-out fold's weight from other **training** groups only. Freeze the final
all-train weight before validation access. Zero preserves fast, any positive weight
requires the full expensive pool. Keep raw score scale out of rank fusion.

Grouped OOF0.196354 versus fast0.197016 and validation .20-blend0.196351 versus fast0.198972
do not establish an improvement. The old logistic routes + blend reach0.199371, but this
is secondary, inconclusive against fast, and not retrained for the changed utility.
Do not silently reuse old raw-deep gain labels to claim a newly trained blended router.
Future fitting must derive targets on train only and select blend weights within folds.

No/some/all positive-pool coverage is211/147/142 (train500),211/152/132 (validation495).
Within-pool qrel oracle ceilings are offline diagnostics, never inference features or
published-system results. Unjudged zero grades do not establish irrelevance; illustrative
manual case observations are not new gold labels or validated temporal-hard groups.
Keep dataset text local and original labels unchanged. Cached stage cost sums omit
unmeasured new fusion/router overhead. Exact replay and188 CPU tests verified; see
[report](../m4_rankblend_report.md). No new neural inference or final-test evaluation.


## M4.3: Matched local native auto-alpha reference

The reviewed TimelyRAG code is pinned separately; original source/evaluation files stay
unmodified outside Git. Native auto top10 parity holds for36 synthetic cases and all1485
real query/backbone pairs, stopping original real-query execution at the first metric
boundary before its oracle. No real best-alpha outputs are generated. Candidate pools/
qrels and source heuristic coefficients remain unchanged; no fitted targets or new model.

Use native reversed NumPy argsort for this comparison. Full-qrel nDCG is primary;
reference-prefix nDCG/MAP are separately named. Generic BM25 requires an explicit native
callback before it can claim this tie policy. Comparisons with past local doc-ID ties
must expose differences. E5 uses GPU; native BM25 and cheap temporal helpers use CPU.

On validation495, native BM250.154621→0.146812 and E5 .141065→.131546 have group intervals
containing zero; hybrid .198335→.173203 has a negative group interval. Do not tune alpha
on validation to replace these findings. Two GPU runs and full cached CPU replay verify
scores/ranks/alpha/metrics. Paper whole-benchmark/BGE-M3 scores are not reproduced.

Query event extraction491/495 is heuristic finest/first-expression selection, not role-aware
validity gold. Document event times absent;47 insertion strings unknown. Source later/equal
insertions become unknown distance with neutral .5 affinity and remain candidates. Preserve
unknowns and labels; no historical known-at claim. Next audit event-role/fallback inputs on
train before further routing. Details and exact commands: [report](../m4_timelyrag_reference_report.md).
