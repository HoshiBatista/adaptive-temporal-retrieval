# Initial benchmark protocol — verified M0–M2

Status: metric foundation, frozen grouped split and CPU BM25 are implemented and tested.
The [M1 development report](m1_bm25_report.md) contains measured validation-subset results;
no final test or official whole-benchmark scores exist.
The [data card](dataset_card.md) records the pinned corpus and observed limitations.
The [M2 report](m2_retrieval_report.md) adds GPU dense, hybrid RRF and an insertion-only
diagnostic on the identical selected validation IDs, with paired deltas and regressions.

## Evaluation unit and candidates

- Evaluate each record/query, with binary positives from `gold_docs_final_global`.
- Aggregate per-query values with an arithmetic mean. Empty-qrel policy in pure metric
  functions is zero; a future dataset adapter must report such queries explicitly.
  The audited domain currently has none.
- Retrieve over the entire declared corpus, not just the ten passages attached to a
  query. Any corpus subset must be identified as a local subset and retain required qrels.
- Keep official IDs and labels immutable. Never deduplicate texts in a way that silently
  changes the ID-level relevance task.
- Candidate budgets are fixed per experiment. Fast/deep reranking comparisons share
  cached candidate IDs; distinct retrieval methods can produce different candidates.

## Metric definitions

`src/temporalproof/evaluation/metrics.py` implements two pure functions:

- **Recall@K:** retrieved distinct positive IDs / all positive IDs for the query.
- **nDCG@K:** binary DCG with `1/log2(rank+1)` discount; ideal DCG contains
  `min(K, number of all positive IDs)` positives, including unretrieved positives.

K must be a positive integer. Duplicate returned IDs are rejected rather than counted
multiple times. Empty rankings and empty qrels yield zero. Metrics consume an ordered
ranking; score ties must be resolved deterministically by the future retriever and saved
in its manifest. No graded relevance, temporal-coverage gold or generation metric is
invented. Primary planned cutoff is nDCG@10; Recall cutoffs are frozen with M1 configs.

The [reference metrics](https://github.com/kaist-dmlab/TimelyRAG/blob/5213a0c1743f23fd52a258005c3887aaf7e8cd0f/metrics.py)
normalize nDCG using positives within the retrieved prefix. With gold `{10, 11}` and
ranking `[10, 99]` at K=2, that formula gives 1.0; the qrel-based definition gives
approximately 0.613147. Reference AP also normalizes by retrieved positives. An official
reproduction must label those metrics explicitly, alongside the main qrel-based metrics;
never merge their result columns. Reference files are inspected locally without alteration
or execution and are not vendored into source control.

The reference [best-alpha branch](https://github.com/kaist-dmlab/TimelyRAG/blob/5213a0c1743f23fd52a258005c3887aaf7e8cd0f/retrieval.py)
selects alpha using the current query's qrels. Treat it as an offline oracle diagnostic.
The automatic-alpha branch can be assessed separately as an inference-available baseline.
The reference retriever initializes BGE-M3 even for BM25; our M1 BM25 implementation
must work independently on CPU. No published score has been reproduced in M0.

## Splits and temporal policy

M1 freezes `data/splits/company_policies_v1.json` using seed 1729. Connected groups keep
normalized exact queries, exact number-masked templates, near templates (character-trigram
set Jaccard >=0.9), and cross-record normalized exact passages together. All passages
participate without qrels. See the report for edge counts and residual family/paraphrase
risks. Group hashing assigns 2,058 train / 495 validation / 447 test queries. M1 evaluates
100 validation queries selected by a seeded hash, not test. The runner verifies grouping,
coverage and assignments against the pinned corpus before each run. Inspect schema and
metadata without using test outcomes for model selection; retain test for frozen comparison.

The initial retrieval baseline retains original qrels and insertion metadata, without
hard timestamp filtering. This does not claim historically available or legally effective
evidence. Explicit event times are absent and 1,253 positives are later-inserted than
their query. Future known-at experiments must disclose eligible corpus rules and gold
coverage; they must not quietly remove or replace positives. Time parsing/normalization
changes require a versioned derived transformation and a new manifest.

## Validation already available

```bash
uv run --frozen pytest -q
uv run --frozen temporalproof-audit --jsonl tests/fixtures/timelyqa_synthetic.jsonl --output artifacts/audits/synthetic.json
uv run --frozen temporalproof-audit --sample-size 50
```

Tests cover malformed/missing timestamps, schema and ID failures, local/global gold
mapping, duplicate query grouping, future-positive diagnostics, checksum/size protection,
offline defaults, sanitized errors, deterministic reports and metric counterexamples.
The fixture was authored for tests; it contains no downloaded benchmark examples.

## M1 outputs now available

`temporalproof-prepare-split` regenerates/verifies the committed ID-only split.
`temporalproof-bm25 --output <new-directory>` runs library BM25 with raw `str.split()`,
fixed parameters, score-descending/document-ID-ascending ties, full domain corpus and top
100 candidates. It emits manifests, per-query metrics, TREC rankings, binary qrels,
summary and failure cases. Reference-formula nDCG is labeled separately. Query/bootstrap
seeds, code/dependency/config hashes, selected IDs, thread settings and environment are
recorded. Both commands are offline; existing differing splits/runs are never overwritten.

The first 100-query validation run and replay have measured quality and sequential query
timings; the report states exact scope. M2 comparisons now have measured query timings
and PyTorch allocator peaks, separately from offline embedding/index construction.
Official full-benchmark reproduction, final test outcomes, load-test p95, throughput and
total process peak VRAM remain **TBD**.

## M2 fixed comparison

The new compare command reuses the common evaluator with a retrieval function accepting
only query text, cutoff and query insertion metadata, never qrels. Encoder assets/revision,
prefixes, pooling, truncation, precision, FAISS version, source/dependency hashes and cache
provenance are recorded. All variants return 100 IDs; hybrid fuses sparse top 100 and dense
top 100 with equal RRF weights and offset 60. Its insertion-time variant reranks the exact
same candidates using min-max RRF plus fixed weight 0.1 insertion affinity. Unknown dates
add no bonus, no future documents are dropped and no applicability label is invented.

Parameters are frozen before observing M2 scores. Paired bootstrap uses aligned query
IDs and seed 1729; conditional intervals do not account for group dependence. A higher
sample mean with an interval containing zero is not a significance claim. Repeat runs
check identical rankings, qrels, metric means and original M1 ID/split parity. Dense/hybrid
retrieval pools can differ; M3 fast/deep comparisons must freeze a shared candidate pool.

## M3 frozen-pool reranking

The M3 config pins its M2 hybrid manifest, not only a directory path. All source output
hashes, selection/config, data/split, exact 100 unique known IDs/order and qrels are checked
before GPU inference. The common evaluator has an ID-aware callback for frozen pools;
the cross-encoder itself receives only query/passage texts. Duplicate query texts cannot
silently share a different query's candidates. Scores are saved for every original pair.

Pinned multilingual MiniLM runs FP32/eager/TF32-off, batch 16, max 512, longest-first pair
truncation, no metadata features. Korean is a transfer language absent from its documented
mMARCO supervised languages. The observed negative result is retained. Recall@100 is
verified unchanged per query. Paired deltas and regressions use the unchanged M2 protocol.

Latency is incremental rerank-only on frozen candidates, including synchronized GPU work
and CPU preprocessing/sorting, with concurrency 1 and three selected-query warmups.
Fast M2 latency is historical context, not a current live pipeline measurement. Do not
add separate p95 values or claim concurrent model capacity from separate model runs.
Allocator peaks are not total process VRAM. See [M3 report](m3_reranker_report.md).
Router targets were not generated; any future targets require train-only outcomes.

The [Korean M3 follow-up](m3_korean_reranker_report.md) adds one preselected pinned
bge-reranker-v2-m3-ko model with the identical source manifest/pool, qrels and runtime
settings. The small explicit registry validates model/revision pairs and chooses the
matching local asset directory. Model-specific tokenizer hashes and position capacities
are recorded; both evaluations remain capped at 512. Raw logits preserve relevance order
without sigmoid saturation ties. Full fine-tuning data is undisclosed, so contamination
is not independently excluded. The larger model still regresses versus hybrid; no
additional variants, reweighting or relabeling were used to hide the result. Before
learning a router, measure fast/deep utility on a predeclared train-only pilot.

## M4 train-only utility pilot

The new pilot config predeclares 100 train queries, unchanged hybrid/Korean models,
15 pre-deep feature fields, lambda penalties 0/0.01/0.05 nDCG per second, disagreement
threshold 0.2 and query-fraction caps 0/0.1/0.25/0.5/1. Config/source/selection/full-group
identities are saved before GPU work, then feature/fast hashes before reranker work.
Outcome construction accepts matching unique train IDs only. Features and targets are
separate files; strict feature schemas reject labels, deep scores and query/document IDs.

Targets use measured incremental rerank time directly, with no second subtraction of
fast time. Fast includes feature extraction. Fixed rules and one seeded random choice
are in-sample controls, not fitted/held-out results. Gold-dependent oracle routing is an
explicit offline upper bound under at-most-query-count budgets. Its outcomes never enter
inference features. Policy costs sum separately measured stages; no combined p95 claim.

The completed 100-query train pilot has 15 gains/18 regressions and 45 empty-gold pools.
The fixed disagreement rule routes 64 queries with a small mean gain whose paired interval
contains zero, at high cost. True version chains, historical known-at evaluation, fitted
routing, final test and domain transfer remain pending. Replay matches all scores,
features, quality/gains and labels at all three lambda values. See
[M4 pilot protocol/report](m4_train_pilot_report.md) and its aggregate/analysis scripts.

## M4 learned router: grouped train500 evaluation

One fresh 500-query GPU collection from pinned clean `30945ff` supplies features/targets;
its initial 100-query scores match the earlier pilot exactly. No full GPU replay is claimed.
The separate router method uses sklearn 1.9.1, five label-independent GroupKFold splits,
fold-local median/indicator/scaler fitting, balanced LBFGS C=1, and predeclared full/
no-explicit-temporal variants at three penalties. Primary operating point is full/0 and
an at-most-10% query cap applied inside each held-out fold. Actual calls=49/500 because
of per-fold flooring. Never allocate a global budget across OOF models or use measured
deep costs at decision time; cross-fold training can contain another fold's labels.

Quality uses the same frozen fast/deep rankings; all source groups are kept intact.
Query-weighted paired group bootstraps resample source groups with fixed OOF predictions,
not model refits. Modeled stage sums include separately measured CPU scoring but exclude
quota allocation overhead; no live p95/service claim. At matched query counts, time costs
can differ by selected query. JSON inference/state replay and held-out-label/outlier
counterfactual tests guard fold isolation. See [full report](m4_router_groupcv_report.md).


## Native TimelyRAG code comparison (M4.3)

A separate fixed validation495 experiment uses original reversed NumPy argsort ties,
not the previous local doc-ID tie policy. All candidate/score maps are preserved within
paired baseline/auto methods; full-qrel nDCG and official prefix-normalized metric columns
remain separate. The generic runner requires an explicit callback for native ties.
Original auto code parity and full cached replay are checked; gold-dependent best-alpha
is not a real-query baseline. BM25 is a matched local code reproduction, E5/hybrid are
GPU-backed transfers with explicitly substituted backbone; published whole-benchmark
results are not reproduced. Two GPU collections preserve scores/ranks/alpha/metrics.
No qrels/time applicability labels are modified; source unknown .5 affinity is not a
validity assertion or hard known-at filter. See [report](m4_timelyrag_reference_report.md).

## Query-date role audit and frozen intervention (M4.4)

The audit applies only to train2058 and frozen train500 pools; validation/test role parsing
and new retrieval evaluation are not run. A fixed40-group original/derived inspection
informs a conservative text-only policy. Selected267 points /1791 fast bypasses and1478
source employment-cued selections are heuristic coverage, not gold/accuracy/error rate.
Missing years/calendar uncertainty and absent document event metadata are not repaired
from ingestion time. Year/month extents describe resolution, not applicability intervals.

The next train500 experiment is predeclared QET-only intervention versus unchanged fast
and native auto, with explicit fast bypass on abstention. Preserve pool/scores/qrels,
query/granularity helpers, source alpha/gates, document fallback and native ties. Verify
native parity, report paired group full-qrel deltas and separate prefix metrics, and pass
the train gate before new validation. See [audit report](m4_query_time_audit_report.md).

## Frozen train500 QET intervention (M4.5)

The M4.4 parser and contract are unchanged; fixed source pools/scores/qrels, original
query/granularity helpers, alpha/gates and insertion fallback. Native NumPy ties use
original corpus order **restricted to the100 frozen candidates**. Outside-pool scores
are not reconstructed, and unchanged fast retains its old tie order. A native-fast tie
control and native-auto/same-bypass control are declared before scoring, alongside fast/
native/role-QET methods. Native parity stops before metrics/oracle with no benchmark gold.

Primary role-QET−fast mean−0.002570, source-group95% interval[−0.007591,0.001085],8/9/483.
The positive-mean/positive-lower-bound train gate fails, so no new validation/test or
adoption. Same-bypass control exposes an unestablished date-only gain, rather than
attributing the whole improvement over weak native auto to role extraction. Keep the
train-engineered/development caveat, all negative results, full-qrel versus prefix columns,
unknown document event times and unchanged candidate Recall100. See
[report](m4_qet_ablation_report.md) for replay, latency boundaries and failure inspection.
