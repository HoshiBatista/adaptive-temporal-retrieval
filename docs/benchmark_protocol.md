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
