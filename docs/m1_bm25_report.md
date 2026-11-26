# M1 — Company Policies CPU BM25 baseline

Measured **2026-11-26**. Development results on **100 validation queries** against the
full **30,000-passage Company Policies corpus**. These are not official full-benchmark
scores or final held-out test results. No model/threshold/tokenizer search was performed.

## Frozen inputs and method

- Dataset revision/checksums: [data card](dataset_card.md).
- Seed: **1729**; [frozen split](../data/splits/company_policies_v1.json), SHA-256
  `4aab958538041f019363a8ffcc5c5d15d030ebe408d869c17063bf9c3c566fa7`.
- Configuration: [bm25_company_policies.json](../configs/experiments/bm25_company_policies.json).
- Library: `rank-bm25==0.2.2`, `BM25Okapi`, `k1=1.5`, `b=0.75`, `epsilon=0.25`.
  [Upstream implementation](https://github.com/dorianbrown/rank_bm25).
- Raw Python `str.split()` tokenization on queries and document text; no case folding,
  punctuation stripping, stemming, Korean morphological analysis, title concatenation or
  text truncation. This follows the pinned TimelyRAG BM25 tokenization/default parameters.
- Score-descending ranking, document-ID ascending ties. This deliberate deterministic
  tie rule differs from the reference's reversed NumPy argsort.
- Return top 100 unique IDs. All original passages, positive IDs and dates are retained;
  there is no timestamp filtering, text deduplication or relevance-label input to retrieval.
- The corpus contains short synthetic policy passages rather than a reconstructed
  document/version hierarchy. No new chunking is applied.

## Split and leakage audit

Connected groups join normalized exact query text, exact number-masked query templates,
near query templates (character-trigram set Jaccard >=0.9), and any normalized exact
passage text shared by associated query records. Numeric masking is used for grouping
only: each query's original text, dates and judgments remain distinct for evaluation.
All associated passages participate, not just positives. No outcomes/qrels choose groups.

The deterministic audit reports 24 exact-query matches, 29 template-only matches,
136 nonidentical near-template pairs, and 3 cross-record shared-passage pairs. Match
counts can overlap and are not independent numbers of merged groups. Result:
**2,891 connected groups**, largest group 20 queries.

Each sorted group's query IDs plus seed are SHA-256 hashed modulo 10,000: buckets
0–6,999 train, 7,000–8,499 validation, 8,500–9,999 test. Group-level target proportions
are 70/15/15; actual query counts are **2,058 / 495 / 447**. The runner reconstructs and
verifies the entire manifest before evaluation and refuses stale or changed assignments.

The development subset is the first 100 validation IDs ordered by
`SHA256("1729:query:<ID>")`, covering 98 connected groups. Its exact ordered IDs are
saved in the run manifest. The M1 runner accepts train/validation only; **test has not
been evaluated**. Full unlabeled corpus text is shared across partitions for indexing;
this is query-held-out retrieval, not corpus-held-out or temporal-held-out evaluation.

Exact-text and query-template links are controlled. Semantic paraphrases, near-duplicate
passages and true document-family relationships are not exhaustively identified; no
explicit family IDs are supplied. This is a conservative heuristic audit, not proof that
all possible family leakage has been eliminated. Domain transfer remains a separate test.

## Quality results

| Metric | Mean | Conditional query-bootstrap 95% interval |
|---|---:|---:|
| **Qrel-based nDCG@10** | **0.175959** | 0.122303–0.230367 |
| Recall@10 | 0.230000 | 0.166667–0.296667 |
| Recall@20 | 0.281667 | 0.211646–0.356708 |
| Recall@100 | 0.393333 | 0.315812–0.471667 |
| Reference prefix-normalized nDCG@10 diagnostic | 0.242762 | 0.176042–0.310019 |

Primary nDCG normalizes against all positive qrels, including unretrieved positives.
The reference-formula diagnostic has a different denominator; it is not a second score
for the same metric definition. See [benchmark protocol](benchmark_protocol.md).

Intervals use 2,000 query resamples, seed 1729; they are conditional on this selected
development subset. The 100 selected queries span 98 groups; this query bootstrap does
not model group dependence or
support population-level/significance claims. No model comparison has been made yet.

**88/100 queries** have incomplete gold coverage at top 10; **45/100** have no gold ID
at top 100. For those 45, reranking this same pool cannot recover any judged-positive
document. Candidate recall is therefore an immediate issue to examine in M2.
Twenty-seven evaluated queries have at least one later-inserted positive; this is a
metadata diagnostic, not a newly labeled temporal-error subset.

## Failure cases and interpretation

Selection is fixed: lowest nDCG@10, then lowest query ID, first five. All five have
nDCG@10=0. The ignored local `failure_cases.json` contains full query/gold/top-three text
for inspection; only IDs and short analytic descriptions are published here.

| Query ID | Top-1 ID | Recall@100 | Observation from local evidence inspection |
|---|---:|---:|---|
| 3 | 22379 | 1/3 | Similar leave/reinstatement vocabulary outranks the judged passages; waiting periods and policy conditions differ. |
| 5 | 13817 | 1/3 | Hire-date cohorts and accrual exceptions differ despite similar date and leave terminology. |
| 62 | 28587 | 0 | The leading scenario uses a different hiring year; another top hit concerns a different employment category. |
| 91 | 25595 | 0 | A lexically similar FAQ about another scenario/year outranks the target leave-policy evidence. |
| 118 | 31768 | 0 | Top hits discuss contract employees, while the query concerns a permanent employee and a different period. |

These examples motivate separating lexical overlap, scenario/entity conditions and
temporal constraints. They are exploratory observations, not a new gold error taxonomy.
Unjudged alternatives may contain useful evidence; unchanged ID-level qrels determine
the scores, and no manual relabeling was performed. The five lowest-ID ties are not a
representative estimate of failure-type frequencies.

## Runtime and reproducibility

Actual environment: AMD Ryzen 5 7500F 6-Core Processor, 12 logical CPUs visible,
Linux/WSL2 x86-64, Python 3.13.11, NumPy 2.5.5, Pydantic 2.14.4. GPU unused.
Queries run sequentially at concurrency 1, after three warmup calls on train query 0.
OMP/MKL/OpenBLAS thread environment variables are each set to 1.

| Measurement | Initial run | Replay |
|---|---:|---:|
| Warm query mean | 70.205 ms | 69.003 ms |
| Warm query p50 | 70.288 ms | 69.707 ms |
| Warm query p95 | 101.995 ms | 97.380 ms |

Initial index construction (including tokenization): 239.033 ms. Query timings include
query tokenization, library scoring, full score ordering and Hit construction. They exclude
loading, grouping/split verification, metrics, bootstrap, output writing and optional
generation. These are two local sequential runs, not API latency, a concurrency load test,
throughput, a latency guarantee or a GPU measurement. Peak memory was not measured in M1.

Both runs produced byte-identical `run.trec`, `qrels.tsv`, and copied configuration;
quality means and bootstrap intervals also matched exactly. Timing values naturally differ.
[Aggregate results and full run provenance](../experiments/results/bm25_company_policies_validation100_v1.json)
include source-file, lockfile, config, split and output hashes. Runs were made from a
dirty working tree rooted at Git commit `ff7322d`; exact source hashes identify the new
implementation rather than falsely attributing it to the parent commit alone.

```bash
uv sync --frozen --group dev
uv run --frozen temporalproof-prepare-split
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen temporalproof-bm25 --output artifacts/bm25/my_validation100_run
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

Use a new output directory for each run; existing runs are never overwritten. Prepare-split
is idempotent for the frozen inputs and refuses to replace a different manifest. Neither
command downloads anything; the M0 audit's explicit `--download` prepares missing inputs.

Local outputs: `manifest.json`, `config.json`, `run.trec`, `qrels.tsv`, `per_query.jsonl`,
`summary.json`, `failure_cases.json`. Query/passage text and qrels stay in ignored artifacts;
the committed split contains IDs only, and committed results contain aggregate metrics
and provenance only. Exact third-party data redistribution terms remain unresolved.

## Next increment

M2: add a pinned compact encoder suitable for Korean and a hybrid candidate pool on
the same frozen validation selection; compare per-query deltas, regressions and recall
before any reranker/router work. Optional [local MLflow tracking](mlflow.md) is now
implemented; M1's original result was imported with unchanged historical provenance.
Retain file-based manifests and keep the final test partition untouched.
