# M2 — dense, hybrid and insertion-time retrieval

Measured **2026-11-26** on the same **100 validation queries / 30,000 passages** as M1.
No test evaluation, local model training, new judgments or hyperparameter search occurred.
These are local development-subset results, not official benchmark or generalization claims.

## Fixed model, corpus and protocols

- [M2 configuration](../configs/experiments/m2_company_policies.json), seed 1729, unchanged
  M1 split SHA-256 `4aab958538041f019363a8ffcc5c5d15d030ebe408d869c17063bf9c3c566fa7`.
  Exact ordered query IDs match M1; 100 queries span 98 connected groups.
- Encoder: **intfloat/multilingual-e5-small**, revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`; model card declares MIT and multilingual
  support including Korean. The author's prescribed query/passage prefixes, masked mean
  pooling and L2 normalization are applied. [Pinned model card](https://huggingface.co/intfloat/multilingual-e5-small/blob/614241f622f53c4eeff9890bdc4f31cfecc418b3/README.md).
- [Asset manifest](../configs/models/multilingual_e5_small.json): six checksum-verified
  files, about 493 MB. Weights/tokenizer are ignored locally, never copied into Git.
- PyTorch **2.10.0**, Transformers **5.5.0**, FAISS CPU **1.16.0**; CUDA **12.8** on RTX
  5060 Ti 16 GB. FP32, eager attention, TF32 off, seed 1729, batch 16 for passages and
  batch 1 for individual query calls, maximum 512 tokens including prefix/special tokens.
- Tokenization uses the pinned fast tokenizer, dynamic padding and truncation. Raw passage
  text is preserved without new chunking/title concatenation. One corpus input reached
  the token limit; this is a potential-truncation count, not proof it was truncated.
- Transformers reports one legacy non-parameter `embeddings.position_ids` buffer as
  unexpected. The loader verifies it is exactly `[0..511]`, the generated default.
  No learned parameters are missing/mismatched. Original weights remain unchanged.
- FAISS `IndexFlatIP` searches L2-normalized vectors for exact cosine similarity.
  All scores are retrieved before deterministic score-descending/ID-ascending ordering,
  so ties at the cutoff cannot depend on FAISS tie order. [FAISS metric documentation](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances).

Every method returns 100 IDs over the same corpus. BM25 uses unchanged M1 parameters and
raw `str.split()`; its quality exactly matches M1. Dense candidates differ from sparse
candidates: this is a retrieval comparison, not a shared-pool reranking comparison.

## Four predeclared variants

1. **BM25**: original lexical baseline, recomputed in the M2 environment.
2. **Dense**: GPU E5 query embedding plus exact CPU FAISS search over all passages.
3. **Hybrid RRF**: equal-weight fusion of BM25 top 100 and dense top 100, score
   `sum(1 / (60 + rank))`, followed by top 100. The union can contain up to 200 IDs.
4. **Hybrid + insertion heuristic**: rerank the exact same hybrid 100 candidates using
   `minmax(RRF) + 0.1 / (1 + abs(document_insert - query_insert) / 365 days)`.
   Unknown/malformed dates contribute no bonus. Past/future gaps are symmetric; nothing
   is removed and no event time or validity interval is inferred.

The last variant tests **insertion metadata only**. It is not effective-time reasoning,
a known-at filter, a recency guarantee or the official TimelyRAG method. The dataset's
later-inserted positives and missing event fields still limit temporal claims.

## Results

| Method | nDCG@10 | Recall@10 | Recall@20 | Recall@100 | Queries with no gold in top 100 |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.175959 | 0.230000 | 0.281667 | 0.393333 | 45 |
| Dense | 0.150042 | 0.195000 | 0.220000 | 0.361667 | 48 |
| Hybrid RRF | 0.207685 | 0.260833 | 0.310000 | 0.415000 | 43 |
| Hybrid + insertion | 0.203661 | 0.257500 | 0.313333 | 0.415000 | 43 |

Primary nDCG uses full positive qrels; separately named reference-prefix diagnostics
remain in each run. [Aggregate results, source manifests and paired deltas](../experiments/results/m2_company_policies_validation100_v1.json).

| nDCG@10 comparison | Mean paired delta | Conditional paired query-bootstrap 95% interval | Improved / regressed / unchanged |
|---|---:|---:|---:|
| Dense − BM25 | −0.025916 | −0.075616–0.025354 | 20 / 28 / 52 |
| Hybrid − BM25 | +0.031726 | −0.008300–0.071907 | 24 / 17 / 59 |
| Hybrid + insertion − BM25 | +0.027702 | −0.011871–0.067251 | 23 / 17 / 60 |
| Hybrid + insertion − hybrid | −0.004023 | −0.011648–0.002148 | 7 / 9 / 84 |

All listed intervals contain zero. They use 2,000 paired query resamples with seed 1729,
conditional on the fixed development sample; group dependence is not modeled. The
aggregate mean improvement does **not** establish significance or superiority on unseen
domains. Dense alone is weaker in this sample. The fixed insertion bonus did not improve
hybrid's mean nDCG, despite helping some queries. Parameters were not retuned to hide this.

Hybrid recall improves modestly, yet 43 queries still have no judged positive in the
frozen candidate pool. A reranker over these candidates cannot recover missing evidence.
The temporal variant's Recall@100 exactly matches hybrid because its candidate set is
identical; candidate parity was checked for every query.

## Gains, regressions and annotation limits

Examples come from the saved paired-delta ordering: highest gains and largest regressions,
with ties handled by the deterministic query-ID order in the comparison output. Text was
inspected locally; raw queries/passages/qrels are not redistributed in this report.

| Query ID | BM25 nDCG@10 | Dense | Hybrid | Observation |
|---|---:|---:|---:|---|
| 855 | 0 | 1.000000 | 1.000000 | Dense recovers the judged vacation-payment policy; BM25 leads with promotion criteria. |
| 2069 | 0 | 0.386853 | 0.613147 | Fusion promotes a judged passage about leave eligibility and accumulated leave. |
| 1443 | 0.932521 | 0 | 0 | Dense/fusion lose strong lexical evidence and promote similar annual-leave rules with different conditions. |
| 1214 | 0.524981 | 0 | 0.193426 | The hybrid top hit is textually similar to the judged restriction but has an unjudged ID; usefulness cannot be inferred from ID scores alone. |
| 2059 | 0.386853 | 0 | 0 | Similar leave/year language favors another hiring cohort and policy scenario. |

These are exploratory observations, not manually created temporal-error labels. ID-level
qrels are unchanged, and unjudged alternatives are scored as zero without being asserted
factually irrelevant. Semantic ambiguity and incomplete judgments remain limitations.

## Resources, timing and replay

Host: AMD Ryzen 5 7500F, 12 logical CPUs visible, Linux/WSL2, Python 3.13.11, one RTX
5060 Ti. OMP/MKL/OpenBLAS thread environment variables are each 1; query concurrency is 1.
Methods run sequentially with three train-query warmup calls per method. CUDA synchronization
is included before returning query embeddings; no asynchronous GPU work is omitted.

| Method | Initial mean / p95 | Replay mean / p95 |
|---|---:|---:|
| BM25 | 77.755 / 109.948 ms | 79.071 / 116.088 ms |
| Dense | 11.913 / 13.033 ms | 11.817 / 12.944 ms |
| Hybrid RRF | 88.590 / 120.321 ms | 93.808 / 127.591 ms |
| Hybrid + insertion | 90.572 / 127.383 ms | 93.565 / 124.533 ms |

These compare a simple CPU BM25 library with GPU query encoding plus CPU FAISS on this
machine; they do not prove universally faster retrieval. They are warm sequential stage
timings, not API/load-test latency or throughput. Metrics, MLflow logging, downloads,
model load and offline passage encoding are excluded.

Offline embedding/cache build (including array writes): **34.076 s**; CPU FAISS
construction: **93.680 ms**; BM25 construction: **214.784 ms**. Corpus embeddings are
~46.1 MB float32 plus ID arrays.
Peak PyTorch allocator memory was **923.195 MiB allocated / 1,518 MiB reserved** during
the initial run including corpus encoding. This is allocator telemetry, not total process
VRAM or peak host RSS. The replay reused passage embeddings and had lower allocation peaks.

The immutable cache keys include ID/text hashes, model/config/assets, encoder/cache source
hashes, device, FAISS version and environment lock hash. Loads verify file hashes and use
NumPy `allow_pickle=False`.
Every replay TREC file (including scores), qrel file, config and quality summary matched
exactly. Original M1 selected IDs/split matched; original M1 file evidence is untouched.

All four final verification runs are FINISHED in local MLflow; the replay was file-only.
An earlier run/replay pair preceded adding the environment lock to cache identity; its
rankings and quality were identical and its timings/provenance are also retained in the
aggregate. These repeated validation checks are not independent query samples. The table
above uses the final cache-guard implementation and its replay. MLflow tags distinguish
`environment_lock_verified` from `before_environment_lock_guard`; use the former for the
current source snapshot. No final-test, domain transfer, neural reranker/router, index
update workload or production service was evaluated.

## Reproduction and next step

```bash
uv sync --frozen --extra tracking --extra dense --group dev
uv run --frozen --extra tracking --extra dense temporalproof-fetch-encoder --download
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  uv run --frozen --extra tracking --extra dense temporalproof-compare --mlflow \
  --output artifacts/m2/my_validation100_run
uv run --frozen --extra tracking --extra dense pytest -q
```

The initial GPU/dependency downloads were explicitly approved by the user. Only the
fetch command's `--download` accesses model assets; evaluation reads local verified assets
and refuses an unreported CPU fallback. Unit tests use tiny arrays/fake encoders and no GPU
inference or model downloads. Keep `--extra tracking --extra dense` on commands using these
optional dependencies. Normal core tests remain CPU-only.

M3 should freeze a candidate pool (initially the saved hybrid top 100), add one fixed
compact multilingual cross-encoder, and measure incremental gain/cost before constructing
any router targets. Continue using train/validation for choices; keep final test untouched.
