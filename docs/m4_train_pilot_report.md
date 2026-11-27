# M4 pilot — train-only fast/deep utility

This is the first M4 increment after both validation rerankers regressed against hybrid.
It measures one **predeclared 100-query training subset**, builds train-only utility targets
and saves inference-available features. No router is fitted and no validation/test outcomes
are evaluated here. This is a feasibility experiment, not a completed M4 result.

## Predeclared configuration and boundaries

[Configuration](../configs/experiments/m4_train_pilot_company_policies.json) was written
before collecting pilot outcomes. The existing grouped split and seeded query-selection
rule (1729, SHA-256 ordering) select 100 train IDs. The CLI saves config, source snapshot,
selected IDs and full-source group identities in `plan.json` before GPU evaluation.
The split is reverified against the pinned original dataset; configs for validation/test
or a non-hybrid fast path are rejected. The full 30,000-passage unlabeled corpus remains
shared; this is query-held-out research, not corpus/domain separation.

Both models are unchanged: pinned multilingual-e5-small for hybrid (BM25 + FAISS cosine,
equal RRF offset 60) and pinned Korean BGE for deep reranking. FP32/eager/TF32-off,
batch 16, max 512, original text/preprocessing, deterministic score/ID ties. Three warmups
per stage; query concurrency 1. Fast uses a train warmup; frozen deep uses its first
selected train query. The encoder is removed before loading the reranker. No model/API
purchase, new download, corpus transformation or label rewriting occurs.

The feature file and fast manifest hashes are checkpointed **before** loading/scoring
the reranker. They must remain unchanged afterward. Separate files hold:

- `features.jsonl`: query IDs as join metadata plus a strict 15-field pre-deep schema;
- `outcomes.jsonl`: train group IDs, fast/deep nDCG, gain and measured incremental cost;
- standard fast/deep TREC, qrels, per-query metrics, manifests and local failure text;
- `summary.json`: descriptive policies/slices and cost-sensitive target counts.

Features use query length, distinct explicit years (1900–2099), top-10 sparse/dense
intersection divided by 10, top-1 agreement, relative BM25/RRF margins, raw cosine margin,
normalized entropy of RRF scores, insertion-date coverage/gaps and normalized text diversity.
Unknown dates remain null where appropriate. Insertion dates/text repetition are proxies,
not true version chains, event time or applicability labels. This retains the existing
retrospective full-corpus protocol; it does not simulate historical known-at availability.
No answer, qrel, gain, deep score, future outcome, query ID or document ID is an inference
feature. IDs in outer rows only join data and preserve grouped splitting.

## Targets and diagnostic controls

For train queries only:

`gain = deep_nDCG@10 - fast_nDCG@10`

`positive(lambda) = gain > lambda * incremental_rerank_ms / 1000 + 1e-12`

Predeclared lambda values are **0, 0.01, 0.05 nDCG/second**. The deep stage is already
incremental reranking on saved candidates: **do not subtract fast latency again**.
Raw gains/costs and all negative cases remain saved. Feature/outcome joins require unique,
matching train IDs and unchanged per-query candidate recall. Group identity uses the full
frozen source group, so later grouped cross-validation can keep related selected queries
together; no training/validation fold is created or fitted in this pilot.

Predeclared controls:

- always fast and always deep;
- rerank when sparse/dense top-10 overlap fraction is **below 0.2**;
- rerank when that disagreement condition and an explicit year both hold;
- one seeded hash-random selection at query-fraction caps **0, 0.1, 0.25, 0.5, 1**;
- an explicitly offline **oracle upper bound** selecting largest positive gains up to
  those query-count caps. It uses training outcomes and is never a deployable policy,
  inference feature or reported learned-router result.

These thresholds/budgets were not selected to maximize pilot labels. Feature slices are
predeclared descriptive comparisons, not evidence of held-out predictive power. Random
controls use one fixed seed, not an estimate averaged over many random policies.

Policy cost is **modeled mean additive stage cost**: measured fast time (including
feature extraction) plus measured incremental rerank time only for routed queries.
Stages are measured separately with sequential models. No combined live pipeline,
combined p95, simultaneous residency or service throughput is inferred from these sums.

## Measured result — 2026-11-27

The selection contains **100 queries from 100 distinct source groups**. It is a training
subset; do not compare its mean directly against the earlier validation100 means.

| Path | nDCG@10 | Recall@10 | Recall@20 | Recall@100 |
|---|---:|---:|---:|---:|
| Hybrid | 0.181688 | 0.247833 | 0.302333 | 0.386333 |
| Fixed Korean reranker | 0.158216 | 0.207000 | 0.233667 | 0.386333 |

Deep reranking improves **15**, regresses **18** and leaves **67** queries unchanged.
Its paired mean nDCG delta is **−0.023473**, conditional 95% query-bootstrap interval
**[−0.059726, 0.013150]** (2,000 samples, seed 1729). The interval contains zero; no
superiority/inferiority claim on unseen data follows. Main nDCG uses full qrels, while
reference-prefix diagnostics remain separately recorded in the aggregate.

**45/100 queries have no judged positive in the frozen top 100.** Reranking cannot
recover these missing candidates. Recall@100 is identical per query between fast/deep.
No positives were removed, added or relabeled. Binary training target positives:

| Lambda, nDCG per second | Positive targets |
|---|---:|
| 0 | 15 |
| 0.01 | 15 |
| 0.05 | 13 |

These targets and their post-hoc costs are training artifacts, never inference features.
Only 15 positive examples support the zero-penalty target here; classification accuracy
or an in-sample policy gain would not demonstrate effective adaptive retrieval.

## Predeclared policy diagnostics

All costs below are **modeled mean sums of measured stages**, not measured complete
pipeline latency. Fast cost includes feature extraction for every policy; a feature-free
fast implementation was not separately timed in this pilot.

| Control | Routed queries | nDCG@10 | Modeled mean stage cost |
|---|---:|---:|---:|
| Always hybrid | 0 | 0.181688 | 94.503 ms |
| Always rerank | 100 | 0.158216 | 1,246.305 ms |
| Low agreement, overlap <0.2 | 64 | 0.187237 | 843.716 ms |
| Explicit year and low agreement | 64 | 0.187237 | 843.716 ms |
| Seeded random, 10% query cap | 10 | 0.180781 | 212.127 ms |
| Seeded random, 25% query cap | 25 | 0.171431 | 389.179 ms |
| Seeded random, 50% query cap | 50 | 0.164078 | 677.457 ms |
| **Offline oracle bound**, at most 10% | 10 | 0.213682 | 202.135 ms |
| **Offline oracle bound**, at most 25% | 15 | 0.217611 | 257.871 ms |

The rule's paired mean nDCG delta is **+0.005549**, interval
**[−0.013982, 0.028870]**, 7 gains / 7 regressions / 86 unchanged. It routes most queries
and adds substantial cost for a small observed mean gain. It is not established as a
useful held-out router. The year condition changes no decisions here: 99 queries contain
an explicit year and the one remaining query is not in the low-agreement set.

Oracle rows select queries using gold-dependent outcomes and give a **training-sample
upper bound only**. At caps 25%, 50% and 100%, they route only the 15 positive-gain
queries, reaching the same 0.217611. This is not achievable-policy evidence, a learned
router or a plan to deploy gold-dependent decisions. The
[aggregate](../experiments/results/m4_company_policies_train100_v1.json) retains all
predeclared budgets, costs, routed IDs, intervals and descriptive slices.

The feature slices show low agreement's mean gain +0.008670 (64 queries, 7 positives)
and other agreement's −0.080615 (36 queries, 8 positives). This conditional observation
does not establish predictive power. Query-insertion time is known for all selected
queries; that particular missingness flag has no variation in this sample.

## Resource scope, interruption and replay

Same RTX 5060 Ti 16 GB, Ryzen 5 7500F / WSL2, Python 3.13.11, PyTorch 2.10.0,
Transformers 5.5.0 / CUDA 12.8. The existing corpus embedding cache was verified/reused;
no new corpus encoding is included in query timing.

| Measured stage | Initial mean / p95 | Replay mean / p95 |
|---|---:|---:|
| Hybrid including features | 94.503 / 123.161 ms | 90.535 / 123.463 ms |
| Incremental reranker | 1,151.802 / 1,438.660 ms | 1,147.255 / 1,413.324 ms |
| Feature extraction only, mean | 1.795 ms | 1.751 ms |

Feature timing is already included in fast time; do not add it a second time. Fast
allocator peaks are **459.697 MiB allocated / 484 MiB reserved**; deep peaks are
**2,728.898 MiB / 4,626 MiB**, identical across these replays. These are stage-specific
PyTorch allocator counters, not total-process VRAM, host RSS or simultaneous capacity.
None of the 10,300 reranker pairs including warmup reached the 512-token limit.

An initial fast-only run stopped on an overly strict requirement for zero allocator
bytes between models. It remains at `artifacts/m4/company_policies_train100_v1` and did
not run the reranker or generate targets. The corrected guard directly verifies the
encoder model is collected by weak reference, while recording residual allocator state
(**9,568,256 bytes = 9.125 MiB**). That counter is not attributed to a specific allocation;
PyTorch 2.10 documents persistent backend workspaces, so zero is not a valid requirement
for model removal. [PyTorch CUDA semantics](https://docs.pytorch.org/docs/2.10/notes/cuda.html#cublas-workspaces).
The stopped attempt's fast rankings/scores, qrels and feature file match the corrected run
exactly. Its timings/source snapshot are retained, not pooled as extra samples.

Completed runs:

- `artifacts/m4/company_policies_train100_verified_v1` — initial, with local MLflow;
- `artifacts/m4/company_policies_train100_verified_replay_v1` — file-only replay.

Replay has byte-identical plans/configs, features, fast/deep rankings **including all
scores** and qrels. Metrics, gains, source group identities and paired deltas match.
All cost-sensitive labels at all three lambda values also match; measured times vary as
shown. The generated fast manifest pins differ because their measured times differ;
this is expected and does not alter candidate membership/order/scores. Both completed
runs match the final method source snapshot. These are repeated checks, not 200 queries.

Local MLflow records are FINISHED:

- Fast `66198b1aee61450a804c7b33e6363776`;
- Deep `001ba53181f041c49df1ca9c88ca8b63`.

Each has six standard evidence artifacts. Deep additionally stores the pilot allowlist
and `pilot/analysis` aggregate, PNG/SVG quality–cost figure and figure-provenance manifest.
Copied pilot artifacts were hash-checked against their local sources; no raw failure text
is uploaded. Full corpus, weights, scores and targets remain in ignored local directories.

Validation: **140 CPU tests pass**, Ruff lint/format and diff checks pass. New tests
cover held-out-ID/config rejection, strict inference schemas, unknown dates, aligned
train outcomes, incremental-cost units, rule/random independence from outcome edits,
explicit oracle boundaries, frozen-before-scoring fake end-to-end integration and the
tracking allowlist. One existing third-party MLflow/SQLAlchemy warning remains.

## Next increment

Predeclare a larger train sample and grouped out-of-fold comparison of a regularized
simple router against the fixed rules, always-fast/deep and random budgets. Keep feature
preprocessing/fitting inside each training fold, reserve validation for subsequent method
selection and keep final test untouched. A pilot with 15 positives and weak/high-cost
rules is insufficient to claim the main ASTR contribution. No fitted policy, held-out
routing quality, actual end-to-end p95 or transfer result exists yet.

## Reproduction

From the repository root with the already verified local models/dataset:

```bash
uv sync --frozen --extra tracking --extra dense --group dev
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  uv run --frozen --extra tracking --extra dense temporalproof-pilot --mlflow \
  --output artifacts/m4/my_train100_pilot
```

Use another new directory without `--mlflow` for replay. No model or dataset download
occurs in the pilot. Existing artifacts are never overwritten. Normal tests are CPU-only
and use independent synthetic text, fake search/scoring, temporary files and fake tracking.

Reproduce aggregate checks and the static scientific figure from completed local runs:

```bash
uv run --frozen --extra tracking --extra dense python experiments/analyze_m4_pilot.py \
  --run artifacts/m4/company_policies_train100_verified_v1 \
  --replay artifacts/m4/company_policies_train100_verified_replay_v1 \
  --measured-on 2026-11-27 --output artifacts/m4/my_verified_aggregate.json
uv run --frozen --extra tracking --extra dense python experiments/plot_m4_pilot.py \
  --results artifacts/m4/my_verified_aggregate.json \
  --output artifacts/m4/my_quality_cost.png
```

Matplotlib is supplied by the existing locked tracking extra. The plot can also read the
committed aggregate, without raw benchmark text. PNG/SVG outputs are standalone local
artifacts; the analysis/figure script hashes and Matplotlib version are recorded locally.
