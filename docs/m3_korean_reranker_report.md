# M3 follow-up — Korean reranker on the frozen hybrid pool

This is a preselected follow-up to the negative [MiniLM experiment](m3_reranker_report.md).
It uses the **same 100 validation queries and 100 hybrid candidates per query**. No
training, final-test outcomes, new judgments or hyperparameter search is part of this run.

## Model choice and comparability

Model: **dragonkue/bge-reranker-v2-m3-ko**, revision
`2aca5884ecac490192af9ebd86836d9073d826cd`. The
[pinned author's card](https://huggingface.co/dragonkue/bge-reranker-v2-m3-ko/blob/2aca5884ecac490192af9ebd86836d9073d826cd/README.md)
declares Korean optimization and Apache-2.0. It derives from BAAI/bge-reranker-v2-m3;
complete Korean fine-tuning data and recipe are not specified in the inspected card.
Benchmark contamination is **not independently excluded**. Author-reported benchmark
numbers are not local evidence and are not used as this project's scores.

The model was selected for explicit Korean optimization before observing local scores.
No sweep over multiple follow-up models/weights was performed. It is larger than MiniLM:
six verified local assets total approximately 2.29 GB, with a 2.271 GB FP32 weight file.
[Asset provenance](../configs/models/bge_reranker_v2_m3_ko.json) records the pinned card
hash, asset sizes/SHA-256 and training-disclosure caveat. Weights stay outside Git.

The [experiment config](../configs/experiments/m3_korean_company_policies.json) preserves
the same M2 manifest pin, ordered validation selection, candidate IDs/order/scores, qrels,
seed 1729 and metrics. Source candidate manifest SHA-256:
`fef933fc67bb78de87ff0c5bcfcb4ba7260dcbe83ba0b53e2185b5e83b7d51dc`.
The frozen pool SHA-256 is
`3b8ed06e05d0357f4d889fcb2b4fcb4d9ca2aba4bdf9bcefa188e5fe8b7913ea`.

Both rerankers use **FP32, eager attention, TF32 off, batch 16, max 512 tokens,
dynamic padding, longest-first truncation, concurrency 1**. Tokenizer assets are
model-specific and recorded; they cannot be identical across these architectures. The
Korean model has 24 layers/1024 hidden dimensions and position capacity 8194, versus
MiniLM's 12/384/514, but neither experiment increases the evaluation token limit.
Inputs remain raw query/passage text only. Every pair receives one raw relevance logit;
rank higher logits first with ID-ascending ties. The author's sigmoid is monotonic;
retaining raw logits avoids saturation-induced ties and does not treat scores as
calibrated probabilities. No E5 prefixes, extra titles, timestamps or qrels reach scoring.

The existing evaluator, checked download and offline CUDA loader are shared through a
small explicit model registry. Mismatched model/revision or asset files fail validation;
no arbitrary remote model/code execution or silent CPU fallback is added.

## Measured quality and failure analysis

Measured **2026-11-26**, using full-qrel nDCG@10 and the unchanged relevance scale:

| System | nDCG@10 | Recall@10 | Recall@20 | Recall@100 |
|---|---:|---:|---:|---:|
| Frozen M2 hybrid | 0.207685 | 0.260833 | 0.310000 | 0.415000 |
| M3 MiniLM | 0.109982 | 0.135833 | 0.169167 | 0.415000 |
| M3 Korean BGE follow-up | 0.163721 | 0.200000 | 0.245000 | 0.415000 |

The Korean model improves over MiniLM, yet **still reduces mean quality relative to
hybrid**. It is not adopted as the always-deep path. The reference-prefix diagnostic
is recorded separately (0.241439); it does not replace the main metric.

| Paired nDCG@10 difference | Mean | Conditional 95% interval | Gains / regressions / unchanged |
|---|---:|---:|---:|
| Korean − hybrid | −0.043963 | [−0.086258, −0.004264] | 11 / 28 / 61 |
| Korean − MiniLM | +0.053740 | [0.024141, 0.084094] | 21 / 5 / 74 |

Paired query bootstrap: 2,000 samples, seed 1729, aligned unique query IDs. Intervals
are conditional on the selected 100 development queries (98 connected groups); they
do not model group dependence or prove behavior on unseen queries/domains. These are
model-selection measurements on validation, not a final independent test.

Candidate IDs and qrels match all three methods exactly. Recall@100 matches per query;
43/100 queries have no judged positive in their candidate pool, so pure reranking cannot
recover that evidence. Neither qrels nor data were changed to improve the result.

Examples below follow the saved largest nDCG gains/regressions versus hybrid; raw
queries/passages remain in ignored local artifacts. Observations are not new labels:

| Query ID | Hybrid | Korean reranker | Inspection |
|---|---:|---:|---|
| 2351 | 1.000000 | 0 | The judged policy drops to rank 23; an unjudged, closely matching application scenario becomes first. This is an annotation limitation, not proof of a false new top hit. |
| 2763 | 0.817530 | 0 | Judged leave-consumption clauses drop to ranks 15/41; a different application-date scenario becomes top. |
| 2487 | 1.000000 | 0.237198 | Workday-calculation clauses drop to ranks 5/23; a minimum-tenure eligibility clause outranks them. |
| 2313 | 0.850345 | 0.177239 | Working-hours rules drop to ranks 10/51; a leave-eligibility scenario becomes top. |
| 2059 | 0 | 0.613147 | A judged annual-leave entitlement passage reaches rank 1; another positive remains outside the pool. |
| 5 | 0.181542 | 0.604931 | Judged leave-accrual clauses reach ranks 1/10/11. |
| 1214 | 0.193426 | 0.613147 | A judged eligibility clause reaches rank 1, while another remains at rank 15. |

The pattern includes both scenario/date confusions and useful but unjudged alternatives.
No assertion of exhaustive relevance, temporal validity or answer correctness follows.
A stronger Korean model alone did not resolve the observed full-corpus retrieval problem.

## Resources, replay and evidence

Hardware/software: the same RTX 5060 Ti 16 GB, Ryzen 5 7500F, Linux/WSL2, Python 3.13.11,
PyTorch 2.10.0, Transformers 5.5.0, CUDA 12.8. OMP/MKL/OpenBLAS environment variables are
1. Three warmups use the first selected validation query/pool without labels. The dense
encoder is unloaded; only one reranker resides on GPU. The loader reports no missing,
mismatched or unexpected weights. A separate two-passage synthetic Korean smoke checks
operation and score direction, not benchmark quality. No pair reached the 512-token
limit (10,300 pairs including warmups); length-limit clipping does not explain regressions.

| Incremental rerank measurement | Initial | Replay |
|---|---:|---:|
| Mean | 1,148.100 ms | 1,148.535 ms |
| p50 | 1,119.577 ms | 1,114.958 ms |
| p95 | 1,500.089 ms | 1,499.717 ms |
| CPU pair-tokenization mean | 20.653 ms | 20.621 ms |
| Synchronized forward/transfers mean | 1,126.857 ms | 1,127.344 ms |
| Peak PyTorch allocated | 2,676.298 MiB | 2,676.298 MiB |
| Peak PyTorch reserved | 3,832 MiB | 3,832 MiB |

Initial incremental stage rate: **0.871 queries/s / 87.100 pairs/s**, concurrency 1,
computed from measured stage times. It is not service throughput. Timings include candidate
and text lookup, pair tokenization, CPU/GPU transfers, synchronized forward, scalar scores
and sorting. They exclude download/model load, warmup, metrics, file writes and tracking.
Forward/transfer time includes CPU dispatch; it is not isolated GPU-kernel time. Peaks
are allocator telemetry including warmups, not total-process VRAM or host peak RSS.

MiniLM's earlier mean/p95 was 149.703/204.211 ms; source hybrid retrieval was
88.590/120.321 ms. These earlier separate runs provide context. No combined end-to-end
latency, sum of p95 values, concurrent model residency, CPU-only inference, mixed-precision
optimization or API/load test was measured. The current local result is limited to the
observed inputs and declared FP32/batch/length settings.

Both Korean runs have byte-identical rankings **including all 10,000 raw scores**,
qrels/config, matching metrics, candidate pools and paired deltas. Their source snapshots
match the delivered implementation. Original BM25, M2 and MiniLM file evidence is retained.
[Aggregate evidence](../experiments/results/m3_korean_company_policies_validation100_v1.json)
contains manifests, paired comparisons versus both references and the replay profile.

Initial local MLflow run **`144b746d2199493fba60acfa5c130f05`** is FINISHED with six evidence
artifacts. Model settings, assets/candidate hashes and all call timings are in its manifest;
`retrieval_ms.*` means **incremental rerank only**. Replay is file-only and is not another
independent sample. Full raw data, pair outputs and failure text remain ignored locally.

## Decision and next experiment

The Korean follow-up is complete. **Hybrid remains the fast/default baseline; neither
measured cross-encoder is adopted as the always-deep path.** The larger Korean model
improves some queries and regresses on others, at substantially greater measured cost.
This does not establish an effective adaptive policy or justify a further model sweep.

The next M4 increment should be a **small deterministic train-only feasibility pilot**:
freeze train hybrid candidates, obtain fast/deep outcomes using the same protocol, inspect
gain/cost and candidate-coverage distributions, and determine whether pre-deep features
could predict useful reranking. Keep all negative outcomes. Choose the pilot size/config
before measuring it. Only then consider training/comparing routing policies. Validation's
11 gains are development observations, never training targets or inference features.
Final test, TEMPO, learned routing and application work remain unimplemented.

## Reproduction

From the repository root, after the original verified M2 candidate run is available:

```bash
uv sync --frozen --extra tracking --extra dense --group dev
uv run --frozen --extra tracking --extra dense temporalproof-fetch-reranker \
  --model-id dragonkue/bge-reranker-v2-m3-ko --download
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  uv run --frozen --extra tracking --extra dense temporalproof-rerank --mlflow \
  --config configs/experiments/m3_korean_company_policies.json \
  --output artifacts/m3/my_korean_validation100_run
```

Use another new output directory without `--mlflow` for a file-only replay. Downloads
are explicit; inference is offline. On another machine/run, first verify M2 pool/selection
parity and explicitly pin its source manifest in a separately named config, as described
in the [original report](m3_reranker_report.md#reproduction-and-next-action). Existing
source/results must not be overwritten.

Normal validation uses synthetic CPU data, temporary files and fake inference/downloads:

```bash
uv run --frozen --extra tracking --extra dense pytest -q
uv run --frozen --extra tracking --extra dense ruff check .
uv run --frozen --extra tracking --extra dense ruff format --check .
```

**124 CPU tests pass.** The seven added cases cover registered selection/revision pairs,
unsupported models, unchanged protocol/manifest, manifest-to-code asset consistency,
verification before GPU/network use and model-specific checked downloads, including
corruption cleanup. The existing third-party MLflow/SQLAlchemy deprecation warning remains.

Follow-up: the [first M4 train-only pilot](m4_train_pilot_report.md) has now measured
those utility targets and predeclared controls. It retains both negative validation
results and does not fit a router or evaluate the final test.
