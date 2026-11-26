# M3 — fixed cross-encoder and incremental profiling

Measured **2026-11-26** on the unchanged **100 validation queries / 30,000-passage corpus**.
The fixed MiniLM reranker **reduces mean retrieval quality** on this subset and adds GPU
work. Preserve this negative result; it does not justify adopting this model as the deep
path or proceeding directly to learned routing. No training or test evaluation occurred.

## Frozen inputs and model

[M3 configuration](../configs/experiments/m3_company_policies.json) pins the original M2
hybrid manifest SHA-256 `fef933fc67bb78de87ff0c5bcfcb4ba7260dcbe83ba0b53e2185b5e83b7d51dc`.
The TREC candidate file SHA-256 is
`8df0d79f34035189c5aa385ea57794b18aa10cee8cd2db51a4d13f7f3700d6ab`.
Exactly 100 original IDs/order/scores per query are copied into an immutable local pool;
its SHA-256 is `3b8ed06e05d0357f4d889fcb2b4fcb4d9ca2aba4bdf9bcefa188e5fe8b7913ea`.
Before inference, the command checks all six source-file hashes, the source method/config,
archive/split/selection, unique known IDs, consecutive ranks, score/tie order and unchanged
qrels. The original M1 split and ordered validation selection remain unchanged (98 groups).

Model: **cross-encoder/mmarco-mMiniLMv2-L12-H384-v1**, revision
`1427fd652930e4ba29e8149678df786c240d8825`, approximately 118 million parameters and 493 MB
of six pinned local assets. The [model card](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/blob/1427fd652930e4ba29e8149678df786c240d8825/README.md)
declares Apache-2.0 and multilingual mMARCO training. **Korean is absent from its documented
supervised training languages**; this experiment measures transfer to Korean from a
multilingual XLM-R-derived backbone, without claiming validated Korean relevance quality.
See [asset checksums and language caveat](../configs/models/mmarco_minilm.json).

Inference uses PyTorch 2.10.0 / Transformers 5.5.0, CUDA 12.8, FP32/eager attention,
TF32 off, seed 1729, batch 16, dynamic padding, maximum 512 tokens including special tokens,
`longest_first` pair truncation. Inputs are query/raw-passage pairs only: no metadata,
titles, E5 prefixes, IDs or qrels reach the model. One scalar **raw relevance logit** per
pair, identity activation as specified in the pinned config; higher scores rank first,
with document-ID ascending ties. All 100 candidates are scored; no new candidate is added.

No learned parameters are missing/mismatched. A legacy non-learned
`roberta.embeddings.position_ids` buffer is checked against generated `[0..513]`;
original weights remain unchanged. A separate synthetic Korean smoke correctly scored
a Seoul/South-Korea passage above a Paris/France passage. This checks operation and score
direction only, not domain quality. None of the 10,300 pairs including warmup reached the
512-token limit, so observed degradation is not explained by length-limit clipping.

## Quality and regressions

| Method | nDCG@10 | Recall@10 | Recall@20 | Recall@100 | No gold in top 100 |
|---|---:|---:|---:|---:|---:|
| Frozen M2 hybrid RRF | 0.207685 | 0.260833 | 0.310000 | 0.415000 | 43 |
| Fixed MiniLM cross-encoder | 0.109982 | 0.135833 | 0.169167 | 0.415000 | 43 |

Main nDCG uses full qrels. The separate reference-prefix diagnostic is 0.165393 for the
reranker; it is not substituted for the main metric. Recall@100 matches **per query**,
not only in aggregate, because candidate membership is unchanged. The 43 queries without
any judged positive in the pool cannot be recovered by this reranker.

| Paired difference, reranker − hybrid | Mean | Conditional query-bootstrap 95% interval | Improved / regressed / unchanged |
|---|---:|---:|---:|
| nDCG@10 | −0.097703 | [−0.144189, −0.056641] | 7 / 33 / 60 |
| Recall@10 | −0.125000 | [−0.180000, −0.076667] | 1 / 26 / 73 |
| Recall@20 | −0.140833 | [−0.193354, −0.090000] | 1 / 29 / 70 |
| Recall@100 | 0 | [0, 0] | 0 / 0 / 100 |

Intervals use 2,000 paired query resamples with seed 1729, conditional on this fixed
sample. They do not model group dependence or establish behavior on other domains.
No model/parameter sweep was performed to conceal the regression.
[Aggregate evidence and manifests](../experiments/results/m3_company_policies_validation100_v1.json)
retain the source fast run, all metrics/deltas and replay profile. Full pair scores are
cached locally in `run/run.trec`; query/positive text remains in ignored artifacts.

Examples were selected from the saved largest positive/negative nDCG deltas, with the
same deterministic ordering as M2. Raw source text is not redistributed:

| Query ID | Hybrid nDCG@10 | Reranker | Local inspection |
|---|---:|---:|---|
| 2351 | 1.000000 | 0 | Judged policy drops to rank 11; the new top hit is an unjudged, closely matching application scenario. This illustrates an annotation limit, not proof that the new top hit is false. |
| 2313 | 0.850345 | 0 | Reduced-working-hours rules drop to ranks 87/89; a leave-application scenario outranks them. |
| 2763 | 0.817530 | 0 | Judged leave-period consumption clauses drop to ranks 88/93; a return-to-work allowance scenario becomes top. |
| 1868 | 0.264068 | 0.613147 | One judged annual-leave cohort clause rises to rank 1; the other remains at rank 38. |
| 758 | 0.167160 | 0.469279 | A judged reward clause rises to rank 1; two other judged IDs remain outside the pool. |
| 200 | 0.386853 | 0.591235 | Judged pay-restriction clauses rise to ranks 2/7, but an unjudged different-cohort clause remains first. |

These are exploratory observations with unchanged ID-level judgments. Unjudged passages
score zero without being asserted irrelevant or false. No new temporal-validity labels
or automatic answer-correctness claims are introduced.

## Incremental latency, memory and replay

Host: Ryzen 5 7500F, 12 visible logical CPUs, Linux/WSL2, Python 3.13.11, RTX 5060 Ti
16 GB. OMP/MKL/OpenBLAS environment variables each 1; query concurrency 1; pair batch 16.
Three warmup calls use the first selected validation query and its frozen candidates,
without consulting labels. They are excluded from latency and included in allocator peaks.
The encoder is **not resident**: the reranker reads saved candidates.

| Measurement | Initial | Replay |
|---|---:|---:|
| Incremental rerank mean | 149.703 ms | 147.124 ms |
| Incremental rerank p50 | 142.781 ms | 139.754 ms |
| Incremental rerank p95 | 204.211 ms | 200.628 ms |
| CPU pair-tokenization mean | 20.179 ms | 19.811 ms |
| Synchronized forward/transfers mean | 129.142 ms | 126.947 ms |
| PyTorch peak allocated | 784.868 MiB | 784.868 MiB |
| PyTorch peak reserved | 1,540 MiB | 1,540 MiB |

Overall incremental timing includes candidate/text lookup, tokenization, CPU/GPU
transfers, synchronized forward calls, conversion to scalar scores and sorting. It
excludes model load/download, warmup, metrics, file writes and MLflow. Forward/transfer
component is wall time including CPU dispatch, not kernel-only GPU time. Allocator
telemetry excludes external CUDA/driver allocations; total process VRAM/host peak RSS
and CPU-only model inference remain unmeasured.

The initial measurement corresponds to **6.680 rerank queries/s / 667.991 pairs/s**,
computed from summed measured stage time at concurrency 1. These are stage rates, not
service throughput. The source M2 fast mean/p95 was **88.590 / 120.321 ms**, measured in
a separate earlier run. **No live end-to-end pipeline, simultaneous model residence,
combined p95 or service load test was measured.** Adding separate p95 values would be
invalid. Shared candidate scores are used for comparable quality, not to report fake
cached-fast latency as a live hybrid+reranker measurement.

Replay produced byte-identical TREC rankings **including all 10,000 pair scores**,
qrels and config; metrics, candidate pool hashes and paired deltas are identical. Source
and dependency hashes retain the original MiniLM implementation; subsequent follow-ups
record their own snapshots. The initial run is FINISHED in local
MLflow, ID `1419b5ef303b4611a38e40a9f61cdfa9`, with six evidence artifacts including the
manifest/profile. `retrieval_ms.*` in this entry means **incremental rerank only**;
the full scope is in its tag/summary. Replay is file-only, not another independent sample.

## Reproduction and next action

```bash
uv sync --frozen --extra tracking --extra dense --group dev
uv run --frozen --extra tracking --extra dense temporalproof-fetch-reranker --download
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  uv run --frozen --extra tracking --extra dense temporalproof-rerank --mlflow \
  --output artifacts/m3/my_validation100_run
uv run --frozen --extra tracking --extra dense pytest -q
```

The default config requires the **original local M2 verified hybrid directory**. It is
not committed because rankings/qrels and raw content stay local. On another machine,
first reproduce M2, inspect its selection/pool parity and explicitly pin that new source
manifest SHA in a separately named M3 config passed with `--config` and `--candidates`.
Manifest hashes include provenance and can differ even with identical rankings. Never
bypass checks or overwrite original evidence. Evaluation performs no network downloads.

Validation: **117 CPU tests pass**, Ruff lint/format and diff checks pass. Tests cover
source tampering, bad/duplicate/unknown IDs, ranks/scores/cutoff, qrel/config/split drift,
stable score ties, explicit verified downloads and a fake-scored end-to-end evaluation
with identical query texts but different IDs. They require no GPU/model downloads.
The existing third-party MLflow/SQLAlchemy deprecation warning remains.

Follow-up completed: the [Korean BGE experiment](m3_korean_reranker_report.md) uses the
same pool/protocol and reaches nDCG@10 0.163721, still below hybrid, at 1,148.100 ms mean
incremental cost. Original MiniLM evidence is unchanged. Next is a small deterministic
train-only feasibility pilot before learning a router. Any router targets must come from
**training queries only**; neither validation run is a training set. Test, TEMPO, routing
and application work remain pending.
