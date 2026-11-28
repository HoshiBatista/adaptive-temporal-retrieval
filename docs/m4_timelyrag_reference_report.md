# M4.3 — Official TimelyRAG code on matched local validation

Measured **2026-11-28** on the full 30,000-passage Company Policies corpus and all
**495 development validation queries / 462 source groups**. Final test447 remains closed.
This stage reproduces the **automatic-alpha code path** with BM25, and evaluates its
transfer to our existing E5 dense/hybrid backbones. It does **not** reproduce the paper's
whole-benchmark numbers or its BGE-M3/NV-Embed/BGE-Gemma backbones.

## Frozen sources and scope

- Official source: [TimelyRAG](https://github.com/kaist-dmlab/TimelyRAG), commit
  `5213a0c1743f23fd52a258005c3887aaf7e8cd0f`.
- Ten source files are pinned by size/SHA256 in
  [`timelyrag_reference.json`](../configs/datasets/timelyrag_reference.json).
  The five newly fetched files total only small source/config text; model weights were
  not downloaded. Original files stay unmodified in ignored `data/raw/`.
- Existing E5-small revision `614241f622f53c4eeff9890bdc4f31cfecc418b3` runs on **CUDA**,
  FP32/eager/TF32-off, configured batch16/max512, with one query per measured call.
  Verified corpus cache is reused; BM25/FAISS/time logic run on CPU.
- Experiment config
  [`m4_timelyrag_reference_company_policies.json`](../configs/experiments/m4_timelyrag_reference_company_policies.json),
  SHA `062e6a9d5379672152e76e5d8ec0a87107cbe84068b1919bfe8c60681d69e966`, was fixed
  before outcomes. Seed1729, top100 pool, top10 reference cutoff, RRF offset60,
  unchanged whitespace BM25 k1=1.5/b=.75/epsilon=.25, three warmups/concurrency1.
- **Primary:** BM25 + native automatic temporal module versus matched native BM25.
  E5 dense/hybrid transfers are secondary. No parameter selection, router fitting,
  new training targets or benchmark oracle evaluation.

Published Company nDCG@10 values from [Table 4](https://aclanthology.org/2026.findings-emnlp.398.pdf#page=7)
(BM25 .226/.227, BGE-M3 .197/.200, NV-Embed-V2 .195/.228, BGE-Gemma2 .197/.230)
are contextual paper values, not local measurements. The query subset, available model
backbones, runtime versions and metric definitions must be aligned before comparing.
The exact mapping from published table entries to repository auto/best-alpha output is
not established by this experiment.

## How source parity is checked

Reviewed `time_utils`, `time_scores`, `alpha` and metric helper modules execute unchanged,
with their absolute import names restored after loading. Only numeric config assignments
are compiled, avoiding config.py's directory-creation side effect. The two original free
functions in retrieval.py are compiled as unchanged AST nodes for parity checks; the
unused model wrapper, FlagEmbedding import and pipeline entrypoint are excluded. There
is no hidden BGE-M3 download. No original evaluation file is edited or vendored into Git.

The qrel-free inference adapter accepts only scores, corpus IDs/timestamps, query text
and insertion time. **36 synthetic cases** compare its top10/alpha/metric outputs with
the complete original auto function, covering ties, float32/64, flat scores and unknown,
future/equal and mixed event dates. The original oracle runs on synthetic labels only.

On real validation, the original function is stopped at its **first metric boundary**
to observe the completed automatic ranking, before best-alpha executes. Ranking/helper
code and published labels are unchanged; this is instrumentation for parity, not a
replacement metric in the experiment. All **1,485 query/backbone top10s** match. Cached
CPU replay verifies complete top100 scores/order/alpha/diagnostics, official per-query
metrics (absolute tolerance1e-12), candidate coverage and paired statistics. All495 BM25
full-corpus score vectors are independently recomputed.

`best-alpha` directly inspects each query's positives in the upstream code. It is an
**offline oracle**, never a normal baseline, routing feature or deployable policy. It is
not evaluated on any actual benchmark query in this stage.

## Metric and tie policies

Primary nDCG@10 normalizes by **all positive qrels**, including unretrieved positives.
The original code's nDCG/MAP normalizations use positives in the retrieved prefix; both
are exported in separately named `reference_prefix_*` columns. Recall@K still uses all
positives. Never merge the definitions or replace one with the larger value.

Native `np.argsort(scores)[::-1]` is preserved for the upstream comparison, including
ties. This differs from previous local document-ID-ascending ties. The generic baseline
runner now refuses a native-tie label without an explicit native callback, preventing
silently mislabeled ordinary BM25 runs. Existing configs/CPU behavior remain supported.

The previous and native-order hybrid have identical candidate/score maps for all495
queries, but ordering within ties changes. nDCG@10 changes on11 queries and Recall@10
on2; old hybrid mean0.198972 becomes native-order0.198335. Therefore all new paired
temporal deltas use the **new native baseline**, with no inference from that tie-only shift.

## Temporal metadata limits and observed code behavior

- Explicit document event times are absent. **47/30,000** insertion strings remain unknown.
  No document effective intervals or version-family gold are invented.
- The upstream heuristic derives a query event time for **491/495** queries. It chooses
  the finest detected expression, then its earliest position. In multi-date questions,
  this can be a hire/cohort date rather than the requested policy period; this extraction
  is a heuristic, not validated applicability ground truth.
- `compute_single_delta` marks `DIT >= QIT` unknown; `time_affinity_from_delta` maps unknown
  to **0.5**. Candidates remain in the pool. The actual code is **not a hard known-at
  filter**. No future/equal or unknown candidates/positives are discarded or relabeled.
- Unknown-distance candidate counts out of49,500: BM25 **12,552**, E5 **11,229**,
  hybrid **12,321**; later/equal insertion candidates **12,518/11,202/12,290**.
- Existing event/insertion fallbacks and automatic weights/gates are preserved, including
  their distinction between explicit and fallback availability terms.

These observations identify limitations to audit. They do not prove a cause of every
regression, establish temporal-hard gold or demonstrate that temporal information is useless.

## Measured results

The measured reference-code comparison is negative on this development selection.
Main full-qrel metric and separately named upstream-prefix metric:

| Backbone | Full-qrel nDCG@10 base | + auto | Reference-prefix nDCG@10 base | + auto |
|---|---:|---:|---:|---:|
| BM25 (primary) | 0.154621 | 0.146812 | 0.217608 | 0.208895 |
| E5 dense transfer | 0.141065 | 0.131546 | 0.205982 | 0.190062 |
| E5 hybrid transfer | 0.198335 | 0.173203 | 0.275678 | 0.240788 |

Paired full-qrel nDCG delta, 2,000 query-weighted **source-group** resamples:

| Backbone | Mean delta | Conditional 95% interval | Gains / regressions / unchanged |
|---|---:|---|---:|
| BM25 | −0.007809 | [−0.017143, 0.002364] | 56 / 74 / 365 |
| E5 dense | −0.009519 | [−0.021288, 0.002741] | 59 / 71 / 365 |
| E5 hybrid | −0.025133 | [−0.037790, −0.012204] | 53 / 95 / 347 |

Primary BM25 and E5 intervals include zero; neither establishes a gain. The secondary
hybrid interval lies below zero. Group intervals condition on fixed rankings and do not
refit anything. No post-hoc alpha/model changes replace the frozen primary.

Recall@100 is unchanged per query: BM25 **0.368923**, E5 **0.362357**, hybrid **0.417744**.
Recall@10 base→auto: **0.206229→0.195455**, **0.179461→0.169360**,
**0.242088→0.225253**. Every auto ranking preserves the exact base top100 pool. Complete
reference Hit/Precision/Recall/prefix-nDCG/prefix-MAP columns are retained in the aggregate.

### Runtime and reproducibility

Measured warm stages, concurrency1/three warmups, milliseconds per query:

| Stage | Mean | p50 | p95 |
|---|---:|---:|---:|
| Native BM25 | 71.044 | 70.566 | 104.421 |
| BM25 temporal increment | 3.437 | 3.308 | 4.403 |
| GPU E5 dense | 9.814 | 9.344 | 10.360 |
| Dense temporal increment | 2.526 | 2.508 | 2.702 |
| GPU E5 hybrid | 87.140 | 87.072 | 121.217 |
| Hybrid temporal increment | 5.718 | 5.673 | 6.064 |

Temporal timings include the adapter's query-time date parsing, diagnostic collection
and pool verification; they are not a replication of the paper's exact timing boundary.
Base and temporal stages are collected separately. Their mean sums **74.481 / 12.341 /
92.858 ms** are modeled additive costs, not observed live pipeline means/p95. Never add
stage p95s. No service throughput/load test or CPU-only encoder comparison was run.

Actual in-collection index builds: BM25 **218.283 ms**, FAISS **85.584 ms**,
combined hybrid **303.867 ms**. These exclude cached corpus encoding. Auto stages build
no index. An independent CPU rebuild is also profiled in the aggregate, with its own
scope. The first prototype's callback index-build zeros were not build measurements;
the final collector records actual source-stage build times explicitly.

E5 runs on the RTX5060 Ti, Torch2.10/CUDA12.8, FP32/eager/TF32-off. Final encoder peak
allocated/reserved **459.794/486 MiB**; zero query inputs at the 512-token cap. This is
allocator telemetry for the cached-corpus query stage, not total process VRAM or a new
corpus-build peak. No cross-encoder or second large model is resident. Exact numerical
versions, asset/source hashes and thread flags are retained in metadata.

Two complete GPU collections are retained: `artifacts/m4/timelyrag_reference_validation495_v1`
and `_v2`. The second adds a generic native-tie callback guard and actual build-time
telemetry; ranking parameters/formula are unchanged. Initial collector source is preserved
in ignored `artifacts/m4/timelyrag_reference_source_v1` with snapshot
`d07f1e3dc413d4ff20d64fb9ebcd238af161ac6f7e3851eb6a87cb7e1ccebef4`.
Final source snapshot **`be142ff248e360c2d016d1f3d48af7745a78987ba2eb907290f518eaa43b3a9b`**,
based on clean HEAD `c71b035` plus the recorded uncommitted stage changes; final manifest
**`7be604b950735ad4993b07b0c06d8d65214a5c26df50401869f72345c6ff751c`**.

All full-corpus score matrices, TREC scores/ranks, qrels, configs and alpha diagnostics
match across GPU collections. All quality metrics and group intervals match; measured
timings vary and remain recorded. CPU replay also matches all1,485 automatic/top100
pairs and original top10s. Raw source files, original datasets and model assets are intact.
**206 CPU tests pass with explicit reference integration enabled**, including native
parity, label counterfactuals, source guards, namespace restoration, no config side effects,
unknown/equal/later date behavior, candidate preservation, JSON and callback/tie contracts.
Ruff lint/format pass. Existing third-party SQLAlchemy deprecation warning only.

Final aggregate: [`m4_timelyrag_reference_validation495_v2.json`](../experiments/results/m4_timelyrag_reference_validation495_v2.json).
The earlier `_v1.json` retains the initial prototype's measured provenance. No value is
silently replaced to improve quality. Final plot PNG/SVG:
`artifacts/m4/analysis/timelyrag_reference_validation495_verified_v2.*`.

Final local MLflow FINISHED IDs:

- `bm25`: `cdbfa0067f0d43b6a17153148e60ba12`.
- `bm25_timelyrag_auto`: `599fef4087d942e9aceb38486d526d5e`.
- `comparison`: `e4bcabb2988340e49c9d976d3b27d8b5`.
- `dense`: `37c08effa69844fbaba3f6e2edb92d19`.
- `dense_timelyrag_auto`: `b19b4149ea7542ff96d5e6a77a142738`.
- `hybrid_rrf`: `8c6c806c96e548c8be0fad288fe618c4`.
- `hybrid_rrf_timelyrag_auto`: `5f2a89dcc6b74678a2cb5c2c3da19b78`.

Five explicit root metadata artifacts plus aggregate/PNG/SVG are attached to the
comparison run; each of six stage runs uses the standard validated evidence list,
excluding failure text. Full score matrices and model weights are never uploaded.
Repeated logging returns exactly the same seven IDs. Initial v1 runs remain separate.
Ordinary checks pass **202 tests with 4 opt-in cases skipped**; enabling reviewed-source
integration passes **all206**. This is not an unrun-check claim: both commands were executed.



## Reproduce

Existing pinned model/data assets and reviewed source files are required. No new large
model dependency or FlagEmbedding installation is introduced. Keep all existing extras:

```bash
uv run --frozen --extra tracking --extra dense --extra routing ruff check .
uv run --frozen --extra tracking --extra dense --extra routing ruff format --check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
TEMPORALPROOF_REFERENCE_TESTS=1 \
  uv run --frozen --extra tracking --extra dense --extra routing pytest -q
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  uv run --frozen --extra tracking --extra dense --extra routing \
  python -u -m temporalproof.evaluation.timelyrag \
  --output artifacts/m4/timelyrag_reference_validation495_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --extra tracking --extra dense --extra routing \
  python experiments/analyze_timelyrag_reference.py \
  --run artifacts/m4/timelyrag_reference_validation495_v2 \
  --earlier-run artifacts/m4/timelyrag_reference_validation495_v1 --measured-on 2026-11-28 \
  --output experiments/results/m4_timelyrag_reference_validation495_v2.json
uv run --frozen --extra tracking --extra dense --extra routing \
  python experiments/plot_timelyrag_reference.py \
  --results experiments/results/m4_timelyrag_reference_validation495_v2.json \
  --output artifacts/m4/analysis/timelyrag_reference_validation495_verified_v2.png
uv run --frozen --extra tracking --extra dense --extra routing \
  python experiments/log_timelyrag_reference.py \
  --run artifacts/m4/timelyrag_reference_validation495_v2 \
  --results experiments/results/m4_timelyrag_reference_validation495_v2.json \
  --figure artifacts/m4/analysis/timelyrag_reference_validation495_verified_v2.png
```

Choose new output paths. Ordinary CPU tests skip four explicit upstream-source integration
cases unless `TEMPORALPROOF_REFERENCE_TESTS=1`; no normal test downloads or invokes neural
inference. The optional cases require only local reviewed source and CPU. A model/corpus
fetch is never hidden in this evaluator. Original source checksums are mandatory.

Raw scores, per-query diagnostics, failure text and model assets remain ignored. Explicit
local MLflow artifact lists exclude raw failure text, full score matrices and weights.
Dataset/source redistribution terms and model decontamination remain unresolved.

## Next increment

Hybrid remains the fast reference. Before further router fits, audit derived query-event
selection and fallback coverage on training data, freeze an explicit derived-time policy,
and evaluate it without turning unknown into validity. Any future changed-utility router
must still generate targets and choose policies within training folds only. Exact paper
backbone reproduction with BGE-M3 remains separate and unrun; its assets are not present.
No automatic method adoption, final-test evaluation, API/UI or generation is introduced.
