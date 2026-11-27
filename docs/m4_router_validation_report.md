# M4 — Frozen train500 router on development validation

## Protocol frozen before deep outcomes

The model is the primary full-feature, penalty-0 balanced logistic trained on the
previously selected **500 train queries / 493 groups**. No fitting, calibration,
threshold search or target generation is performed on validation. The original
train collection/model artifacts are unchanged.

The predeclared configuration is
[`m4_frozen_validation_company_policies.json`](../configs/experiments/m4_frozen_validation_company_policies.json).
It pins the training manifest `1ec906f15ff132ccda05f1172c169c68f5d2ffa243f6fd6541d1e2aa5784982a`,
final model `4a8d3265bf6545640a14060a76cfc67872e4e70d0b8dcd875777310b19d60755`
and training config `7db69c1f1c46c80738ee36285f038effa6c07d9e271eddd38bf25e074f4adfab`.
The loader checks all seven training files, canonical train IDs/groups, primary policy,
feature schema, and the original feature extraction/JSON inference implementation hashes.

Evaluation covers **all 495 validation queries / 462 source groups** against the full
30,000-passage Company Policies corpus. Source archive, grouped split, seed 1729,
tokenizer, full-positive binary qrels, E5/KO model revisions, FP32/eager/TF32-off,
batch 16, max length 512, top-100, and equal RRF offset 60 remain fixed. BM25 uses
k1 1.5 / b .75 / epsilon .25. Unknown dates stay unknown; no timestamp filtering or
qrel changes. The method is retrospective evidence retrieval, not historical known-at
simulation or validated event-time applicability.

Primary: **10% query cap**, `floor(495 * .1) = 49` calls. Rank all validation scores
from the single frozen model, descending, with query ID ascending for ties. Unlike OOF,
there is one model trained without any validation labels, so a whole-cohort quota is
legitimate. This is an offline cohort budget, not an online per-request latency guarantee.
Seeded random and lowest top-10-overlap-first agreement controls use exactly 49 calls;
equal query counts do not imply equal measured cost. Secondary caps 0/.25/.5/1,
threshold .5 and the original disagreement/year rules are diagnostics; none replaces
the primary point after inspecting validation. Class-weighted scores are uncalibrated.

Features, scores, selected IDs, model/config/plan and fast-manifest hashes are checkpointed
**before loading the deep model**. The reranker subsequently evaluates all 49,500 frozen
query/candidate pairs offline to estimate every fixed policy. Only query/passage text
enters the scorer. Validation gain labels are never constructed as training artifacts.
Models run sequentially; weak-reference collection and residual allocator bytes are
recorded before the reranker loads.

This is **development validation**, including the original validation100 used for earlier
retrieval/reranker choices. It is not an independent final test. The 447-query test split
remains unevaluated. Group bootstrap intervals use 2,000 source-group resamples with fixed
predictions, without refitting; secondary comparisons are exploratory.

Measured on 2026-11-27: AMD Ryzen 5 7500F (12 logical CPUs), RTX 5060 Ti
(16,311 MiB reported), driver 610.88 / Torch CUDA 12.8, Linux WSL2, Python 3.13.11.
All model stages are sequential, concurrency 1, with OMP/MKL/OpenBLAS threads set to 1
and three warmups. Exact package/asset versions are preserved in the aggregate.

## Measured results

Complete GPU run and evidence verification finished on **2026-11-27**. The machine-readable
[aggregate](../experiments/results/m4_router_validation495_v1.json) preserves all predeclared
points and provenance. This is a negative primary routing result.

| Policy | Deep calls / 495 | nDCG@10 | Recall@10 | Recall@20 | Modeled mean ms/query |
|---|---:|---:|---:|---:|---:|
| Hybrid reference | 0 | 0.198972 | 0.242088 | 0.292929 | 87.299 |
| Always Korean deep | 495 | 0.148561 | 0.186700 | 0.238552 | 1220.241 |
| Seeded random 10% cap | 49 | 0.194573 | 0.234848 | 0.287710 | 203.016 |
| Agreement 10% cap | 49 | 0.198817 | 0.242761 | 0.292593 | 200.530 |
| Frozen logistic 10% cap (primary) | 49 | 0.190746 | 0.238384 | 0.289226 | 186.391 |

Recall@100 is **0.417744** for every policy: reranking cannot recover passages missing
from the fixed candidate pool. **211/495** pools contain no positive judgment. Full deep
has **54 gains / 123 regressions / 318 unchanged** queries. Primary calls include **11
gains / 17 regressions / 21 unchanged**; the matched agreement control has 4/4/41.
Thirteen primary calls, versus 28 agreement calls, have no gold in their pool. These
post-hoc diagnostics are not routing features or newly created training targets.

Primary minus hybrid nDCG@10: **−0.008226**, paired group interval
**[−0.016980, −0.000566]**. The conditional interval is wholly below zero on this
development set. Primary minus random: **−0.003827**, interval
**[−0.013632, 0.004883]**; minus agreement: **−0.008071**, interval
**[−0.017396, 0.000287]**. The latter intervals include zero: the matched-control mean
ordering is not independently confirmed superiority. Always deep minus hybrid is
**−0.050411**, interval **[−0.070763, −0.032468]**. No learned advantage is established.

Diagnostics do not rescue the primary result: logistic 25%/50% caps reach nDCG@10
0.187326/0.169536, threshold .5 routes 192 and reaches 0.177559. Agreement 25%/50%
reaches 0.195999/0.191492. None replaces the predeclared primary. Year counts occur in
491/495 queries, so this feature does not define a sparse temporal-hard slice. The
319 low-agreement queries have mean deep gain −0.016343; the 491 explicit-year queries
have −0.049395. These slices are exploratory metadata diagnostics, not temporal gold.

Largest deep regressions include query IDs 705/2351/2449 (1 → 0 nDCG@10), 1132
(0.967468 → 0) and 2800 (0.919721 → 0). Largest gains include 2077
(0.237198 → 1), 2571 (0.301030 → 1) and 1584 (0.386853 → 1).
Full source text and unchanged qrels remain local; these are numeric cases, not manual
labels or claims about answer correctness.

Quality–modeled-cost PNG/SVG: `artifacts/m4/analysis/router_validation495_verified.*`.
The plotted curves all decline as more queries are reranked at these measured points.
Hybrid stays the fast reference. Next increment: inspect the deep/routing failures to
predeclare the next **train-only** hypothesis before spending on more collection or
opening final test; do not adopt another validation sensitivity point after the fact.

### Reproducibility and resources

Source snapshot: `5c83e6deccd2613780e5beadf1ece564cb954d6a089eb7dc5e1880eb200bdcf8`;
run manifest: `9d89e4079b6744f87001023c12df13073dd08550a9ace13037d34e84bb463892`.
Measured code was based on `285d454` with the reported uncommitted validation changes;
all exact source hashes are in the aggregate. Original train-model evidence remains pinned.

All root and child output checksums passed. CPU replay exactly reproduces scores,
decisions, metrics and intervals. The original validation100's fast **and deep** scores,
rankings and qrels are identical to the earlier M2/M3 runs. The serialization-only
restart has identical config/model/features/decisions/scores. Only the corrected run
completed all 495 deep queries; no second complete 495-query GPU replay was performed.

Fast mean/p50/p95: **87.299 / 87.115 / 122.577 ms**. Incremental deep:
**1132.942 / 1112.628 / 1516.032 ms**. Feature extraction mean **1.801 ms** is already
in fast. Warm parsed-model numeric CPU prediction mean/p50/p95:
**0.009056 / 0.008710 / 0.009624 ms**, excluding parsing/loading/feature extraction/HTTP.
Primary modeled mean **186.391 ms**, agreement **200.530 ms**, always-deep **1220.241 ms**;
these are offline stage sums, not observed live routed latency or matched-time budgets.

Fresh corpus embedding cache construction in the interrupted attempt took **32.340 s**;
the completed run reused it, with **314.344 ms** measured BM25/FAISS build time. No new
model download. Completed-run allocator peaks (allocated/reserved): E5 query stage
**459.794/486 MiB**, deep **2706.176/4590 MiB**. Corpus-build peak is not included in
the completed-run E5 query peak. Residual allocation between collected models:
**9,568,256 bytes**; no simultaneous model references. Deep profile records **49,800**
pairs including three 100-pair warmups, zero at the token cap.

Local MLflow FINISHED runs: fast `536e5d8412a84faeb8faf98ac84b78d1`,
deep `fa31b63560844d8581b7c68b7bae53eb`,
frozen router `d3258e57ad3a43709832aa94b3a1c499`. Standard stage runs and an explicit
seven-file router metadata list plus aggregate/figures are logged. No raw failure text
or model parameters are uploaded. Repeated logging returns the same run IDs.


## Reproduce locally

Existing verified data and model assets are required; no download or GPU inference occurs
in ordinary unit tests. Commands from repository root:

```bash
uv sync --frozen --extra tracking --extra dense --extra routing --group dev
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --extra tracking --extra dense --extra routing pytest -q
uv run --frozen --extra tracking --extra dense --extra routing ruff check .
uv run --frozen --extra tracking --extra dense --extra routing ruff format --check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_DISABLE_PROGRESS_BARS=1 \
  uv run --frozen --extra tracking --extra dense --extra routing \
  python -u -m temporalproof.evaluation.router_validation \
  --output artifacts/m4/router_validation495_v2
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --extra tracking --extra dense --extra routing \
  python experiments/analyze_router_validation.py \
  --run artifacts/m4/router_validation495_v2 \
  --interrupted-run artifacts/m4/router_validation495_v1 \
  --measured-on 2026-11-27 \
  --output experiments/results/m4_router_validation495_v1.json
uv run --frozen --extra tracking --extra dense --extra routing \
  python experiments/plot_router_validation.py \
  --results experiments/results/m4_router_validation495_v1.json \
  --output artifacts/m4/analysis/router_validation495_verified.png
uv run --frozen --extra tracking --extra dense --extra routing \
  python experiments/log_router_validation.py \
  --run artifacts/m4/router_validation495_v2 \
  --results experiments/results/m4_router_validation495_v1.json \
  --figure artifacts/m4/analysis/router_validation495_verified.png
```

Output paths must be new. For a fresh reproduction, choose new paths and omit
`--interrupted-run` unless comparing the original interrupted collection. Raw benchmark
text, qrels, per-query features/scores and model weights stay in ignored local artifacts;
the committed aggregate contains metadata/metrics only. Dataset redistribution terms
remain unresolved.

## Validation and limitations

**167 CPU tests pass**, including synthetic full validation collection with fake scoring,
train/test partition refusal, exact complete-cohort selection, training-source integrity,
wrong/rehashed held-out/group/policy/source guards, no-fit monkeypatch, quota/tie rules,
changed validation qrels leaving pre-deep decisions unchanged, checkpoint tamper refusal,
fixed candidate recall, CPU replay and JSON serialization. Ruff lint/format pass.

The first attempt was deliberately interrupted before complete deep output after finding
NumPy integer counters that needed conversion to JSON-native integers. Its artifacts and
log remain under `router_validation495_v1`; it has no complete deep/policy result. The
corrected collector restarts in `v2` with the same frozen model/config. Initial cache
rebuild was required because adding the routing extra changed the lock fingerprint;
model asset/configuration/numerical package versions did not change. The second attempt
uses that verified cache. The aggregate verifies identical features/model/config/decisions
and numeric router scores across this serialization-only restart.

Modeled means sum measured fast (including feature extraction), selected incremental deep
and warm numeric CPU router inference. They exclude quota sorting, parsing/loading,
network/HTTP and concurrency effects; no live routed p95/throughput claim is made.
Host/process peak memory is not measured; GPU allocator peaks exclude driver/other processes.
The scientific plot is a descriptive budget curve, not a proof of Pareto superiority.

Grouping covers exact/template/near-query matches and shared exact passages. Residual
semantic/version-family leakage has not been exhaustively excluded. The Korean reranker's
full training corpus is undisclosed; no independent model decontamination claim is made.
Explicit-temporal feature ablation is available on training OOF only, not this validation
run. This experiment does not establish temporal novelty or applicability correctness.
