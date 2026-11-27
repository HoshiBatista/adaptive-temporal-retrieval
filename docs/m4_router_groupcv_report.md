# M4 — first learned router, grouped OOF on train500

Measured **2026-11-27**. A logistic router is implemented and fitted, but **its quality
advantage is not established**. At the predeclared 10% query cap it reaches nDCG@10
**0.198929** versus hybrid **0.197016**, and falls below the agreement-ranked control's
observed **0.204155** at the same number of reranker calls. Paired intervals include zero.
This is grouped out-of-fold evaluation on training queries, not final validation/test,
production readiness or evidence of the main ASTR contribution.

## Predeclared experiment and two source snapshots

- [Data configuration](../configs/experiments/m4_train500_company_policies.json): 500
  hash-selected train queries, seed 1729, original corpus/split, hybrid top 100 and the
  pinned Korean cross-encoder. Freshly measure all 500 queries; do not mix the earlier
  100-query costs into a larger sample. The 500 contain the original 100 as a prefix.
- [Router configuration](../configs/experiments/m4_logistic_groupcv_company_policies.json):
  5 group folds, fixed seed, C=1, balanced class weights, LBFGS, max 2,000 iterations,
  tolerance 1e-6, median imputation with missing indicators/empty-column preservation,
  standard scaling. No search over C, solver or decision threshold.
- Targets use penalties **0 / 0.01 / 0.05 nDCG per incremental rerank second**. Feature
  sets are full (15 fields) and without the six explicit temporal fields (9 fields).
- Predeclared primary point: **full features, penalty 0, at most 10% queries per fold**.
  Fixed score threshold 0.5 and query caps 0/0.1/0.25/0.5/1 are separate diagnostics.
  They are not selected after inspecting the OOF outcomes.

GPU collection ran from a clean detached checkout of
`30945ffaf14d91f53c151630b882be5031464060`, with source snapshot
`eaff8985ec0882614d9bdf417f9ac5cae4170c1da1193ac0bfc191247e770bf7`.
Existing model/data/cache paths were supplied explicitly. Router implementation proceeded
in the main workspace while that data process stayed on the committed collector. The
actual package versions were unchanged; adding the routing extra changed lock metadata
only. Data and router source/dependency hashes are recorded independently. No new model
weights were downloaded, and no neural model was trained.

The loader checks all source artifact hashes, archive/split/config/source pins, clean
collector revision, pre-deep checkpoint, exact train selection, canonical group IDs and
agreement of outcome gains/times with the original per-query measurements. Features
have the existing strict pre-deep schema; IDs are join/fold metadata only. There are
**500 queries / 493 source groups**, with held-out fold sizes **101 / 100 / 100 / 101 / 98**.
Related selected queries stay together; the source grouping is still not an exhaustive
semantic/version-family audit. The full unlabeled corpus remains shared.

## Fold-local fitting and decisions

Each imputer, missingness pattern, scaler, class weighting and logistic fit sees only
that fold's training rows. Each scored query's source group is excluded. The primary
quota is allocated **within each held-out fold**, by its scores alone, using
`floor(fold_size * cap)` and query-ID ties. It never uses gains, actual reranker costs or
other folds' scores. A global OOF score ranking is avoided because other folds' models
can have seen this fold's labels. Group-fold behavior follows the
[GroupKFold contract](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html);
preprocessing is kept inside training fits as in sklearn's
[data-leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

At the 10% cap this yields **49 calls = 9.8%**, not exactly 50. Random and agreement-ranked
budget controls use the same per-fold caps. They therefore match query counts, not exact
milliseconds: selected query lengths and measured deep costs can differ.

Class-weighted classifier outputs are **routing scores, not calibrated probabilities**.
ROC-AUC is a diagnostic, not the research objective. There is an explicit constant-score
fallback for single-class training folds; none was needed in the 30 actual fold fits.
One primary model is also fitted on all 500 selected train queries for later validation;
its in-sample predictions are not used as OOF results.

Models are persisted as validated version-1 JSON parameters, without pickle or remote
code. The cached JSON predictor is checked against the sklearn pipeline, including
unseen missing values. `final_model.json` supplies scores; the allocation rule remains
in the separate experiment configuration. Model weights/fold checkpoints stay outside Git.

## Collection and target distribution

| Fixed path, same 500 train queries | nDCG@10 | Recall@10 | Recall@20 | Recall@100 |
|---|---:|---:|---:|---:|
| Hybrid | 0.197016 | 0.251133 | 0.306767 | 0.426433 |
| Always Korean reranker | 0.162597 | 0.199467 | 0.247467 | 0.426433 |

Reranking helps **71**, harms **121**, and leaves **308** queries unchanged. Target
positives at penalties 0/0.01/0.05 are **71/71/67**. The first two labels are identical,
so corresponding fitted variants are identical rather than independent findings.
**211/500 pools contain no judged positive**; no reranker/router over these fixed pools
can recover that evidence. Earlier MiniLM/Korean negative validation results are retained.

The complete corpus/qrel scale, date uncertainty, retrospective full-corpus scope and
model-training-data disclosure caveats from the [pilot](m4_train_pilot_report.md) and
[Korean model report](m3_korean_reranker_report.md) still apply. No event-time correctness,
known-at availability or benchmark decontamination claim is introduced.

## OOF quality and modeled cost

| Control / policy | Deep calls | nDCG@10 | Modeled mean stage cost |
|---|---:|---:|---:|
| Always hybrid | 0 | 0.197016 | 93.563 ms |
| Always rerank | 500 | 0.162597 | 1,262.242 ms |
| Fixed overlap <0.2 rule | 326 | 0.197966 | 858.235 ms |
| Seeded random, 10% cap | 49 | 0.194740 | 207.429 ms |
| Agreement-ranked, 10% cap | 49 | 0.204155 | 207.566 ms |
| **Primary logistic, full, penalty 0, 10% cap** | **49** | **0.198929** | **195.949 ms** |
| Logistic without explicit temporal fields, penalty 0, 10% cap | 49 | 0.191997 | recorded in aggregate |

Primary Recall@10/20/100 is **0.249300 / 0.301100 / 0.426433**. Its small nDCG increase
does not extend to Recall@10/20. The agreement-budget control reaches
**0.257300 / 0.310600 / 0.426433**. Primary calls help 15 queries, harm 13 and leave 21
unchanged. Accuracy on the imbalanced binary target would conceal this quality/cost mix.

| Predeclared primary contrast | Mean nDCG delta | Conditional paired group-bootstrap 95% interval |
|---|---:|---:|
| Primary − always hybrid | +0.001913 | [−0.006638, 0.010611] |
| Primary − matched random | +0.004189 | [−0.005810, 0.013785] |
| Primary − matched agreement rank | −0.005226 | [−0.015983, 0.004889] |
| Full − no explicit temporal features, penalty 0 / primary cap | +0.006932 | [−0.002086, 0.015823] |

All intervals contain zero. They resample whole source groups with query-count-weighted
means, seed 1729, 2,000 resamples and **fixed OOF predictions**, without refitting. They
are conditional diagnostic intervals, not population guarantees or a substitute for
independent validation/test. No adjustment for secondary multiple comparisons is claimed.

Full-feature score ROC-AUC is 0.628222 versus 0.557405 without explicit temporal fields;
pooling scores from multiple fold models is itself diagnostic. These numbers do not
establish better retrieval, calibrated gain probabilities or causal temporal reasoning.

The fixed threshold-0.5 full model routes 185 queries and scores 0.196306, below hybrid.
At the predeclared 25% cap it scores 0.200393 with 124 calls. The penalty-0.05/full model
at the primary cap scores 0.199624. None replaces the original primary operating point
post hoc. [Aggregate evidence](../experiments/results/m4_router_train500_groupcv_v1.json)
retains all six specified variants, controls, budgets and regressions.

Costs above sum separately measured fast/deep stages. They include measured warm CPU
JSON-router inference where applicable, but **are not measured complete routed latency**.
Fast includes feature extraction; a feature-free fast path was not independently timed.
Quota allocation/sorting overhead is not measured. There is no claim of exact matched
time budgets, live p95, service throughput or concurrent
neural-model residency. The standalone PNG/SVG figure compares these modeled quantities.

## Resources, persistence and replay

Same RTX 5060 Ti 16 GB, Ryzen 5 7500F / WSL2 / Python 3.13.11, PyTorch 2.10.0,
Transformers 5.5.0, CUDA 12.8. Neural inference remains FP32/eager/TF32-off, batch 16,
max 512, query concurrency 1, three warmups, thread environment variables 1. The verified
E5 passage cache was reused. Encoder removal was verified before loading the reranker.

Actual collection times: fast mean/p95 **93.563 / 133.831 ms**, incremental deep
**1,168.680 / 1,640.079 ms**. Feature extraction averages **1.830 ms**, already in fast.
Fast allocator peak allocated/reserved **460.514 / 486 MiB**; deep
**2,728.898 / 4,626 MiB**. These are allocator counters, not full-process memory. None
of 50,300 pairs including warmups reached the 512-token limit. All 50,000 scored pairs
are saved locally; original 100-query rankings/scores match the first 100 in this collection.
No second complete 500-query GPU run was performed.

Routing uses pinned **scikit-learn 1.9.1** through the explicit `routing` extra. Numerical
package versions were retained. Primary warm CPU JSON prediction mean/p50/p95 is
**0.009164 / 0.008603 / 0.010073 ms**. This covers numeric vector construction, imputation,
scaling and scoring of one already extracted feature object; it excludes parsing/loading
JSON, feature extraction, HTTP/queues and all GPU work. Fit/measurement details are local
artifacts; do not interpret this microbenchmark as request latency.

A complete CPU replay produces exact OOF scores, fold assignments, fitted model states,
policy choices, quality/intervals and final JSON model. Plans/final model are byte-identical;
fit/prediction times vary and remain recorded. Serialization also reproduces every held-out
prediction from the actual fold states. No validation/test queries were evaluated or used
for fitting, thresholds or quota selection.

Local MLflow runs are FINISHED:

- Data fast `f33af6d892db41d098858d92b0c010a0`;
- Data deep `0498406345ce4eb48db41c563688f4b7`;
- Grouped OOF router `a47d56cea741452d82e64501cf051aa0`.

The original frozen collection files are not rewritten to add tracking IDs; a separate
local sidecar links their logs. Router logging validates its explicit artifact allowlist,
uses the local file store and is idempotent by manifest hash. `analysis/` holds aggregate,
PNG/SVG and provenance. Raw passage/failure text is excluded. Weights, features, targets
and OOF prediction files remain local; Git receives code/configs and aggregate metadata.

Validation: **157 CPU tests pass**, Ruff lint/format checks pass. Seventeen new synthetic
cases cover group/label independence, fold-local preprocessing under held-out outliers,
JSON inference on unseen missingness/all-empty columns, explicit single-class fallback,
forbidden model columns, within-fold budgets, clustered intervals, group-overlap rejection
and canonical/hash-checked train-only loading even after rehashing malicious metadata.
No model/network/GPU is needed for normal tests; routing tests skip without sklearn.
Existing MLflow/SQLAlchemy deprecation warning remains. Full source/lock hashes and
separate data/router provenance are verified in the aggregate.

## Reproduction and next step

With the original frozen collection already present:

```bash
uv sync --frozen --extra tracking --extra dense --extra routing --group dev
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --extra tracking --extra dense --extra routing temporalproof-router-cv \
  --pilot-run artifacts/m4/company_policies_train500_v1 --mlflow \
  --output artifacts/m4/my_router_groupcv
```

Replay into another new directory without `--mlflow`; outputs are never overwritten.
Training's source guard deliberately requires the pinned clean collector, not an arbitrary
later working-tree run. To recreate the data, use a clean detached checkout of `30945ff`,
sync its original frozen `tracking`/`dense` environment and run its existing pilot with the
new 500-query config and explicit archive/split/encoder/reranker/cache/output paths pointing
into the main research workspace. The actual collection used the existing environment
whose numerical package versions matched that lock. Its data plan/manifest records both
configuration and original source hashes. Model/data downloads remain separate explicit
commands; this collector performs no hidden downloads.

Aggregate/replay verification and the figure are reproducible without committing model
weights or raw text:

```bash
uv run --frozen --extra tracking --extra dense --extra routing python experiments/analyze_router_cv.py \
  --run artifacts/m4/router_train500_groupcv_v1 \
  --replay artifacts/m4/router_train500_groupcv_replay_v1 \
  --data artifacts/m4/company_policies_train500_v1 --measured-on 2026-11-27 \
  --output artifacts/m4/my_router_aggregate.json
uv run --frozen --extra tracking --extra dense --extra routing python experiments/plot_router_cv.py \
  --results artifacts/m4/my_router_aggregate.json --output artifacts/m4/my_router_curve.png
```

Next freeze the primary logistic policy and the simple budget control for a validation
comparison, retaining this inconclusive/negative learned-router result. Do not tune on
final test or relabel unjudged passages. A gain-regression alternative or richer features
would be a separately predeclared experiment, not a retroactive replacement of this one.
M4 remains incomplete: no independent validation/test routing, real routed-service p95,
full-train fit, TEMPO transfer or validated true version-chain features exist yet.
