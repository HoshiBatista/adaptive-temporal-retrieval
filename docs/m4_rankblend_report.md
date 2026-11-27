# M4.2 — Candidate/reranker failures and conservative rank blending

Measured **2026-11-27** using the existing immutable train500 and validation495 rankings.
This is a CPU-only cached experiment, with **no new neural inference or downloads**.
The [aggregate](../experiments/results/m4_rankblend_train500_v1.json) contains metadata
and metrics only. Source text and unchanged qrels remain in ignored local artifacts.

## What the failure audit establishes

| Pool coverage | Train500 queries | Validation495 queries |
|---|---:|---:|
| No listed positive in top100 | 211 | 211 |
| Some, but not all, listed positives | 147 | 152 |
| All listed positives | 142 | 132 |

Recall@100 is fixed at 0.426433 on train and 0.417744 on validation. A reranker cannot
recover absent passages. The label-aware best ordering *within the existing pool* has
mean nDCG@10 ceiling 0.461027 / 0.453718 (train/validation). This is an **offline oracle
bound**, not an inference feature, trained system, temporal gold or deployable policy.

Candidate coverage does not explain every failure: on the 132 validation queries with
**all** positives already present, hybrid scores 0.478254 and raw deep 0.359584. On
152 partial-coverage queries, they score 0.232643 and 0.171530. Raw deep moves 106
listed-positive passages out of the original top10; its first result came from outside
hybrid's top10 on 194 queries. These counts describe ranking movement, not new labels.

Numeric and local textual inspection of extreme cases supports distinct hypotheses:

- Query 705: the query explicitly asks about a later policy, while the promoted passage
  repeats an earlier employment/leave condition. This illustrates a requested-period
  mismatch, without claiming that insertion metadata encodes validity.
- Query 2449: the promoted passage mixes another period and filing condition with
  broadly similar employment terms. Query 1132 also changes the employment-transition
  scenario. These are exploratory semantic/temporal mismatch observations.
- Query 2351: the promoted cross-query passage looks answer-like but is not among the
  published positives. This exposes a **possible incomplete-judgment limitation**, not
  evidence that a new gold label is correct. Keep official qrels unchanged.
- Queries 2077 and 2571 illustrate useful promotions of listed positives. The scorer
  does sometimes help; the problem is selective utility, not universally wrong scores.

The automatic taxonomy uses observed pool coverage and ranks only. The interpretations
above are qualitative examples, not a manually labeled dataset or validated error-class
counts. `train_inspection.json` / `validation_inspection.json` hold ten selected cases
per partition locally; no passage/query text is committed or uploaded to MLflow.

## Predeclared experiment

Configuration: [`m4_conservative_rankblend_company_policies.json`](../configs/experiments/m4_conservative_rankblend_company_policies.json),
SHA256 `adf67b8e29b4039fa7bd879006b8ede1a67fef372dfa749ab898335f0e8a2c41`.
The hypothesis was designed **after observing negative development validation**. It is
not an independent confirmation or final-test result.

For each identical top100 pool, combine rank positions rather than incompatible raw
RRF scores/cross-encoder logits:

```text
score(doc) = (1 - weight) / (60 + hybrid_rank(doc))
             + weight / (60 + cross_encoder_rank(doc))
```

Sort by score descending, document ID ascending. Weight zero preserves the exact hybrid
ranking/scores and requires no deep call; weight one preserves raw deep as a control.
The predeclared selection grid is **0 / .05 / .10 / .20 / .30**. No validation grid
sweep, timestamp filtering, corpus/candidate changes, neural training or score calibration.
Every **positive** weight still requires all 100 expensive candidate scores; smaller
weight does not make the neural stage cheaper.

Reuse the original five grouped folds, seed 1729, on **500 train queries / 493 groups**.
In each fold select a weight by mean nDCG@10 on the **other training groups**, then
apply it to held-out queries. Ties within 1e-12 prefer the smaller weight. This evaluates
the selection procedure; all-train grid scores are explicitly in-sample, not OOF results.
The selected fold weights are **.10 / .20 / .20 / .10 / .20**, with held sizes
**101 / 100 / 100 / 101 / 98**.

Select the final weight on all train500 and save the config/plan/OOF/fold/final-selection
hashes **before opening the cached validation ranking/outcome bundle**. The final weight is **.20**. Only this
one new weight is evaluated on all **495 validation queries / 462 groups**. Test447 is
not evaluated. Source archive, qrels, top100 candidates, E5/KO weights, tokenizer,
truncation and earlier stage measurements are unchanged.

Primary is the **fold-selected all-query blend**. Applying the new ranking to the old
frozen logistic/agreement/random 49-query routes is a predeclared **secondary transfer
diagnostic**. The old router was trained for raw-deep utility, not the changed utility;
no new router or targets were fitted/generated in this stage. Do not promote the
best-looking transfer result to the primary after observing validation.

## Results

Train grid, in-sample nDCG@10:

| Weight | 0 | .05 | .10 | .20 | .30 |
|---|---:|---:|---:|---:|---:|
| Train500 | 0.197016 | 0.196985 | 0.198075 | 0.198126 | 0.193297 |

Grouped OOF selection reaches **0.196354**, versus hybrid **0.197016**. Delta
**−0.000661**, conditional paired group interval **[−0.006120, 0.005015]**. The small
all-train improvement does not survive OOF selection. OOF has 53 gains / 59 regressions /
388 unchanged; Recall@10/20/100 **0.248967 / 0.311767 / 0.426433**.

| Validation policy | Deep calls | nDCG@10 | Recall@10 | Recall@20 | Source-stage mean ms |
|---|---:|---:|---:|---:|---:|
| Hybrid | 0 | 0.198972 | 0.242088 | 0.292929 | 87.299 |
| Raw deep | 495 | 0.148561 | 0.186700 | 0.238552 | 1220.241 |
| Frozen .20 blend | 495 | 0.196351 | 0.241414 | 0.298822 | 1220.241 |
| Old logistic routes + blend | 49 | 0.199371 | 0.243098 | 0.294613 | 186.382 |
| Old agreement routes + blend | 49 | 0.198656 | 0.241414 | 0.295286 | 200.530 |
| Old random routes + blend | 49 | 0.198800 | 0.241246 | 0.295623 | 203.016 |

Recall@100 remains **0.417744** for every validation policy. Blend has 41 gains /
55 regressions / 399 unchanged versus hybrid, compared with raw deep's 54/123/318.

Frozen blend minus hybrid: **−0.002621**, interval **[−0.008260, 0.003274]**.
It greatly reduces damage compared with raw deep: delta **+0.047790**, interval
**[0.032093, 0.065265]**, while retaining raw deep's expensive calls. This does **not**
establish superiority to hybrid.

Old logistic routes + blend minus hybrid: **+0.000399**, interval
**[−0.001881, 0.002920]**, with 6 gains / 6 regressions / 483 unchanged. Against the
original same-route raw-deep policy (0.190746), the change is **+0.008625**, interval
**[0.001906, 0.016005]**. This secondary improvement reduces the old policy's damage;
it is neither a new trained-router result nor confirmed improvement over hybrid.
Agreement/random transfer intervals against hybrid also include zero.

Intervals use 2,000 source-group bootstrap resamples conditional on fixed selected
weights/folds/predictions, without redoing selection or fitting. Secondary comparisons
are exploratory and not multiplicity-corrected. Earlier model-selection validation100
is part of validation495; no population, temporal-applicability or final-test claim.

Cost is **reused source fast + selected incremental deep** at concurrency 1. New blend
sorting and router overhead are unmeasured and excluded, not zero. These are not newly
measured pipeline means/p95/throughput. This CPU replay does not measure GPU memory or
inference speed. The smaller transfer total than the earlier 186.391 ms excludes the
old 0.009056 ms router microbenchmark; it does not represent a speedup.

## Integrity, reproduction and handoff

Source snapshots/pins:

- Train collection manifest `b3f0738a8e635f09930ae7aff8dfd368502ba73695c56fcedffe2a3f1d213049`.
- Validation collection manifest `9d89e4079b6744f87001023c12df13073dd08550a9ace13037d34e84bb463892`.
- New source snapshot `29acfcd9a5f830c4427f96a98c313d8e6706a12a62d9c853916948ffdeb3e707`,
  based on `9a5e7a6` with the recorded uncommitted stage changes.
- New manifest `97e0d19f35f82b6cf128d1ed07b3178b5ad523386000735cbc7b499144f0e1fa`.
- Pre-validation train checkpoint `6e76f196f56cfa83195ab2f3e32d4f47b59825aefad534509fcdc1a0fce397d9`.

All original source hashes, query groups/selections, official qrels, candidate membership
and source nDCG/Recall are checked. A second complete CPU replay has **byte-identical
files including the manifest**. Training selection is unchanged after validation.
Synthetic tests cover endpoints, raw-score scale invariance, ties, corrupt pools, source
pins, official-qrel replay even after rehashing, test refusal, held-out-label
counterfactuals, group separation, zero-weight bypass and a single validation weight.
**188 CPU tests pass**; Ruff lint/format and diff checks pass. The existing third-party
SQLAlchemy deprecation warning remains. No ordinary test downloads assets or invokes neural inference/network.

Local MLflow **FINISHED**: `93c50c8dd83842caba35022102083b74`. Seven explicit metadata
artifacts plus aggregate/PNG/SVG are logged; raw inspection text/qrels/candidate scores
and model parameters are excluded. Logging twice returned the same ID. The revised
figure with clearer tick labels is `artifacts/m4/analysis/rankblend_train500_verified_v2.*`;
both versions retain the same measurements.

Commands from repository root (choose new outputs for reproduction):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --extra tracking --extra dense --extra routing pytest -q
uv run --frozen --extra tracking --extra dense --extra routing ruff check .
uv run --frozen --extra tracking --extra dense --extra routing ruff format --check .
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --extra tracking --extra dense --extra routing \
  python -m temporalproof.evaluation.rank_blend --output artifacts/m4/rankblend_train500_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --extra tracking --extra dense --extra routing \
  python -m temporalproof.evaluation.rank_blend --output artifacts/m4/rankblend_train500_replay_v1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run --frozen --extra tracking --extra dense --extra routing \
  python experiments/analyze_rank_blend.py \
  --run artifacts/m4/rankblend_train500_v1 --replay artifacts/m4/rankblend_train500_replay_v1 \
  --measured-on 2026-11-27 --output experiments/results/m4_rankblend_train500_v1.json
uv run --frozen --extra tracking --extra dense --extra routing \
  python experiments/plot_rank_blend.py --results experiments/results/m4_rankblend_train500_v1.json \
  --output artifacts/m4/analysis/rankblend_train500_verified_v2.png
uv run --frozen --extra tracking --extra dense --extra routing \
  python experiments/log_rank_blend.py --run artifacts/m4/rankblend_train500_v1 \
  --results experiments/results/m4_rankblend_train500_v1.json \
  --figure artifacts/m4/analysis/rankblend_train500_verified_v2.png
```

Keep tracking/dense/routing extras together; no dependencies or model versions changed.
Dataset redistribution terms and independent model decontamination remain unresolved.
Dates/text diversity remain proxies, not validated event time or version chains.

**Decision:** retain hybrid as the reference. Blending reduces raw-deep regressions but
has no established advantage over hybrid. Next small increment: generate **train-only**
changed-utility targets and evaluate a router for the blended deep path, keeping weight
selection within each training fold. Reuse cached scores; final test remains untouched.
Candidate coverage is a separate bottleneck requiring a later controlled retrieval study.


Subsequent planning priority: reproduce the official TimelyRAG baseline on a matched
protocol before further router fits. Published paper values are not directly comparable
to these local scores; preserve separate reference-prefix/full-qrel metric columns and
automatic-alpha versus gold-dependent oracle results. No official reproduction has yet
been executed. The changed-utility router remains a later train-only increment.
