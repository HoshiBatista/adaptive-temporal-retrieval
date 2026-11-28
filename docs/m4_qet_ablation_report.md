# M4.5 — Frozen train500 QET-only ablation

Measured 2026-11-28. The frozen explicit-role query-time policy **does not pass the
train promotion gate**. Full-qrel nDCG@10 is **0.1944454822**, versus unchanged hybrid
**0.1970158265**; paired source-group delta **−0.0025703443**,95% interval
**[−0.0075911697,0.0010854833]**. It improves on the source auto heuristic, but does not
establish an advantage over the fast reference. No new validation/test experiment runs;
no default ranking or learned router is adopted.

## Protocol fixed before outcomes

The preceding M4.4 parser and frozen policy contract are unchanged, including16/32-character
cue windows and exact fast bypass. Use the existing seed1729 train500/493 source groups,
original hybrid top100 IDs/scores, source corpus order restricted to each frozen pool,
original published qrels, source coefficients/gates, query text/granularity helpers and
document insertion fallback. No grid, new target generation or fitting. Day points remain
heuristic anchors; missing document event metadata remains missing, not applicability gold.

`configs/experiments/m4_qet_ablation_train.json` and the copied plan are saved before
scoring. Primary: role-QET versus unchanged fast. The predeclared train gate requires a
positive mean **and** a strictly positive paired-group interval lower bound. It fails.
The policy was engineered on train text, including queries evaluated here: these are
conditional development diagnostics, not held-out generalization or formal novelty evidence.

Before outcomes, two secondary controls were also recorded in the plan:

- Native-fast sorting on the restricted original-corpus axis exposes NumPy tie effects.
- Native auto on exactly the same selected queries, fast otherwise, separates changing
  QET from bypassing temporal reranking. This is a diagnostic, not an outcome-selected policy.

This experiment reranks a **frozen100 pool**, not the full corpus. Outside-pool scores are
not reconstructed. Native NumPy argsort applies to this100-vector; equal-score ordering
need not equal argsort over a full30k vector. The unchanged-fast control retains its old
doc-ID tie order. Do not directly compare these new train means with M4.3 validation means
or published BGE-M3 whole-benchmark results. No corpus, split, source data or labels change.

## Results

| Method | nDCG@10 | Recall@10 | Recall@20 | Recall@100 | Reference-prefix nDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Unchanged hybrid fast | 0.197016 | 0.251133 | 0.306767 | 0.426433 | 0.264425 |
| Native-fast tie control | 0.196208 | 0.251133 | 0.309433 | 0.426433 | 0.263110 |
| Native auto, all500 | 0.164114 | 0.223933 | 0.307500 | 0.426433 | 0.227515 |
| Native auto, same bypass | 0.192918 | 0.248533 | 0.306433 | 0.426433 | 0.256943 |
| Frozen role-QET + fast bypass | 0.194445 | 0.249300 | 0.306000 | 0.426433 | 0.263134 |

Primary nDCG normalizes using all positive qrels. Reference-prefix nDCG is a separately
named compatibility diagnostic; all2,500 method/query cells agree with the unchanged
reviewed official metric. No best-alpha/oracle outcome is calculated on benchmark data.

| Candidate − control | Mean nDCG delta | Paired group95% interval | Gains / regressions / unchanged |
| --- | ---: | --- | --- |
| Role-QET − fast (primary) | −0.002570 | [−0.007591,0.001085] | 8 / 9 / 483 |
| Role-QET − native auto | +0.030332 | [0.017206,0.044203] | 99 / 62 / 339 |
| Role-QET − native auto with same bypass | +0.001528 | [−0.003739,0.006624] | 8 / 11 / 481 |
| Native auto with same bypass − fast | −0.004098 | [−0.008637,−0.000263] | 8 / 10 / 482 |
| Native auto − native fast | −0.032095 | [−0.045550,−0.018936] | 61 / 101 / 338 |
| Native fast − unchanged fast | −0.000808 | [−0.001849,−0.000017] | 4 / 10 / 486 |

Intervals use2,000 source-group resamples, seed1729, query-weighted paired means; fixed
predictions, no policy refits or multiple-comparison correction. Secondary intervals
are descriptive. The all-query gain versus native auto consists of **+0.028804** from
using the same bypass with original QET and **+0.001528** from the QET replacement.
It must not all be attributed to better date identification.

The policy calls the cheap temporal kernel on **63/500** queries (26 as-of/37 policy);
**437** preserve fast order **and scores exactly**. Of63,55 QET points change,8 agree
with native extraction. On these63, QET replacement versus the same-bypass control gives
mean delta **+0.012125**, interval **[−0.027144,0.051063]**,8/11/44; no established date-only gain.
These calls are CPU temporal reranking, not expensive neural-reranker calls.

All five methods preserve every candidate pool and per-query Recall100. **211/500**
queries have no judged positive in the pool; **25/63** active queries have none. A
permutation cannot retrieve an absent positive. Unjudged IDs scored as zero are not
proven irrelevant, so these are qrel-based coverage limits, not complete semantic gold.

## Failure inspection without rule tuning

Inspected the top query, fast/role top passages and positive-rank movement for the three
largest primary regressions and three largest gains, with deterministic query-ID ties.
Full text stays in ignored `inspection_failures6.json`; only query IDs/scalars appear in
the committed aggregate. No new gold judgments, error-class labels or rule changes result.

- **428:** nDCG1→0.135652; three positives move from ranks1/2/3 to12/10/11. The query
  includes employment, leave, policy-start and before/after wording; the fixed regex still
  selects a policy anchor. Its best source passage explicitly states the requested rule,
  but has insertion after query insertion. This illustrates both limited comparison cues
  and the risk of treating insertion fallback as applicability; no gold time is inferred.
- **2340:**1→0.5; positive rank1→3. Both an as-of observation and a later policy start
  appear; as-of priority does not establish the rule's actual evaluation point. An older
  rule version is promoted. The policy remains unchanged; this is not a relabeled example.
- **2262:**0.636439→0.289308; a rank1 positive moves to9. A related but different benefit
  passage rises. Correctly recognizing a policy-looking date alone does not establish
  passage applicability or semantic relevance.
- **2194:**0→0.264068; a positive moves16→4. Other relevant positives still remain deep
  in the pool; neither top1 nor the whole answer is certified correct.
- **1404:**0.386853→0.613147; positive rank2→1, **QET unchanged**. An unjudged, near-identical
  passage had rank1. The gain is temporal permutation under existing QET and incomplete
  judgments, not evidence of improved date parsing.
- **1997:**0.156426→0.234639; positive rank7→3 while another moves20→43. This is a mixed
  per-document change even though aggregate nDCG improves.

The three largest regressions all involve a leading positive inserted later than query
insertion. This selected-small-sample observation is not a causal estimate or historical
validity gold. No qrels/candidates are filtered and no dates are repaired from insertion.

## Incremental CPU timing

| Callback | First mean / p50 / p95 ms | Replay mean / p50 / p95 ms |
| --- | --- | --- |
| Native auto | 0.579 / 0.576 / 0.661 | 0.577 / 0.577 / 0.644 |
| Role-QET + bypass | 0.168 / 0.094 / 0.686 | 0.168 / 0.094 / 0.694 |

Three warmups, serial concurrency1; OMP/MKL/OpenBLAS threads1; CPython3.13.11,
NumPy2.5.5, Linux/WSL2. Measurement includes the callback's pool checks/date parsing and,
when selected, temporal kernel. Excludes archive/pool loading, array assembly, source
parity, metrics, file I/O and original hybrid retrieval. This is **not live service p95**,
original full-corpus helper timing, a GPU speedup or end-to-end routed throughput.

The experiment reuses existing GPU hybrid results. No new neural inference, model load,
GPU allocation/VRAM measurement or dataset/model download is required. No new build/index
cost exists in this cached stage. Original retrieval costs are not remeasured or silently
added to claim a live latency result.

## Reproduction and validation

- Parent: `2b92b47`, dirty experiment implementation recorded at measurement.
- Source snapshot: `47a453949af1b9e9038fa035a5dd42036e47f0ff80ee6a5c9f9a981a0a03e1af`.
- Config SHA: `8b63a18da3c37806eb776bac81d420ff68a2ece4fa99d7b5e153a918d06a3e65`.
- Run manifest: `8b1b4c6c613249b87b1901b136d538ac0f802019bbe4204ecc698c87804de056`.
- Frozen preceding policy: `f9d4cf0912c5f8300276cbec7207322ef0d6f6af70ee858ef9ad7799f290a162`.
- Input audit manifest: `823be6ec0589e7b1797e5bb16da457bc7e147f58f237987bddcd132ccf0c3598`.
- Source train500 manifest: `b3f0738a8e635f09930ae7aff8dfd368502ba73695c56fcedffe2a3f1d213049`.

Two independent cached runs have **20 byte-identical non-timing evidence files**:
config/plan/frozen contract, summary/diagnostics and all five complete top100 rankings,
qrels and per-query metric files. Three timing files are intentionally distinct and
retained; manifests differ accordingly. The exporter verifies both evidence sets, current
implementation and aggregate means; no raw query/date text or qrels are exported to Git.

The original automatic function API and native diagnostics remain backward-compatible;
only its shared kernel accepts an explicit naive QET override through a new qrel-free
entry point. No other extraction, granularity, fallback or numeric computation changes.
Independent upstream execution matches **all500 native top10s and all63 overridden top10s**.
It runs in a private namespace and stops at the first metric boundary with an access-
forbidden gold sentinel: no benchmark labels are supplied to parity, no metric is
substituted for the benchmark, and the oracle branch cannot execute.

**262 CPU tests pass** with reviewed-source integration, including20 new cases; normal
suite **257 pass/5 optional skips**. Tests cover unchanged native API/parity, identical-QET
replay, explicit overrides, original query/time-array preservation, no fallback on bypass,
strict pool alignment, forbidden gold inputs/access, private namespaces, held-out/config
rejection and the train gate. Ruff lint/format and diff checks pass. Existing SQLAlchemy
third-party deprecation warning only. No model/network work in ordinary tests.

Local MLflow **`f76c35e84ca5423384869823fc830c57`** is FINISHED; repeated import returns the
same ID. Explicit artifact allowlist: config, plan, frozen policy, summary, timing,
manifest and aggregate. Excludes diagnostic query text, ranks/qrels and manual inspection.

Commands executed from the repository root using existing ignored local assets:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uv run --frozen --extra tracking --extra dense --extra routing python -u -m temporalproof.evaluation.qet_ablation --output artifacts/m4/qet_ablation_train500_v1
uv run --frozen --extra tracking --extra dense --extra routing python -u -m temporalproof.evaluation.qet_ablation --output artifacts/m4/qet_ablation_train500_replay_v1
uv run --frozen --extra tracking --extra dense --extra routing python experiments/analyze_qet_ablation.py --run artifacts/m4/qet_ablation_train500_v1 --replay artifacts/m4/qet_ablation_train500_replay_v1 --measured-on 2026-11-28 --output experiments/results/m4_qet_ablation_train500_v1.json
uv run --frozen --extra tracking --extra dense --extra routing python experiments/log_qet_ablation.py --run artifacts/m4/qet_ablation_train500_v1 --results experiments/results/m4_qet_ablation_train500_v1.json
TEMPORALPROOF_REFERENCE_TESTS=1 uv run --frozen --extra tracking --extra dense --extra routing pytest -q
uv run --frozen --extra tracking --extra dense --extra routing pytest -q
uv run --frozen --extra tracking --extra dense --extra routing ruff check .
uv run --frozen --extra tracking --extra dense --extra routing ruff format --check .
git diff --check
```

Do not overwrite completed directories. Dataset/reference source remains unmodified and
ignored, with unresolved redistribution terms. Metadata aggregates are in
`experiments/results/m4_qet_ablation_train500_v1.json`; inference/evaluation modules are
`retrieval/qet_policy.py` and `evaluation/qet_ablation.py`.

## Next increment

Keep hybrid and all negative baselines; do not evaluate this failed candidate on
validation or tune another query-date rule. Next audit whether explicit **document rule
start/effective dates** can be conservatively represented from train passage text,
separately from insertion dates. Use a deterministic unlabeled train sample, preserve
uncertainty/multiple roles and freeze a policy before any new ranking comparison. This
is a proposed data audit, not a claim that document-time extraction will improve quality.
No change to the primary retrieval/routing research problem, test, API/UI or generation.
