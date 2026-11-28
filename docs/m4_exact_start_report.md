# M4.7 — Exact policy-start matches on train500

Promoting a passage with the same policy-start day as the query produced nDCG@10
**0.197092** against **0.197016** for frozen hybrid. The paired gain was **+0.000076**
with a 95% group-bootstrap interval of **[−0.001052, +0.001424]**. The predeclared
train gate failed. Hybrid remains the reference. Evaluation covered training only.

## Protocol

The config fixes 500 queries in 493 source groups, original hybrid top100 scores,
published qrels, and the M4.4 and M4.6 parser versions. The query must have a unique
policy target with day precision. The highest-ranked match moves up a single position
if its original rank is 2–20. Matches at rank 1 or below rank 20 leave the order
untouched. This bounds the intervention to an adjacent swap and preserves
candidate scores and membership.

The explicit-date method uses M4.6's train representatives. Documents outside
that scope contribute no match, including excluded duplicate IDs, though they can
still appear in the rankings. The insertion-day control uses the same query
eligibility, document scope and swap. It has greater date availability. Its result
measures the two deployable signals under the same mechanics, with unequal coverage.
Exact matches are text cues. The query rule selects a date from the question but
does not resolve general historical applicability.

The first run counted coverage with `--coverage-only`. Quality scoring followed
with the same config. It checks the cached qrels against original records and
replays the original baseline metrics. Paired group resampling uses 2,000 draws,
seed 1729. The primary gate requires a positive lower bound for the interval
versus hybrid. The other comparisons are descriptive.

## Coverage and outcome

There were **37** eligible queries. The 50,000 candidate occurrences contained
9,963 unique documents. The document audit covered 6,234 of those unique IDs
(30,695 occurrences). Within those IDs, 1,649 had an explicit start day and
6,232 had a known insertion day.

| Date source | Queries with top100 match | Top20 match | Match already first | Swaps |
| --- | ---: | ---: | ---: | ---: |
| Explicit rule start | 33 | 28 | 3 | 25 |
| Insertion day | 37 | 34 | 11 | 23 |

| Method | Full-qrel nDCG@10 | Recall@10 | Recall@100 |
| --- | ---: | ---: | ---: |
| Hybrid | 0.197016 | 0.251133 | 0.426433 |
| Explicit start | 0.197092 | 0.251133 | 0.426433 |
| Insertion day | 0.197648 | 0.251133 | 0.426433 |

The explicit rule improves 3 queries and regresses 3. Its mean difference from
the insertion-day control is **−0.000556**, interval **[−0.002475, +0.001026]**.
The insertion-day control differs from hybrid by **+0.000632**, interval
**[−0.000011, +0.001875]**. All three intervals contain zero. Recall@10, @20 and
@100 are equal in aggregate. Recall@100 also agrees per query.

For query 2925, the swap lifts a judged positive from rank 2 to 1 and gains
0.226294 nDCG. For query 636, the same swap displaces a judged positive with an
unjudged document and loses 0.173197. For query 1404, a match at rank 3 displaces
a positive at rank 2 and loses 0.080279. These examples show how the same exact
date cue can help or hurt ranking with the published judgments. Unjudged
documents may still be relevant.

The result covers this specific local match rule on development queries. The
small number of swaps limits resolution, and the frozen pool prevents retrieving
positives that are absent from the candidates. A separate candidate-generation
experiment can address that bottleneck.

## Reproduction

The experiment uses existing GPU hybrid rankings and runs the date logic on CPU.
The complete result replayed byte-for-byte. All 284 tests with upstream integration
enabled passed. The local MLflow run `ccf627f04f834f808cbecd1734e72cc8` holds the
aggregate and scalar metrics. [Aggregate](../experiments/results/m4_exact_start_train_v1.json)

```bash
uv run --frozen --extra tracking --extra dense --extra routing python -m temporalproof.evaluation.exact_start --coverage-only --output artifacts/m4/exact_start_coverage_repro.json
uv run --frozen --extra tracking --extra dense --extra routing python -m temporalproof.evaluation.exact_start --output artifacts/m4/exact_start_train_repro.json
cmp artifacts/m4/exact_start_train_repro.json experiments/results/m4_exact_start_train_v1.json
TEMPORALPROOF_REFERENCE_TESTS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --frozen --extra tracking --extra dense --extra routing pytest -q
```
