# M4.4 — Train-only query-date roles and timestamp coverage

Measured 2026-11-28. This stage audits temporal inputs and freezes a prospective
QET-only ablation. It does **not** measure retrieval quality or temporal-role accuracy,
fit a router, or run a new validation/test experiment. Hybrid remains the reference.

## Scope and reproducibility

All **2,058 train queries / 1,994 source groups** from the existing split are parsed.
Forty groups are selected before the audit by SHA-256 of seed1729 and sorted group IDs;
the minimum query ID represents each selected group. All forty original train questions
and their derived date annotations were inspected. Neither date-role gold nor new
relevance labels are created. Sample IDs are in the committed aggregate; original text,
mention/date annotations and raw inspection output stay in ignored local artifacts.

The archive and frozen split are verified as before. Dataset-wide grouping verification
loads records, but the date-role analysis invokes the parser only on train queries.
Relevance labels, answers and expensive-reranker outcomes are not inputs to role selection.
Normal tests use independently authored synthetic examples, without downloads/models.

The cached hybrid train500 top100 pools are reused unchanged. No new neural inference,
GPU measurement, model download or training takes place: text rules and timestamp counts
run on CPU. The cached retrieval experiment already used the requested GPU for neural work.
There is no reason to re-encode these candidates for this audit.

## Observations and measured coverage

| Item | Count |
| --- | ---: |
| Train queries with parsed date mentions | 2,047 / 2,058 |
| Queries with multiple date mentions | 1,674 |
| Parsed mentions | 4,485 |
| Explicit calendar mentions / missing-year mentions | 4,471 / 14 |
| Mention precision: year / month / day / second | 1,581 / 59 / 2,844 / 1 |
| Native reference selects a matched mention with an employment cue | 1,478 |
| Native reference selects a matched mention with **only** an employment cue | 1,174 |
| New policy selects a precise explicit anchor | 267 (12.97%) |
| Preserve original fast ranking | 1,791 |
| Selected points agreeing / differing from native QET | 51 / 216 |
| Selected anchors later than query insertion | 97 |

These are **heuristic cue/coverage counts**, not error rates or evidence that any of
these choices improves nDCG. Native extraction succeeds for2,047 queries:2,023 unique
explicit matches,23 repeated matching mentions,1 without a matching explicit mention.
A date of employment can legitimately constrain the requested rule; ignoring its role
would also be a mistake. This audit only motivates a controlled alternative to the
source's finest-date/first-position heuristic.

| Decision reason | Queries |
| --- | ---: |
| Unique precise as-of target | 106 |
| Unique precise policy target | 161 |
| No target-role cue | 1,558 |
| Comparison request | 105 |
| Conflicting role cues | 77 |
| Coarse target precision | 48 |
| Multiple target values | 2 |
| Non-point target relation | 1 |

Manual inspection of the frozen forty-group sample found common employment/cohort dates
followed by leave/return dates, with no explicit evaluation point. It also found explicit
as-of dates, policy-change anchors, year-only references, ranges and comparisons. The
parser selects4 sample targets (1 as-of/3 policy);28 have no target cue,3 conflicting cues,
3 coarse targets and2 comparisons. This is a descriptive inspection, not a labeled
accuracy evaluation. The rules can abstain unnecessarily: policy and leave wording in
the same local context produces conflict in sampled queries819 and2106. Keep that
limitation visible rather than treating all abstentions as correct.

## Timestamp uncertainty

| Coverage scope | Passages/candidates | Unknown insertion | Explicit event metadata |
| --- | ---: | ---: | ---: |
| Original passages associated with train queries | 20,580 | 25 | 0 |
| Frozen train500 hybrid top100 candidate occurrences | 50,000 | 25 | 0 |

The pool has **12,672** candidate occurrences inserted at/after query insertion. They
remain in the pool; no qrels are removed. The source reference makes such insertion
comparisons unknown/neutral, not invalid. Candidate occurrences and distinct original
passages are different denominators; their equal unknown counts are incidental.
Of the canonical500 queries, **63** receive a precise target from the new policy.

Insertion time does not establish event/applicability time. Missing document event
metadata is still missing. The97 later query anchors are not silently clamped to query
insertion or called historical truth. This stage establishes neither valid-time labels
nor historically available evidence.

## Frozen policy and the next experiment

`retrieval/query_time.py` is a text-only parser with NFKC-normalized span offsets and
bounded local cue windows (16 characters before/32 after, clipped at the next date and
punctuation). It records explicit resolution and calendar bounds, missing-year/invalid
status, employment/event/policy/as-of cues and conflicts. Invalid longest spans cannot
silently become valid shorter dates. Explicit timezone suffixes are unsupported/unknown.

Prefer a unique explicit as-of target, otherwise a unique explicit policy anchor. Use
only day-or-finer points; represent a full date at naive midnight without a timezone or
legal-validity claim. Year/month bounds describe **text resolution uncertainty**, not
applicability intervals; never turn them into a Jan1/first-day QET. Missing years acquire
no insertion-year backfill. Conflicting, comparative, ambiguous or non-point targets
abstain. Policy start/after anchors are an explicit heuristic, not proven query-event time.

Abstention means **keep the original fast ranking**. Passing `None` into a native adapter
that falls back to query insertion would implement a different policy. No current/default
date is substituted. The frozen contract and module hash are saved before any future
retrieval evaluation of this candidate.

The next increment is **train500 cached QET-only ablation**:

1. Reuse the exact frozen hybrid top100 IDs/scores and original published qrels.
2. Compare unchanged fast, native auto-alpha, and auto-alpha with only selected QET
   overridden; new-policy abstention returns fast. Keep native ties, query text/granularity
   helpers, document insertion fallback, alpha and gates unchanged.
3. Verify native parity and unchanged candidate coverage. Report full-qrel nDCG@10,
   paired group intervals, regressions and separate reference-prefix diagnostics.
4. Evaluate the train gate before any new validation invocation; do not search a rule,
   alpha or router grid using validation. Test stays closed. A positive train result
   alone cannot establish held-out superiority or research novelty.

This follow-up has **not run**. This stage changes neither the existing source auto
baseline nor any stored benchmark/reranker output.

## Evidence and checks

Config: `configs/experiments/m4_query_time_roles_train.json`.
Aggregate: `experiments/results/m4_query_time_roles_train_v2.json`.
Local evidence: `artifacts/m4/query_time_roles_train_v2` and
`artifacts/m4/query_time_roles_train_replay_v2`.

- Source snapshot: `5077e97ff9237bab3f5629cee5f73e54db5b86ef64bc394525a62063e720ec6d`.
- Config SHA: `a69c4b950d95017424b148bc49e4282cf7a76808a78030540c4c624377b04f8c`.
- Query-time module SHA: `6fccdecf951acdb64cff94c6854cd05d4e2fb5472e51428e407ffc99f6837c4e`.
- Frozen contract SHA: `f9d4cf0912c5f8300276cbec7207322ef0d6f6af70ee858ef9ad7799f290a162`.
- Manifest SHA: `823be6ec0589e7b1797e5bb16da457bc7e147f58f237987bddcd132ccf0c3598`.
- Source train500 manifest: `b3f0738a8e635f09930ae7aff8dfd368502ba73695c56fcedffe2a3f1d213049`.
- Native-source manifest: `dbd0dce4db92abc276d65dafe16166ccafb691fec33fb9f5a70e68461d04bb55`.

Final v2 fixes recognition of timezone suffixes attached directly to Korean text; six
synthetic timezone cases cover spaced/attached notation. Both v1 audits and the original
metadata aggregate remain preserved. Their per-query annotations and summaries are
byte-identical to v2; only implementation/policy provenance changes. The final v2 evidence
and two final logging invocations are the ones reported here. The earlier FINISHED local
run `ad8503d34b964cf5aeee6d20778fce85` remains separate.

All seven manifested files and the manifest match byte-for-byte across the two v2 audits.
**242 tests pass** with reviewed-source integration enabled, including36 new CPU cases;
ordinary suite **238 pass /4 optional skips**. Ruff lint/format and `git diff --check`
pass. Existing SQLAlchemy deprecation warning only. Tests cover role isolation, context
boundaries, coarse precision, calendars, missing years, unsupported timezones, ambiguous
and comparative requests, inference schema consistency, label/answer counterfactuals,
train-only sampling/analysis and unknown/future timestamp counting without filtering.

Local MLflow run **`3399daba47cb48198c4f02c7059f962b`** is FINISHED. Only config/plan,
summary, frozen policy, manifest and aggregate are uploaded, through an explicit allowlist;
raw query annotations and forty-question inspection are excluded. Logging is idempotent.
No nDCG, temporal accuracy, latency or VRAM metric is attributed to this audit.

Commands executed from the repository root, with existing local archive/reference assets:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
uv run --frozen --extra tracking --extra dense --extra routing python -u -m temporalproof.evaluation.query_time_audit --output artifacts/m4/query_time_roles_train_v2
uv run --frozen --extra tracking --extra dense --extra routing python -u -m temporalproof.evaluation.query_time_audit --output artifacts/m4/query_time_roles_train_replay_v2
uv run --frozen --extra tracking --extra dense --extra routing python experiments/analyze_query_time_audit.py --run artifacts/m4/query_time_roles_train_v2 --replay artifacts/m4/query_time_roles_train_replay_v2 --measured-on 2026-11-28 --output experiments/results/m4_query_time_roles_train_v2.json
uv run --frozen --extra tracking --extra dense --extra routing python experiments/log_query_time_audit.py --run artifacts/m4/query_time_roles_train_v2 --results experiments/results/m4_query_time_roles_train_v2.json
TEMPORALPROOF_REFERENCE_TESTS=1 uv run --frozen --extra tracking --extra dense --extra routing pytest -q
uv run --frozen --extra tracking --extra dense --extra routing pytest -q
uv run --frozen --extra tracking --extra dense --extra routing ruff check .
uv run --frozen --extra tracking --extra dense --extra routing ruff format --check .
git diff --check
```

Choose new audit/output directories for reruns. Raw/reference redistribution terms remain
unresolved; no source text or upstream implementation is committed by this stage. No
commit/push has been performed. BGE-M3 whole-paper reproduction, TEMPO and final test
remain unrun; API/UI/generation/ChangeImpact remain outside this increment.
