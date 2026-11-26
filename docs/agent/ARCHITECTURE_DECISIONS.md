# Architecture constraints and decision log

This document records **planned architectural boundaries**, not implementations. Agents must inspect the repository before asserting any component exists.

## Core design principles

1. **Retrieval quality is independent of generation quality.** Evaluate ranked evidence without requiring an LLM API.
2. **Time is a first-class query constraint**, not just a recency boost.
3. **Provenance survives every transformation:** corpus document ID → version ID → passage/chunk ID → retrieved evidence → saved answer reference.
4. **Fast path and deep path share candidates and metadata.** Avoid comparing different candidate pools unless that is the explicitly studied variable.
5. **Optimize only after measuring.** Build a CPU-friendly baseline before adding a Rust gateway, GPU reranking, async pipelines or a web dashboard.
6. **Degrade gracefully.** If timestamps are missing, return unknown applicability; if a reranker is offline, use the fast path; if answer generation is disabled, still return ranked evidence.

## Logical components (planned)

```text
Versioned source documents
         |
  Data loader / metadata normalizer
         |
 [Document + version + passage IDs]
         |
 +-------+---------------+
 |                       |
BM25 index             Vector index
 |                       |
 +----- hybrid retrieval-+
                |
     fast semantic-temporal features
                |
       lightweight route policy
             /        \
        fast path    deep reranker
             \        /
       ordered evidence + citations
                |
  optional grounded response generator
                |
 answer → cited passage references
                |
 source revision → ChangeImpact review flag
```

## Proposed package modules

| Module | Responsibilities | Do not put here |
|---|---|---|
| `src/temporalproof/data/` | benchmark loaders, manifests, validation | retriever training business logic |
| `src/temporalproof/temporal/` | temporal filters, validity intervals, version semantics | HTTP requests and DB side effects in pure functions |
| `src/temporalproof/retrieval/` | sparse, dense, hybrid candidate retrieval | answer generation |
| `src/temporalproof/reranking/` | fixed fast/deep rankers, features | query gold judgments at inference |
| `src/temporalproof/routing/` | route prediction, routing budgets, routing features | access to gold/qrels or post-deep inference |
| `src/temporalproof/evaluation/` | exact benchmark protocols and metrics | live user-data writes |
| `src/temporalproof/provenance/` | evidence refs and persistent mappings | irreversible rewriting of original revisions |
| `src/temporalproof/changeimpact/` | review-required notification on changed cited evidence | assertions that the original answer is false |
| `src/temporalproof/api/` | typed service routes, validation | ML experiment training loops |

Use the README's proposed structure as a **guide**; create folders only when concrete code needs them.

## Interfaces to decide before implementation

Data record (conceptual; adapt to verified benchmark):

```text
DocumentVersion {document_id, version_id, published_at?, valid_from?, valid_until?, passages[]}
Passage {passage_id, version_id, text, source_uri?, section_id?}
SearchQuery {query_id?, text, as_of?, time_range?, requested_time_mode?}
SearchHit {passage_id, version_id, fast_score, deep_score?, temporal_status, evidence_uri}
RouterDecision {path: FAST|DEEP, reason_codes[], features_version, policy_version}
SavedAnswer {answer_id, question, answer_text?, evidence_refs[], time_intent, created_at}
ChangeImpactEvent {answer_id, changed_evidence_refs[], status: REVIEW_REQUIRED|REVIEWED}
```

`valid_from`/`published_at` are optional only when unknown. Never synthesize fake values for missing dates without a policy and an audit trail. Use fixed rules for interval boundaries.

## Data and storage boundaries

- **Raw benchmarks:** immutable, gitignored, with source manifest/checksum.
- **Derived data:** versioned preprocess outputs, feature caches and candidate lists; deterministic rerun supported.
- **Metadata/product state:** PostgreSQL when persistence is necessary; use minimal test fixtures before introducing a database.
- **Vector search:** FAISS *or* pgvector for MVP, not both without evidence.
- **Lexical search:** library-based BM25 first; consider Tantivy/Rust only after a measurement.
- **Monitoring:** structured logs and request IDs first; Prometheus/Grafana for the integrated service.

## Decisions (proposed; review when new evidence arrives)

| ID | Status | Decision | Rationale | Revisit trigger |
|---|---|---|---|---|
| ADR-001 | PROVISIONAL | Python-first, CPU-testable research package | Iteration speed; baseline clarity | Measured CPU bottleneck |
| ADR-002 | PROVISIONAL | TimelyQABench primary; TEMPO external validation | Existing relevance judgments | License/schema incompatibility |
| ADR-003 | PROVISIONAL | Simple learned router before neural policy | Lower overfitting risk and cost | Measured failure at fixed routing budget |
| ADR-004 | PROVISIONAL | ChangeImpact is dependency-based review notification | Avoids need for own staleness labels | A separate validated staleness benchmark becomes available |
| ADR-005 | PROVISIONAL | Optional generator, not part of required research metrics | Prevents LLM confounders | A reliable answer QA benchmark is verified |
| ADR-006 | PROVISIONAL | Rust gateway deferred | Avoids premature complexity | Stable Python benchmark reveals a service bottleneck |

### ADR-007: Preserve raw benchmark timestamps and audit uncertainty
Date: 2026-11-26
Status: ACCEPTED for the M0 adapter

Company Policies contains 47 insertion strings outside the exact reference-format policy,
no explicit event-time fields, and positive evidence inserted after query issuance. The
typed adapter retains raw strings; parsing returns unknown for missing or malformed
values, and the audit reports malformed counts separately. It does not repair values,
derive event times, infer validity intervals or filter judgments. Unknown schema fields
fail explicitly, so a different revision/domain needs an observed-schema review. Input
errors omit source values and unknown field names. Revisit on a documented derived
normalization or an expanded benchmark adapter; raw inputs stay immutable.

### ADR-008: Separate qrel-based metrics from reference reproduction
Date: 2026-11-26
Status: ACCEPTED for initial evaluation

Binary nDCG uses ideal DCG from the full positive set; Recall uses the full positive-set
denominator. The reference normalizes nDCG using the retrieved prefix, and its best-alpha
selection uses qrels per query. Preserve reference files unchanged and label any future
reproduction separately; best-alpha is an offline diagnostic. Tests cover the missing-gold
normalization counterexample. See [`benchmark_protocol.md`](../benchmark_protocol.md).

### ADR-009: Freeze connected query groups before retrieval evaluation
Date: 2026-11-26
Status: ACCEPTED for Company Policies v1

The split joins normalized exact query text, number-masked templates, near templates
(character-trigram set Jaccard >=0.9) and any normalized exact passage shared across query
records. All associated passages participate without consulting qrels. Groups are assigned
by a seeded SHA-256 bucket rule (70/15/15 target proportions). The ID-only manifest is
recomputed and checked before evaluation; different outputs need a new version. This
controls observed query/exact-passage reuse, without claiming exhaustive semantic-family
separation. No corpus deduplication or label rewriting occurs. M1 evaluates a fixed
100-query validation selection and refuses test configs; test is reserved for later frozen
comparisons. See [`m1_bm25_report.md`](../m1_bm25_report.md).

### ADR-010: Independent CPU BM25 and file-based run manifests
Date: 2026-11-26
Status: ACCEPTED for M1

Use rank-bm25 0.2.2 with explicit default Okapi parameters and raw `str.split()` to match
the reference's sparse preprocessing, with deterministic score-descending/ID-ascending
ties. Search the entire domain, not per-query attached candidates. Save run/qrel files,
per-query metrics, reference-formula diagnostic, fixed-rule failure examples, configuration,
source/dependency hashes and timing scope. Existing outputs cannot be overwritten.
MLflow is deferred until multi-model comparisons in M2; manifests remain the portable
record even when tracking is added. Raw text/qrels remain ignored while exact data terms
are unresolved. Revisit tokenization/models on train/validation only.

### ADR-011: Optional local SQLite MLflow with immutable file evidence
Date: 2026-11-26
Status: ACCEPTED at the user's explicit request after M1

The `tracking` extra pins MLflow 3.18.0. Explicit low-level clients write to a project-local
SQLite database and file artifact store under ignored `artifacts/mlflow/`, independent
of the UI process or remote tracking environment variables. A wrapper serves that store
on loopback only with a single worker and disabled SDK/UI telemetry. MLflow owns its
schema; application persistence remains unimplemented.

BM25 tracking is opt-in and runs after evaluation/output, outside retrieval timing.
File manifests remain the portable evidence. Import validates all original output hashes,
copies an explicit artifact allowlist, preserves original Git/source/package provenance,
and records historical measurement date separately from MLflow import clock timestamps.
Finished imports are idempotent by manifest fingerprint; different historical dates are
rejected. Tracking failure leaves completed research files intact and returns an error
for explicit tracking requests. Local raw failure text is not duplicated into MLflow.
This implements the tracking previously deferred in ADR-010; no remote account or model
execution is required. See [`mlflow.md`](../mlflow.md) for actual commands and scope.

## Decision entry template

```text
### ADR-XXX: <short title>
Date: YYYY-MM-DD
Status: PROPOSED | ACCEPTED | SUPERSEDED
Context:
Options considered:
Decision:
Evidence / benchmark / issue:
Consequences and reversal plan:
```

Do not mark provisional decisions accepted solely because a code agent likes a particular technology.
