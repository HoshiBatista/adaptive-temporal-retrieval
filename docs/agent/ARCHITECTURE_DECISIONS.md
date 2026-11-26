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
