# Ready-to-use coding-agent prompts

Copy **one task at a time** into Codex CLI or Claude Code, starting from the repository root. The agent instructions are in `AGENTS.md` and `CLAUDE.md`; these prompts state the outcome and acceptance criteria rather than scripting every implementation detail.

## Prompt 0 — First session: review scope, inspect, and plan

> You are working on TemporalProof. Read README.md, AGENTS.md, docs/agent/START_HERE.md, docs/agent/RESEARCH_PROTOCOL.md, docs/agent/ROADMAP.md and docs/agent/CURRENT_STATUS.md. Inspect the actual repository and git status. Do not assume planned modules or commands are implemented. Provide: (1) actual repository inventory, (2) gaps against M0, (3) a minimal implementation plan for M0 with acceptance tests, (4) sources/commands you need to verify dataset availability. Do not download large datasets, train models, or scaffold the full service yet.

## Prompt 1 — M0: Benchmark schema audit and reproducible loader

> Implement the smallest reproducible TimelyQABench data-audit path. First inspect the official kaist-dmlab/TimelyRAG repository and check dataset access and license. Add a typed loader/validator based only on **observed** fields, a small CPU-only sample or fixture, and unit tests for malformed/missing timestamps and labels. Produce or update docs/dataset_card.md with verified schema, split and leakage notes. Do not invent fields or download the full corpus without approval. Report exact commands, test results and blockers; update docs/agent/CURRENT_STATUS.md only with facts verified in this session.

## Prompt 2 — M1: Tested retrieval metrics and BM25

> Implement a deterministic query-grouped retrieval evaluation with tested nDCG@10 and Recall@K. Build the simplest BM25 baseline on a declared small subset of the real benchmark; use the observed query/qrel schema and preserve temporal metadata. Export per-query results and a run manifest. Add at least three hand-crafted metric tests, including all-zero relevance and ranking ties/boundaries. Do not train a neural model. Clearly distinguish local-subset scores from official benchmark results.

## Prompt 3 — M2: Dense and time-aware ranking

> Add a pinned compact sentence-embedding baseline and explicit time-aware candidate features. Ensure BM25, dense, hybrid and temporal systems are compared on the same split and documented candidate settings. Add time-validity boundary tests, confidence intervals or per-query deltas and an error analysis for wrong-version retrieval. GPU model downloads/training require a cost/size estimate and permission first.

## Prompt 4 — M3/M4: Leak-free adaptive router

> Build a fast/deep reranking comparison from frozen, cached candidate pools. Generate router labels **only from training-fold gold relevance and offline fast/deep outcomes**. Restrict online router inputs to pre-reranking query/metadata/fast-ranking features. Train a simple rule baseline and logistic/CatBoost router. Validate on unseen queries, plot nDCG@10 vs measured reranker cost, and add explicit tests preventing deep scores or qrels from becoming online features. Include a no-improvement scenario in the report and preserve all experiment manifests.

## Prompt 5 — Versioned service end-to-end vertical slice

> Design and implement one minimal vertical slice: ingest a text document revision; retain its original version and stable passage IDs; search using an optional as_of date; return cited evidence with exact version; ingest a newer revision and repeat the historical query without changing its historical meaning. Start with FastAPI. Add typed request/response models, integration tests, a schema migration if persistent DB state is introduced, and a working example. Avoid adding Rust, Go, queues or optional generation.

## Prompt 6 — ChangeImpact product feature

> Add a conservative ChangeImpact feature on top of stored answer-to-evidence references. When a cited passage changes, emit a review_required status with old/new evidence links. Do NOT mark the answer factually false on this signal alone. Historical as_of answers must not be invalidated merely by a later publication. Add unit and E2E tests with an unchanged clause, a changed clause, and a historical question, plus an audit log of review status updates.

## Prompt 7 — Independent experiment review

> Audit our ML pipeline for data leakage and evaluation inconsistency. Review dataset/qrel joins, split overlap, timestamp semantics, router target generation, feature provenance, candidate-pool parity, score definitions and latency methodology. Do not silently change results. Provide a prioritized list of concrete issues with file/line references, minimal patches and regression tests; report unknowns separately.

## Prompt 8 — Load tests and production-quality handoff

> Add controlled load tests for fast-only searches, mixed fast/deep routing, and ingestion while searches are active. Report concurrency, warm-up, payload sizes, p50/p95, error rate, RPS and peak GPU memory. Keep optional LLM generation separate. Add structured request IDs and metrics, deterministic smoke tests and a clear local deployment guide. Never state performance numbers that were not actually measured.

## Prompt 9 — Review the current project status

> Verify the implementation against docs/agent/ROADMAP.md. Inspect source files and run only safe available tests. For each milestone mark: implemented and tested / partially implemented / not started / blocked. Update docs/agent/CURRENT_STATUS.md using verified facts and exact commands, then recommend one smallest next task. Do not mark planned README features as completed merely because they are described there.

## Good interaction pattern

For any multi-file task request:

1. Ask the agent for a **plan only** if architecture or scope is uncertain.
2. Approve a single milestone or vertical slice.
3. Ask for tests and evidence of the result.
4. Have it write a short factual handoff note before switching agents/sessions.

Keeping every session bounded makes cross-agent context stable and avoids accidental architectural drift.
