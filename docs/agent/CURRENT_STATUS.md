# TemporalProof — verified project status and agent handoff

> **Update this file after implementing or verifying real work.** Do not claim source files, tests, model weights, datasets or scores exist without checking them in the current repository/environment.

## Current state at creation of this agent kit

- **Status:** documentation and research planning.
- **Verified in the generated starter kit:** `AGENTS.md`, `CLAUDE.md`, and the six guidance files under `docs/agent/`.
- **Existing project README:** previously provided separately; the agent must inspect its actual location and contents.
- **Application code / dependency lockfile / trained models / benchmark results:** **not verified** by this kit. Inspect the repository before making any claim.
- **Primary research scope:** adaptive semantic–temporal retrieval using public relevance judgments.
- **ChangeImpact scope:** change-to-citation dependency notifications; no staleness classifier assumed.
- **Next recommended milestone:** M0 dataset and repository audit.

## Verified work log

Use a short entry after each completed task:

```text
### YYYY-MM-DD — task ID / title
- Goal:
- Files changed:
- Commands run (exact):
- Test / lint / benchmark results:
- Dataset/model/source versions involved:
- Verified facts:
- Not run / blockers:
- Risks / open decisions:
- Recommended next task:
```

## Current open questions

1. Which TimelyQABench files are accessible on the development machine, and under what license?
2. What is the correct time semantics of each timestamp field in the observed dataset?
3. Which official query relevance and split protocol is reproducible locally?
4. Which compact dense model and reranker meet the 16 GB VRAM constraint and latency budget?
5. Should a metadata-only product MVP use SQLite initially or PostgreSQL immediately? Decide from implementation needs.
6. Are TEMPO reference judgments available without a mandatory remote LLM judge for the chosen metrics?

## Working agreement

An agent must not auto-check task boxes just because it wrote code. It should record what was actually executed. If a command fails, include the failure in the entry and keep the milestone open.
