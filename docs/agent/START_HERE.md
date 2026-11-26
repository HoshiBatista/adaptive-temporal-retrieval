# Agent onboarding — TemporalProof

This folder is a **developer-agent operating guide**, not an implementation. The authoritative specification is the root `README.md`. No module, trained model, API endpoint or benchmark score is assumed to exist merely because it is described there.

## Files and where to place them

```text
temporalproof/
├── README.md                         # your existing long-form project README
├── AGENTS.md                         # repository-wide instructions for Codex/agents
├── CLAUDE.md                         # Claude Code entry point; imports AGENTS.md
└── docs/
    └── agent/
        ├── START_HERE.md
        ├── RESEARCH_PROTOCOL.md
        ├── ARCHITECTURE_DECISIONS.md
        ├── ROADMAP.md
        ├── CURRENT_STATUS.md
        └── PROMPTS.md
```

Copy the whole kit into the **root of the existing Git repository**, preserving `docs/agent/`. Keep your previously generated `README.md`; it does not need to be replaced.

## Who reads what

- **Codex / agents supporting AGENTS.md:** read the root `AGENTS.md` automatically according to their instruction-file behavior.
- **Claude Code:** reads `CLAUDE.md`, which explicitly imports `@AGENTS.md`. This keeps one shared ruleset and a small Claude-specific supplement.
- **Human contributors:** start with `README.md`, then `ROADMAP.md`, then `CURRENT_STATUS.md`.

The tool must inspect actual repository files before acting. Agent instructions guide implementation; they do not guarantee access to data, APIs, model weights or external services.

## The research/product split

**Core research (mandatory):** adaptively choose a lightweight ranking path or a deep temporal reranking path, using features available before the expensive stage. Compare against genuine baselines on a fixed held-out protocol.

**Application (mandatory for a complete product):** version-aware document ingestion, time-bounded search, provenance/citations, REST API and demonstrable UI, tests and observability.

**ChangeImpact (secondary):** notify that a stored answer's cited evidence was modified and requires review. This is not proof that the old answer is factually wrong.

**Optional:** grounded generation with a small BF16 model, Rust gateway if measurements justify it, extra benchmarks. None should block the initial reproducible ranking research.

## Recommended first task

Send the agent the **Milestone M0 kickoff prompt** from `PROMPTS.md`. It should perform a repository audit and inspect benchmark schemas on a small subset. The outputs should be a verified dataset card, split/field audit, smoke tests and a list of verified blockers.

## Avoid these traps

- Asking an agent to “build the whole TemporalProof architecture” in one prompt.
- Immediately committing 30 empty folders, a fake API, or placeholder passing tests.
- Training a large LLM before proving that time-aware retrieval is measurable.
- Using future document revisions to answer a historically constrained question.
- Training a router on an oracle label computed from the **test** queries.
- Claiming that a changed document implies that every related answer is invalid.
- Comparing nDCG from different evaluation definitions.

## External sources to verify in the actual environment

- Main data and baseline: <https://github.com/kaist-dmlab/TimelyRAG>
- Secondary temporal-retrieval benchmark: <https://github.com/tempo-bench/Tempo>
- Agent instructions convention: <https://agents.md/>
- Claude Code project memory: <https://code.claude.com/docs/en/memory>

Use the benchmark repositories' own licenses, dataset cards and pinned commits as the final authority. Do not assume their interfaces or data size never change.
