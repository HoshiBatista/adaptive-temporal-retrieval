# Agent documentation index

The root `README.md` is the **project specification**. The root `AGENTS.md` is the **shared agent contract**. Root `CLAUDE.md` imports that contract for Claude Code.

| File | Use when |
|---|---|
| [START_HERE.md](START_HERE.md) | Installing the agent kit and understanding the workflow |
| [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md) | Handling datasets, training, baselines, leakage, metrics, experiment quality |
| [ARCHITECTURE_DECISIONS.md](ARCHITECTURE_DECISIONS.md) | Defining services, data model, module boundaries, technical decisions |
| [ROADMAP.md](ROADMAP.md) | Deciding the next milestone and its acceptance criteria |
| [CURRENT_STATUS.md](CURRENT_STATUS.md) | Handing off verified work between agents/sessions |
| [PROMPTS.md](PROMPTS.md) | Copy-paste task prompts for Codex or Claude Code |

**Start with M0, not the full product:** verify TimelyQABench and its evaluation protocol, then create a small reproducible BM25 baseline. The large README describes future capabilities, not implementation status.
