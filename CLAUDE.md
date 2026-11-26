# CLAUDE.md — TemporalProof

@AGENTS.md

## Claude Code operating guidance

- Treat imported `AGENTS.md` as the shared project contract; avoid duplicating it here.
- At session start inspect `README.md`, `docs/agent/CURRENT_STATUS.md`, the real repository tree, and the most relevant file under `docs/agent/`.
- For nontrivial multi-file tasks, propose a compact plan before editing. Implement one bounded milestone rather than generating an entire imagined application.
- Before invoking a command, check that its entry point exists. The README's example commands and endpoints are **planned, not guaranteed to work**.
- Do not assume results from earlier chat sessions exist in the codebase. Verify files, trained models and measured metrics on disk.
- On research work, inspect `docs/agent/RESEARCH_PROTOCOL.md` first; on platform changes, inspect `docs/agent/ARCHITECTURE_DECISIONS.md`.
- Keep the latest verified progress in `docs/agent/CURRENT_STATUS.md`, including exact commands executed and any remaining blockers. Never change checked-off task statuses without evidence.
- Prefer small diffs, explicit dependency choices, deterministic CPU tests and reproducible commands. Avoid background downloads or expensive GPU training unless explicitly requested.
- For your final task report use **Summary / Files / Validation / Remaining risks / Next action**. State clearly when validation was not run.

## Immediate first-session task

If the user asks you to start coding without specifying a component, follow **Milestone M0** in `docs/agent/ROADMAP.md`: inspect and validate the main benchmark schema and build a tiny reproducible data-audit/test foundation. Do not start with a UI, LLM, Rust gateway or large-scale indexing.
