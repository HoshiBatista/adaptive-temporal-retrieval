# TemporalProof — verified project status and agent handoff

Last verified: **2026-11-26**. Only executed work is recorded below.

## Current state

- **M0:** reproducible Company Policies data audit, strict typed JSONL adapter, synthetic
  CPU fixture and metric/schema/audit tests implemented and run.
- **Environment:** `pyproject.toml`, `uv.lock`, Python >=3.11 declared; validation executed
  on Linux x86-64, CPython 3.13.11. Actual locked versions include Pydantic 2.14.4,
  pytest 9.2.0 and Ruff 0.16.10.
- **Research results:** no BM25/dense/hybrid/reranker/router experiments or retrieval scores.
- **Application:** no API, database, UI, generation or ChangeImpact implementation.
- **Data:** pinned 5.8 MB archive in ignored `data/raw/`; derived reports in ignored
  `artifacts/`. No source benchmark content is committed; test data is independently authored.
- **Next milestone:** M1 split protocol and CPU BM25, with the M0 limitations below.

## Verified work log

### 2026-11-26 — M0 / Company Policies audit foundation

- Goal: turn the initial repository/dataset inspection into a repeatable, CPU-only audit.
- Files changed: `pyproject.toml`, `uv.lock`, `.gitignore`, package files under
  `src/temporalproof/data/` and `src/temporalproof/evaluation/`, three test modules and
  `tests/fixtures/timelyqa_synthetic.jsonl`, `configs/datasets/timelyqa_company_policies.json`,
  `docs/dataset_card.md`, `docs/benchmark_protocol.md`, README and agent handoff/protocol/roadmap/decisions.
- Source revision: `kaist-dmlab/TimelyRAG@5213a0c1743f23fd52a258005c3887aaf7e8cd0f`.
  Archive SHA-256: `a006481ffdf723d4b15554da2c621ebdfefd59b78ad37c58a633b3b71de247ad`.
- Setup command executed: `uv sync --group dev`.
- Frozen setup rechecked successfully: `uv sync --frozen --group dev`.
- Validation commands executed:

  ```bash
  uv run --frozen ruff format .
  uv run --frozen ruff check .
  uv run --frozen pytest -q
  uv run --frozen ruff format --check .
  /usr/bin/time -v uv run --frozen temporalproof-audit --sample-size 50
  uv run --frozen temporalproof-audit --download --archive data/raw/download_smoke/datasets.zip --output artifacts/audits/download_smoke.json
  uv run --frozen temporalproof-audit --output artifacts/audits/repeat.json
  cmp artifacts/audits/company_policies.json artifacts/audits/repeat.json
  cmp artifacts/audits/company_policies.json artifacts/audits/download_smoke.json
  git check-ignore data/raw/timelyqa/5213a0c1743f23fd52a258005c3887aaf7e8cd0f/datasets.zip
  git diff --check
  ```

- Results: **61 tests passed**; Ruff lint/format checks passed. Fresh opt-in download,
  cached offline audit and repeated audit succeeded; JSON reports were byte-identical.
  Full audit counted 3,000 queries / 30,000 unique document IDs. The first 50 records
  (500 documents) provide a deterministic schema sample.
- Audit observations: 2,976 distinct query texts, 27,199 distinct document texts;
  no local/global gold mapping disagreements; 7,533 binary positive judgments;
  one empty answer; ten optional document titles; no explicit event-time fields;
  47 unparseable document insertion strings; 1,253 later-inserted positives across
  911 queries (all positives are later-inserted in 146 queries).
- Early checks that failed, now resolved: unknown JSON keys appeared in error locations;
  sanitizer now redacts them and its regression test passes. Initial strict timestamp
  validation and nonempty-answer assumptions stopped the real audit; the adapter now
  preserves malformed timestamp strings, reports unknown parsed times, and accepts the
  observed empty answer without discarding qrels. Initial Ruff line-length findings and
  the final diff's README trailing-whitespace finding fixed.
- Research decisions: ADR-007 preserves raw time uncertainty; ADR-008 separates qrel-based
  metrics from reference reproduction. Reference source files remain unchanged.
- Not run: BM25, any model download/training, final train/test evaluation, GPU checks,
  API/load tests, Python 3.11 execution or clean-machine setup.

## Open questions / constraints

1. **License:** inspected pinned repository/archive contain no explicit license file;
   GitHub license metadata was null. The paper describes a permissive research release
   without specifying precise terms in the inspected statement. Do not redistribute
   benchmark text/reference code on the basis of the project's MIT license.
2. **Temporal evaluation:** insertion time is not valid time. No event-time/validity labels
   are supplied. Strict known-at filtering conflicts with later-inserted positives;
   keep original labels intact and define any temporal slice separately.
3. **Splits:** the archive has domain files, no train/validation/test assignments. The audit
   emits normalized query groups, but near-duplicate templates and shared evidence/families
   still need inspection before freezing splits. No split has been created yet.
4. **Official comparison:** reference nDCG normalizes over retrieved positives; reference
   best-alpha uses query qrels. Neither can silently replace the main qrel-based protocol.
5. **Models / transfer:** Korean support and actual memory/latency are not measured.
   TEMPO integration and any judge API are not implemented.

## Recommended next task

Implement M1: inspect duplicate/family links, freeze a grouped query split and declared
corpus/tokenization policy, then run an independent CPU BM25 baseline. Export per-query
rankings/metrics and failure cases. Use the original positives without silent timestamp
filtering; label all results by their exact protocol. See `docs/benchmark_protocol.md`.
