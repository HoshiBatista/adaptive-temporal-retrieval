# TimelyQABench v2 — Company Policies: verified M0 data card

Audited on **2026-11-26**. These are data-integrity observations, not retrieval scores.
The executable adapter currently supports only this domain and revision.

## Source and provenance

- Repository: <https://github.com/kaist-dmlab/TimelyRAG>
- Revision: `5213a0c1743f23fd52a258005c3887aaf7e8cd0f`.
- Archive: [pinned datasets.zip](https://github.com/kaist-dmlab/TimelyRAG/blob/5213a0c1743f23fd52a258005c3887aaf7e8cd0f/datasets.zip), 5,765,226 bytes.
- Archive SHA-256: `a006481ffdf723d4b15554da2c621ebdfefd59b78ad37c58a633b3b71de247ad`.
- Selected member: `timelyrag_company_policies_dataset_v2.jsonl`, 14,282,297 uncompressed bytes.
- Member SHA-256: `fe76970ecb895debe3e4713de67c7f5a265a63be5209bb478850c59234aad6f5`.
- Machine-readable provenance, including inspected reference-code hashes:
  [source manifest](../configs/datasets/timelyqa_company_policies.json).

The archive contains four domain JSONL files and macOS resource-fork files. It contains
no separate train/validation/test files or license file. The official configuration calls
the university input `timelyrag_yonsei_dataset_v2.jsonl`, while the archive member is
`timelyrag_university_dataset_v2.jsonl`. The Company Policies path matches.

## Origin and rights

The [paper, Section 4 and ethics statement](https://aclanthology.org/2026.findings-emnlp.398.pdf)
describes synthetic Korean QA/corpus generation with Llama-3.3-70B, model-based validation,
and a small human validation study. It describes a permissive research release but does
not identify precise license terms in the inspected passage.

At the pinned revision, the repository tree has no LICENSE/COPYING/NOTICE file; GitHub
license metadata was null at audit time. Exact data/code redistribution terms remain
**unresolved**. This project's MIT license does not license third-party benchmark data.
Commit aggregate observations, provenance and independently authored synthetic fixtures;
keep raw content and reference code in ignored local storage. No benchmark text is copied
into this card or the tests.

## Observed schema

Every one of the 3,000 records has these fields:

| Field | Raw JSON type | Observed meaning / handling |
|---|---|---|
| `index` | integer | Unique record/query ID within the domain |
| `category` | string | One of 10 observed category strings; do not equate them with the paper's six-category taxonomy without mapping |
| `query_ko` | string | Query text; all first 50 contain Hangul |
| `query_insert_time` | string | Query insertion/issuance time; not the requested event time |
| `answer` | string | Reference answer; one empty string; unused by retrieval metrics |
| `docs_ko` | array of objects | Ten associated passages per record; corpus is their domain-wide union |
| `gold_docs_final` | array of integers | Zero-based local offsets into `docs_ko` |
| `gold_docs_final_global` | array of integers | Positive document IDs, binary relevance |

All 30,000 document objects have `global_doc_id` (integer), `text` (string), and
`doc_insert_time` (string). Ten also have `title` (string); this field was absent in
the first-50-record sample. IDs identify benchmark passages; no separate source-document,
version-chain, or validity-interval field is provided. Scope IDs by domain when combining
datasets. Do not invent stable document families from numeric IDs.

Metadata-only example shape, with all content and actual identifiers omitted:

```text
record: {index: integer, category: string, query_ko: string,
         query_insert_time: string, answer: string,
         docs_ko: [{global_doc_id: integer, text: string,
                    doc_insert_time: string, title?: string}],
         gold_docs_final: integer[], gold_docs_final_global: integer[]}
```

## Counts and relevance integrity

| Measurement | Full Company Policies domain |
|---|---:|
| Query records / unique record IDs | 3,000 / 3,000 |
| Document occurrences / unique document IDs | 30,000 / 30,000 |
| Unique exact query texts | 2,976 |
| Repeated exact-query groups / extra occurrences | 17 / 24 |
| Unique exact document texts | 27,199 |
| Repeated exact-document groups / extra occurrences | 2,157 / 2,801 |
| Total positive query–document judgments | 7,533 |
| Queries without positive judgments | 0 |
| Local/global positive-set disagreements | 0 |
| Empty reference answers | 1 |

Positive-document cardinality per query: **1: 124; 2: 1,441; 3: 1,254; 4: 147;
5: 31; 6: 1; 8: 2**. These are counts of binary positives, not graded relevance levels.
Unjudged corpus passages are not proven negatives. Identical text under different IDs
requires error analysis; preserve official IDs and labels instead of silently deduplicating.

## Temporal metadata

- Explicit `query_event_time` and document `event_time` are absent throughout the domain.
- All 3,000 query insertion strings parse under the documented naive format.
- **47 / 30,000 document insertion strings do not parse** under the exact
  `YYYY-MM-DD HH:MM:SS` policy. These include Korean-formatted dates and corrupted values.
- Raw timestamp strings are preserved. Malformed, missing, null and empty timestamps
  produce unknown parsed times; malformed and absent counts are reported separately.
  No Korean-date normalization, corruption repair, UTC assumption, event extraction or
  validity inference is performed in M0.
- The 29,953 parsed document timestamps and 3,000 parsed query timestamps have no timezone.
  Parsed query range: 2022-01-01 09:00:00–2025-12-31 14:30:00; document range:
  2002-01-01 09:00:00–2031-01-01 00:00:00. These are synthetic data values.
- **1,253 positive documents across 911 queries** have a parseable insertion time later
  than their query insertion time. For **146 queries**, every positive is later-dated.
  This is a diagnostic mismatch for strict known-at evaluation, not proof that a label
  is wrong. Do not discard these labels or reinterpret insertion time as valid time.

The [official loader](https://github.com/kaist-dmlab/TimelyRAG/blob/5213a0c1743f23fd52a258005c3887aaf7e8cd0f/TimelyQABench_datasets.py)
can derive a query event time from text, and reference temporal scoring uses fallbacks.
Derived times must be identified separately if implemented later.

## Sampling, groups and splits

The audit inspects every record structurally and summarizes the **first 50 records / 500
documents** as a deterministic, non-random schema sample. This is not a representative
language or quality estimate and is not a test split.

The report exports metadata-only query groups using
`SHA256(NFKC(text).casefold() with collapsed whitespace)`, version
`nfkc_casefold_whitespace_v1`. There are 2,976 groups. Eleven repeated groups have different
query insertion times: retain each record's temporal context, while keeping its group
together in any future split. Grouping does not merge records or gold sets.

**M0 did not create a train/validation/test assignment.** M1 now freezes an
[ID-only grouped split](../data/splits/company_policies_v1.json): 2,058 train / 495
validation / 447 test queries. Its expanded audit connects numeric query templates,
near-query templates and normalized exact shared passages, producing 2,891 groups.
See the [M1 report](m1_bm25_report.md) for policy, hashes and limits: semantic families
and near-duplicate passages are not exhaustively established. Shared corpus evidence is
normal for retrieval; label-derived training features and family memorization must be
assessed separately. Held-out domain evaluation remains a separate generalization test.

## Reproduction and resources

From the repository root, after installing `uv`:

```bash
uv sync --frozen --group dev
uv run --frozen temporalproof-audit --download
uv run --frozen temporalproof-audit
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

Only `--download` permits benchmark network access. It fetches the pinned 5.8 MB archive,
checks a size bound and SHA-256, and refuses to replace a mismatching cached file.
Normal tests use an independently authored fixture and mocked downloads; no dataset or
model is fetched. The archive is read directly without extraction; macOS files are ignored.

Default raw path: `data/raw/timelyqa/<revision>/datasets.zip`.
Default derived report: `artifacts/audits/company_policies.json`.
Both are gitignored. Reports contain counts, schema presence, hashes and record IDs,
not query, answer or passage text. Repeated audits are byte-identical.

Full corpus text length is 4,178,381 Unicode characters (median 120, minimum 9, maximum
958 per passage). This does not establish encoder token lengths. One local CPU audit
using Python 3.13.11 on Linux x86-64 completed in 0.56 seconds with peak process RSS
76,272 KiB reported by `/usr/bin/time -v uv run --frozen temporalproof-audit --sample-size 50`.
This is one audit
measurement, not retrieval latency, throughput or a cross-machine memory guarantee.

## M0 gate and remaining questions

Access, schema, document-level positive IDs and deterministic auditing are verified.
The data supports a label-based retrieval baseline. Full valid-time/known-at correctness
is not established by the provided metadata. Exact license terms, residual semantic-family
grouping risks and the future-positive mismatch remain open. A basic retrieval experiment may
retain original labels and explicitly state these limits; any historical filtering
experiment needs a separately documented protocol without silently rewriting gold labels.
