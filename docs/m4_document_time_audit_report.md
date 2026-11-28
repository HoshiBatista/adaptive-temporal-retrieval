# M4.6 — Explicit rule-start dates in training passages

The frozen extractor found a unique day in **5,515 of 18,682 distinct training
passages (29.5%)**. In **1,029** cases the selected day differed from the insertion
day. This makes the two fields useful to track separately. The benchmark supplies
neither document applicability labels nor verified dates for these statements.
This stage measured cue coverage. Retrieval evaluation remains pending.

## Population and inspection

The audit verifies the pinned archive and recomputes the existing split. It takes
the 20,580 passages associated with the 2,058 training queries, removes normalized
text shared with validation or test, and uses the lowest document ID for each
remaining normalized text. Held-out texts participate only in the overlap check
and split verification. Near duplicates and shared semantic families can remain.
The extractor consumes passage text. Insertion timestamps enter only the comparison
after extraction. Relevance judgments and answers play no role.

A fixed sample of 40 documents was drawn by hashing the seed 1729 and representative
document ID. All 40 texts were inspected during rule development, and the outputs
were reviewed. The sample contains explicit start statements, hiring thresholds,
annual reset dates, rule replacement, a bare timestamp and date ranges. In one
passage a recurring leave interval causes abstention despite a separate explicit
rule-start statement. A clause about possible future changes can also suppress
a clear start. These tradeoffs follow the conservative rule and limit coverage.
The sample informed the extractor and supports qualitative inspection only.

| Decision | Training population | Inspection sample |
| --- | ---: | ---: |
| Unique explicit start day | 5,515 | 16 |
| No recognized calendar mention | 4,633 | 8 |
| Other dates only | 5,697 | 9 |
| Range or qualified wording | 2,761 | 7 |
| Start with precision other than day | 54 | 0 |
| Multiple start dates | 12 | 0 |
| Missing year or invalid start | 10 | 0 |

Among the 5,515 selected days, 4,475 matched the calendar day of insertion, 323
preceded it, 706 followed it, and 11 had an unknown insertion time. All 16 selected
days in the inspection sample matched their insertion day. The population counts
provide the broader comparison.

## Frozen extraction contract

The parser reuses the existing normalized calendar spans and recognizes explicit
start wording near a rule, a named start-date field, or a standalone enactment
statement. It keeps date mentions for inspection and emits a day only for a unique
explicit start. Range signals and recognized qualifications trigger passage-level
abstention. Repeated statements of the same start are allowed. Each date span
retains its textual precision. Employment thresholds, standalone timestamps and
amendment dates do not supply the rule start. The corpus's language and templates
limit the rule's portability. Unsupported wording can hide a relevant start and
unrecognized qualifications can still produce false matches.

The selected day is a claim made in the passage. It carries no implied end of
validity or assertion of historical availability. Keep this extractor and the
M4.4 query parser pinned for any future train comparison. First count how many
frozen candidate pools contain an exact day match for a query that explicitly
requests a policy start. Report coverage before scoring. A later comparison can
test a bounded promotion of matches against unchanged hybrid with the original
candidate pools and judgments. Treat general as-of applicability as a separate
problem.

## Reproduction

Run from the repository root with the existing local archive and split.

```bash
uv run --frozen --extra tracking --extra dense --extra routing python -m temporalproof.evaluation.document_time_audit --output artifacts/m4/document_time_audit_train_repro.json
cmp artifacts/m4/document_time_audit_train_repro.json experiments/results/m4_document_time_audit_train_v1.json
TEMPORALPROOF_REFERENCE_TESTS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --frozen --extra tracking --extra dense --extra routing pytest -q
```

Two runs produced identical JSON. The [aggregate](../experiments/results/m4_document_time_audit_train_v1.json)
records source and extractor hashes, the sample IDs and the coverage counts.
The raw inspection sample stays under ignored `artifacts/m4/`. The local MLflow
run `8b1c724c338047ebb5580e2c5c54c849` contains the aggregate only. All 282 tests
passed with upstream integration enabled. The audit runs on CPU.
