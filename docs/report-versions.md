# Report editions and original-study retention

The **original study, v1** is retained separately from the **extended final
report, v2**. Later findings do not replace the original scores or rewrite
completed experiment snapshots. The original version has no corpus-transfer
results; the extended report contains both stages and identifies the follow-up.

| Edition | Public PDF | Scope |
| --- | --- | --- |
| Original study v1 | [Preserved original paper](../output/pdf/original-study-v1-public.pdf) | Original matched training, validation, case-loss ablation and122-forward held-out study;10 pages |
| Extended report v2 | [Current final paper](../output/pdf/turkish-punctuation-report.pdf) | Original findings plus train-only corpus evidence and52-forward supplemental controlled transfer |

The original PDF is an **exact byte copy**, not a reconstruction. Public SHA-256:
`0ee8a5a5c917702166ef9f663fcfd649410d7b053659d153badc976baa200139`.
The old private course PDF is likewise preserved unchanged locally. Student
metadata and private wrappers remain excluded from Git.

## Local copies

Paths below are relative to the primary project checkout; these private files
are intentionally absent from public Git.

- `output/pdf/original-study-v1-public.pdf`: exact original public edition.
- `output/pdf/original-study-v1-course-submission.pdf`: exact original student edition.
- `output/pdf/course-submission-report.pdf`: extended student edition.
- `artifacts/publication/original-study-v1/original-study-v1-report-and-public-evidence.zip`:
  independently read-back,216-file pre-edit snapshot plus edition manifest.
  It preserves the original report sources/tables/figures, both original PDFs,
  original aggregate metrics/protocols/code and private course metadata. It is a
  full checkout snapshot from commit`105226c93c3e15a4e5a402391703c85ac90cbc92`,
  so already-completed corpus/preparation documents are also present; the report
  itself contains only the original study. **Keep this ZIP local.**
- `artifacts/publication/original-study-v1/preservation.json`: exact snapshot
  inventory, bytes, SHA-256 values, baseline commit and original run/analysis IDs.

Snapshot ZIP:3,019,922 bytes, SHA-256
`6caa26fd646589d41610fd07202434c1d1e4970e7c70fb825b97ccf666f7b0ec`.

## Original experimental evidence stays separate

- [Original held-out results](held-out-test-results.md), all aggregate CSV/JSON,
  original report table bindings and original figures remain unchanged.
- Raw122-condition results and the completed analysis remain at
  `artifacts/evaluations/held-out-test-v1/`.
- The verified original results ZIP remains at
  `artifacts/evaluation-preparation/held-out-test-v1/completion-2026-10-23/held-out-test-v1-results.zip`:
  16,011,054 bytes, SHA-256
  `2463a23ed22dc9543341393bedfb902f693448330da860374c708748d4830925`.
- Full epoch models, shared initialization and training/validation archives
  remain in their existing ignored locations; see the [backup map](../artifacts/README.md).
  They are not duplicated into the new reporting archive.
- [Supplemental results](transfer-evaluation-results.md) have their own namespace,
  source snapshots, identities, analysis and [completion record](transfer-evaluation-completion.md).

The new local results archive may include old evidence for convenience; these
separate original copies remain available. No completed experiment is rerun
to create a report edition, and no Drive deletion is part of this update.
