# Current project state

Updated **2026-10-23**. The experiments and final report are complete.
Read this first, then the relevant linked record. This file authorizes no new run.

## Project and delivered report

**Robustness of Turkish Punctuation Restoration under Informal Writing Noise**
(CMPE 586), **Kagan Can**. Compare published punctuation BERT, matched clean and
noise-augmented BERTurk fine-tuning, and a case-loss removal arm.

- [Public report PDF](output/pdf/turkish-punctuation-report.pdf): **10 pages**;
  7 main-text pages, references, and 2 pages of appendices; official unchanged
  NeurIPS 2026 **preprint** style. No conference submission/acceptance claim.
- [Editable source and build instructions](report/README.md), complete main and
  sensitivity tables, per-class/contextual diagnostics, training history,
  primary-source references, limits and AI-assistance disclosure.
- Author/student number supplied by the user. The number appears only in the
  Git-ignored local `output/pdf/course-submission.pdf` and private author file.
  Public PDF/source excludes it. Final course-specific format is not yet supplied.
- [README](README.md) presents results; [documentation guide](docs/README.md)
  separates current methods/results from dated records. [Reproduction](docs/reproduction.md)
  distinguishes public report rebuilding from exact scientific reruns.
- Original code/docs have user-authorized [MIT licensing](LICENSE); external
  corpus/model/template terms remain separate in [third-party notices](THIRD_PARTY_NOTICES.md).
  [CITATION.cff](CITATION.cff) provides author/title metadata.

## Completed experiments and fixed method

- All three fine-tunings used shared initialization, training seed586, identical
  rows/order, 3 epochs / **10,968 updates**, batch8 × accumulation4, LR2e-5,
  FP32/eager. All selected epoch3 by highest **clean-validation** punctuation
  macro-F1, earliest exact tie. No noisy-validation checkpoint selection or
  hyperparameter search. Three epochs are a matched budget, not proven optimal.
- Four labels O/PERIOD/COMMA/QUESTION_MARK; first-subword supervision and ignored
  `-100` continuation/padding. Noise preserves original word targets/IDs and
  retokenizes. Case loss, Turkish folding, interior swaps; component budgets
  2/5/10%. Combined has larger union coverage. Removal does not redistribute dose.
- Main GPU validation grid: **117 conditions**. Removal validation: **5 new +
  15 reused references**. [Grid](docs/robustness-grid-results.md),
  [ablation](docs/case-loss-ablation-results.md).
- Held-out: **122/122 forward conditions / 244 cohort scores**, completed
  October23 13:43:43 Istanbul; frozen protocol r2 before project test scoring.
  Independent local raw-target recomputation matches Colab; no extra inference.
- Primary: **14,600 rows / 181,485 words** after 61-row empty-word quarantine.
  Sensitivity: **14,445 / 181,116**, removing 155 exact train/validation input
  matches from the same predictions; no resampling or independent-test claim.

| Primary held-out macro-F1 (%) | Clean | Augmented | Paired gain (pp) |
| --- | ---: | ---: | ---: |
| Clean | 84.3437 | 84.4824 | +0.1387 |
| Combined-heavy | 76.7502 | 80.5289 | +3.7787 ± 0.2191 |

Mean/SD measures three corruption seeds only. Medium/seed586 Augmented minus
Without case loss: clean +0.0905, case +0.9585, folding −0.0205, swaps −0.1078,
combined +0.8799 pp. [All results](docs/held-out-test-results.md),
[protocol/literature](docs/held-out-test-protocol.md),
[completion/archive](docs/held-out-test-completion.md).

## Evidence, storage and publication checks

- Run `60e31867535de346accb64be45df04984434f6ff328b31c8819115cb689538a0`;
  analysis `e39f8803a6e6614363a6db97c1edce870757a2e909dec9dba5e93d0d9edc7f91`.
- Raw outputs: `artifacts/evaluations/held-out-test-v1/`; verified final ZIP
  (570 files + manifest, 16,011,054 bytes), local CPU analysis and receipts:
  `artifacts/evaluation-preparation/held-out-test-v1/completion-2026-10-23/`.
  ZIP SHA: `2463a23ed22dc9543341393bedfb902f693448330da860374c708748d4830925`.
- All epoch models/shared initialization remain in existing local training
  archives; selected Drive exports are retained. [Backup map](artifacts/README.md).
- Scoped Drive cleanup completed: **1,148 backed copies / 925,753,181 bytes**;
  all 291 protected IDs plus new readback receipt remain. The manifest mismatch
  was resolved with its exact historical Colab counterpart and local supplement;
  original files/hashes remain. No further cleanup is pending.
  [Outcome](docs/drive-cleanup.md), receipts under
  `artifacts/storage-audit/drive-2026-10-23-v1/cleanup-2026-10-23/`.
- Publication checks parse sources/JSON/7 cleared notebooks, verify public links,
  report bindings and absence of data/model/private metadata in Git. PDF pages
  are rendered/reviewed; public/private ID separation is checked. No test suite,
  training, inference, noise generation or model loading ran for publication.
- Local read-only repository and fatal/undefined-name lint checks pass. Seven
  pre-existing style findings in frozen historical grid/calibration scripts are
  retained; new publication scripts pass full Ruff. CSV bytes are preserved by
  `.gitattributes` so fresh checkouts retain the original audited SHA bindings.
- User granted `workflow` scope and authorized activation. The
  [Repository checks workflow](.github/workflows/repository-checks.yml) is live:
  push/PR, read-only contents, pinned actions/Ruff, Python3.11; repository/binding
  checks and fatal/undefined-name lint only. The
  [first GitHub run](https://github.com/kckagancan/cmpe586-turkish-punctuation/actions/runs/39975998764)
  **passed every step** on activation commit `8373859`. No scientific run or
  historical unit suite was executed. README shows the live workflow badge.
- Publication receipts, exact Git/remote/CI outcome and full changed-file inventory:
  `artifacts/publication/final-report-2026-10-23/`. Frozen recipes, numerical
  sources and original snapshots were preserved; prior uncommitted experiment
  work is included in the publication history.

## Next action and remaining limits

The planned experiments, analyses, report, GitHub publication, backups, scoped
cleanup and repository automation are complete. No experiment or cleanup job is
running and no additional research step is required by the recorded plan.
The remaining course action is submission, adapting the paper only if the
instructor supplies a final template/page limit. CI activation receipts and final
job outcomes are under
`artifacts/publication/github-actions-2026-10-23/`.

One training seed and synthetic noise only. Dataset licensing/exact split
provenance and document independence remain unresolved; published-model prior
test exposure is unknown. Validation informed development; this is not a blind
preregistration. Future test-informed changes are exploratory and need genuinely
unseen data for strong independent confirmation. Do not rerun completed work
without a reason or alter frozen recipes to pass a check.
