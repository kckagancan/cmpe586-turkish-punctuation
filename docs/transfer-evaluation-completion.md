# Supplemental transfer: completion and verified retention

Completed **2026-11-04**. This supersedes the running-job handoff in the
[execution record](transfer-evaluation-execution.md); frozen experimental
sources and the original study remain unchanged.

## Completed work and checks

- All **52/52 forwards /104 cohort score sets** completed on the pinned T4.
  Four existing selected models each have a clean anchor plus four conditions
  at seeds586/587/588. No new training or checkpoint selection occurred.
- The completed Drive run and original readiness/source tree were packed into
  a304-file ZIP. Every member and the final ZIP passed SHA-256 readback.
  Local download:6,785,495 bytes, SHA
  `b278f33ee26508b6f7bf7125140be65d569739d11282fedefb42214cb97081ef`.
  Safe restoration refused unsafe paths or conflicting existing files.
- Local CPU analysis independently recomputed every saved prediction condition
  against original raw targets. Exact row coverage/order/targets,52 four-file
  condition commits, all104 metrics/supports/confusions, own-anchor drops and
  realized input exposure matched. It also reconciled frozen inputs/profile,
  runtime/readiness/models/source hashes and shared padded batch work.
- All four fresh clean anchors exactly match the original held-out metrics
  in both cohorts. The same sensitivity mask filters the original primary
  predictions without resampling. No inference was repeated for these checks.
- Every predefined augmented-minus-clean, augmented-minus-without-case and
  partial-minus-matched-full contrast retains per-seed values, mean and sample
  SD. Full class precision/recall/F1/support and four-class confusions are public
  aggregates; no raw text or per-row predictions are published.
- The current public/private final reports have15 pages. All pages were
  rendered with system Poppler and visually reviewed; detailed new tables and
  figures were also inspected. Numeric contents and student-metadata separation
  were checked. There are no unresolved references or overfull boxes.
  Tectonic0.17.0 was downloaded from its official release and verified by the
  published asset SHA because the former temporary compiler was unavailable.
- Source/Ruff and repository/report-binding checks are publication checks;
  **no unit suite was added or run**. Dataset licensing/provenance and model
  exposure limitations remain unresolved.

## Results and interpretation

[All results](transfer-evaluation-results.md) include both cohorts and every
predefined contrast. Augmented-minus-clean primary macro-F1 gains are
**0.2491 ±0.0843 pp** for elongation, **0.2290 ±0.0635** for internal omission,
and **0.1882 ±0.0425** for partial folding. The matched-full control is
**0.1279 ±0.0446**. These are descriptive differences for one training seed;
corruption SD is neither a confidence interval nor training uncertainty.

Part of each absolute advantage already exists on clean inputs(+0.1387pp).
Own-anchor drops are reported separately. Retaining case loss has positive
mean absolute scores against the removal arm, but does not uniformly improve
relative degradation. Partial-minus-full signs differ by model; identical word
masks do not match character dose. This is a follow-up on controlled distortions
of already scored project texts, not an independent natural-corpus test.

## Identities and locations

- Run: `8cf43b9d6a166baf005fb0320a52177f859c4b6215d06baaa0f93be57156d200`.
- Analysis: `7847a6b85dcf7d4f1296cff6495974bcbf4103c635b628fce2d8820a45ef06f6`.
- Raw run and immutable CPU analysis:
  `artifacts/evaluations/transfer-noise-v1/` and its`analysis-v1/`.
- Local completion receipts/logs/PDF checks/inventory:
  `artifacts/evaluation-preparation/transfer-noise-v1/completion-2026-11-04/`.
- Final local ZIP: `transfer-noise-v1-results-and-report-v2.zip` in that directory.
  **411 files plus manifest /27,337,920 bytes**, SHA-256
  `c956e2ee2eb505982df9a0adfb9a09d0263b1f9598be7f54cba1e76798d6d6e1`.
  Its`final-archive.json` records exact members/bytes/SHA and verification state.
  It includes raw saved results, analysis, freezes/readiness, report sources,
  both new PDFs and preserved original report/results archives. **Keep local:**
  private student metadata is included. Weights remain separately retained.
- Drive raw run remains under the project root at
  `artifacts/evaluations/transfer-noise-v1/`; raw transfer ZIP/receipt remain at
  `preparation/transfer-noise-v1/completion-2026-11-04/`.
- [Original study versions and backup map](report-versions.md): exact10-page
  PDFs, prior source/metric snapshot and original16MB results ZIP remain separate.

No training/evaluation job or cleanup job is left running. Colab remains visible
with a read-only completion summary. No Drive file is deleted in this task.
Any later cleanup requires its own scoped backup/retention check; no account-wide
Trash or permanent deletion is scheduled.
