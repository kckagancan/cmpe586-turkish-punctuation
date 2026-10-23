# Drive storage inventory and scoped cleanup proposal

**Historical pre-cleanup record.** The subsequent user-authorized cleanup is
complete; see [verified outcome and manifest resolution](drive-cleanup.md).
Statements below about pending scope/readback and retained originals describe
the initial inventory handoff, before that later operation.

Audited **2026-10-23** after the held-out result archive was verified. This is a
read-only inventory/backup-matching outcome: **no existing Drive file was changed,
trashed or permanently deleted**. It is not evidence that cleanup was executed.

## Complete project inventory

The Google Drive connector recursively listed the observed project root and
all descendants: **424 folders including the root**, **1,439 unique files**, no
unlisted folder or listing error. Listings were metadata-only; no model weights
were downloaded, rehashed or loaded. No account-wide search or Trash cleanup was
performed. The inventory is scoped to `MyDrive/cmpe586-turkish-punctuation`.

Total current file sizes: **2,859,545,316 bytes / 2.860 GB / 2.663 GiB**.
This is the sum of listed live project-file sizes, **not an account storage-quota
measurement**. Drive's quota, other account content, revisions and Trash were not
queried. These numbers do not promise an immediate quota reduction.

| Project scope | Files | Bytes | Decimal MB |
| --- | ---: | ---: | ---: |
| `training-v1/` | 165 | 1,346,527,747 | 1,346.53 |
| `initializations/` | 7 | 441,153,747 | 441.15 |
| `inputs/` | 75 | 586,511,782 | 586.51 |
| `artifacts/` | 1,104 | 380,033,140 | 380.03 |
| `preparation/` | 84 | 73,750,079 | 73.75 |
| `archives/` | 4 | 31,568,821 | 31.57 |

Three selected epoch-03 model files plus the shared initialization account for
**1,760,570,624 bytes**. These are the deliberately retained large assets, not
leftover optimizer checkpoints. The earlier scoped training cleanup remains
complete; it was not repeated.

## Locally backed candidates, pending a new cleanup scope

| Candidate group | Files | Bytes | Decimal MB |
| --- | ---: | ---: | ---: |
| Augmented training views, seed586 / all three epochs | 31 | 221,310,174 | 221.31 |
| Without-case-loss training views, seed586 / all three epochs | 31 | 209,588,210 | 209.59 |
| Archived validation/ablation/held-out evaluation copies | 1,083 | 379,001,017 | 379.00 |
| Three temporary transfer ZIPs listed below | 3 | 115,853,780 | 115.85 |
| **Total proposed file scope** | **1,148** | **925,753,181** | **925.75** |

The three transfer files are:

- `inputs/case-loss-ablation-v1-seed586-inputs.zip` (42,800,098 bytes).
- `preparation/held-out-test-v1/transfer-v1/held-out-inputs-v1.zip` (72,979,242 bytes).
- `preparation/held-out-test-v1/transfer-v1/held-out-source-v1.zip` (74,440 bytes).

The proposed evaluation scope is the three model-specific GPU validation grids,
the without-case-loss validation evaluation and `held-out-test-v1`. It excludes
the shared `artifacts/evaluations/robustness-grid-gpu-v1/` reference/calibration
tree. Four zero-byte `.run.lock` files were deliberately absent from the original
result archives and are **retained outside the candidate list**. This is an exact
**file-ID scope**. A container operation is allowed only after a fresh recursive
listing proves that every file beneath it belongs to the agreed subset; it must
not include an unlisted file. The four lock files and their parent run folders
are not approved as whole-folder targets.

The exact 1,148 IDs, paths, sizes, expected local hashes and backup paths are in
`artifacts/storage-audit/drive-2026-10-23-v1/cleanup-candidates.json`.
Candidate identity:
`db40e93dce484d9dce363ec5994052836d142a515bc73ade52234b42a9e09368`.

Existing policy retained these inputs and original evaluation outputs on Drive.
Removing them is a **new retention decision**, not a continuation of the earlier
optimizer/epoch cleanup. [AGENTS.md](../AGENTS.md) requires only the agreed scoped
Drive cleanup; the user's current request authorized inventory and identifying
candidates. This document is the concrete scope for a subsequent decision.

## Backup verification actually performed

- Existing validation-grid, ablation-evaluation and held-out result ZIP SHA-256s
  matched their original verified receipts; manifest membership was checked.
- Every proposed evaluation file has a local file with the archive's exact
  byte count and SHA-256. Its current Drive-listed byte count also matches.
- Both local three-epoch training-view trees match their frozen epoch hashes;
  augmented historical source snapshots also match their saved hashes.
- The existing without-case-loss input ZIP was reused. Its SHA and member index
  matched the frozen input recipe; matching local files were hashed.
- A new **local-only** augmented-input ZIP was prepared from the existing audited
  31-file tree. No noise generation, re-tokenization, model inference or training
  was performed. All ZIP member paths/counts/bytes/SHA values passed readback.
- Three temporary transfer ZIPs match their recorded local SHAs and current
  Drive-listed sizes. They remain on Drive until a scope is agreed.
- All three local selected model files and the shared initialization remain
  present with expected sizes. Existing model-archive/offline-load receipts were
  reused; large weights and model archives were not rehashed or reloaded.

New augmented input archive:

- Path: `artifacts/storage-audit/drive-2026-10-23-v1/augmented-training-inputs-seed586.zip`.
- **44,255,470 bytes**, **31 source files plus manifest**.
- SHA-256: `0296e955bb772835d47bfe8529e317cf49fdb4f6cfde252f8e2b95c74d02652d`.
- Prepared input identity:
  `cf0acd2af581f9e145b53ece0578e52bb847ec6a955621c31892c8014f70e019`.

The connector's metadata view did not expose provider checksum fields, even when
requested. **Current remote file contents have not been rehashed in this audit**;
the matching above combines current metadata sizes with verified local archives
and prior transfer receipts. Before an agreed deletion, recheck the current
remote content hashes and exact file IDs/membership. Any missing, new or different
file stops that target; never weaken a check to make cleanup pass.

## Preserve on Drive

- Three selected epoch-03 models, their training metadata and tokenizer files.
- Shared initialization and its tokenizer/metadata.
- Clean train/validation cache and `validation-inputs-noise-v1.zip`.
- Shared GPU reference/calibration results, readiness and preparation receipts,
  source/helper records and live notebooks.
- The small validation-grid and held-out results ZIPs and archive receipts.
- Four unarchived zero-byte locks; any file absent from the exact candidate list.
- Every local model/data/result copy, archive, raw dataset and frozen snapshot.

If the complete proposed scope is later removed, listed live project-file sizes
would be **1,933,792,135 bytes / 1.934 GB** before any later changes. Future runs
needing removed views/transfer bundles must restore the exact existing local
archives; do not regenerate noise, modify frozen configs or repeat experiments.

The connector's deletion operation is permanent. **No deletion tool was called**.
No permanent-deletion automation or account-wide Trash operation was created.
Choosing a cleanup scope does not authorize emptying Trash or permanently deleting
other files. The actual operation and its outcome must be recorded separately.

## Files and reproduction

- `scripts/audit_drive_storage.py`: offline inventory/backup matching and local
  input archiving only; it contains no Drive mutation or deletion operation.
- `artifacts/storage-audit/drive-2026-10-23-v1/drive-inventory.json`: complete
  current metadata tree; `project-files.csv`: every file's classification.
- `cleanup-candidates.json`: exact candidate scope, backup hashes and explicit
  pending remote-content/scope checks.
- `local-backup-matching.json`, `protected-model-presence.json`,
  `augmented-input-archive-verification.json`: actual verification evidence.
- `inventory.json`, `files-changed.md`: complete local outputs and project edits.

The first matching attempt stopped at an unarchived transient lock; its log is
retained. The revision explicitly preserves all unarchived files and completed
successfully. No completed experiment was rerun. AST, Ruff and whitespace checks
are recorded separately; no test suite was requested, added or run.
