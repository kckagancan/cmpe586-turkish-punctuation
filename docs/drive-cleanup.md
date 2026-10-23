# Verified scoped Drive cleanup — 2026-10-23

Completed at **17:06:47 Istanbul** after the user delegated deletion decisions
for locally backed or no-longer-needed project copies. The earlier
[inventory/proposal](drive-storage-audit.md) remains a dated pre-cleanup record.
No experiment was repeated.

## Outcome and retained assets

| Removed Drive copies | Files | Bytes |
| --- | ---: | ---: |
| Augmented training views | 31 | 221,310,174 |
| Without-case-loss training views | 31 | 209,588,210 |
| Archived validation, ablation and held-out evaluation outputs | 1,083 | 379,001,017 |
| Three temporary input/source transfer ZIPs | 3 | 115,853,780 |
| **Total** | **1,148** | **925,753,181** |

The connector permanently removed these verified Drive copies. All local data,
predictions, results, model archives and historical snapshots remain. Three
selected epoch-03 models, their metadata/tokenizers, shared initialization,
clean/validation inputs, shared reference/calibration, preparation/readiness,
notebooks, retained small result ZIPs and four unarchived zero-byte locks remain
on Drive. No account-wide Trash operation or scheduled deletion was used.

Fresh post-cleanup recursive listing: **111 folders / 292 files /
1,934,065,963 logical bytes**, with no listing errors or unlisted folders.
All **1,148 candidate IDs are absent**; all **291 original protected IDs** and
the newly created remote hash-readback receipt are present. Every pre-cleanup
file outside the candidate set remains at its expected path, parent and size,
allowing the live Colab notebook's own autosave size change.
These logical project sizes are **not an account-quota measurement**.

## Manifest mismatch resolved before deletion

The initial strict preflight stopped on
`inputs/training-augmentation-v1/seed-586/manifest.json`; no deletion occurred
at that stop. All other 1,147 remote content hashes already matched. The remote
manifest exactly matches the locally preserved October 19
`augmented-preparation-seed-586/colab-view-manifest.json` and its original
verification receipt:

- Local preparation manifest SHA:
  `de7456083abbf54e5f01ce6632fd1d77aefd359a790bf14067ee40bd6ef7c09a`.
- Exact Colab counterpart SHA:
  `09cbdf9bdc0e40c29e1aefd84d5fda2edf0e7fbf241f907d0ded9ca74c92c05d`.

Structural comparison found only preparation timestamps, elapsed seconds,
Python version, and the `datasets`/`pandas` versions differed. Every other
manifest field, including prepared input identities and epoch/source records,
is identical. Both original files remain untouched. A local supplemental ZIP
preserves the exact Colab manifest and historical/resolution evidence:

`artifacts/storage-audit/drive-2026-10-23-v1/cleanup-2026-10-23/augmented-colab-manifest-supplement.zip`

**9,705 bytes**, three source members plus archive manifest, SHA
`edd9e56f606285f2d2a5e5c420f267f0dd00bacce920c2a8425ec90ed0cb036a`.
Every member passed byte/hash readback. The original 44,255,470-byte local-input
ZIP is unchanged; restoring the original Drive tree requires the supplement's
manifest in addition to that ZIP's input contents.

The versioned `cleanup-candidates-v2.json` binds that one remote file to its
exact historical local counterpart and supplement. It preserves all original
1,148 Drive IDs, paths, parents and byte counts. Original candidate identity
`db40e93dce484d9dce363ec5994052836d142a515bc73ade52234b42a9e09368`
and original manifest remain unchanged; corrected identity:
`7ed5bdd0c1ad24eece5a59fe4ac3111159b7b152174edff26f2a333d4f544a4a`.
The verifier retains strict content hashes and rejects any scope change.

## Checks and execution receipts

- Mounted Drive content readback covered all 1,148 files / 925,753,181 bytes;
  local receipt SHA
  `47f8a3da880963999a76a6b1f16134ded9aa1b820954021c66efa2f9a6831e89`.
  Its remote receipt ID is `1kTe0FPIB_kezpTK4rAtHwlQyKVd9K5Wf` and is retained.
- Fresh preflight listed 426 folders / 1,440 files. Candidate IDs, paths,
  parents, sizes and modification times match the reviewed/hash-read copies.
  All 1,148 exact local file hashes and **eight backup ZIP hashes** matched.
- One small source ZIP was deleted first. Then 11 fully inventoried containers
  containing only approved candidate files and 14 individual files were removed
  sequentially: **26 acknowledged connector operations** in total. The four
  lock-bearing run parents were excluded from container deletion.
- One post-acknowledgement local journal write exceeded the process argument
  limit. The deleted held-out container was confirmed absent in its parent
  listing, and its 526 file receipts were saved in small chunks. That deletion
  was not retried. Final exact-ID/receipt/post-inventory checks all passed.
- All eight backup ZIP hashes still match after cleanup. Existing model/archive
  and offline-load receipts were reused; local selected models/shared
  initialization were checked for presence and expected size, without loading
  or rehashing weights.
- Ruff, AST parsing and whitespace checks are recorded with the final file
  inventory. No test suite, training, inference or noise regeneration ran.

All operational evidence is ignored under:
`artifacts/storage-audit/drive-2026-10-23-v1/cleanup-2026-10-23/`.
Key files: `scope-decision.json`, original and corrected preflight metadata,
`current-content-readback.json`, `manifest-resolution.json`, supplemental ZIP
verification, `cleanup-candidates-v2.json`, `ready-for-cleanup.json`,
`deletion-intent.json`, `container-operation-plan.json`,
`operation-results.json`, 1,148-line `deletion-receipts.jsonl`,
`after-metadata.json` and `cleanup-verification.json`. `inventory.json` and
`files-changed.md` provide the complete local handoff inventory.

The read-only verifier is [scripts/verify_drive_cleanup.py](../scripts/verify_drive_cleanup.py);
it contains no Drive deletion code. No reusable/scheduled deletion automation
was added. No further Drive cleanup is pending. Future work needing removed
copies must restore the existing exact local archives, rather than regenerate
inputs or rerun completed experiments.
