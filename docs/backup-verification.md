# Clean baseline backup verification

Verified on **2026-10-19**, after the user downloaded the recovery ZIP through
a normal browser. This audit performs **zero training updates** and does not
open the held-out test split.

## Durable local backup

The downloaded archive is 448,623,671 bytes. Its SHA-256 exactly matches the
archive created and checked in Colab:

`9a45deb24d8db5e7cf6c0cf99d81ea22c6a0c8119d39e56f5786a35ee680e6d5`

It was copied into the ignored project artifact directory and safely extracted:

```text
artifacts/training/training-v1/train-clean-seed-586-v1/
  clean-seed586-5ddecc9b8bfa.zip
  recovery/
    models/epoch-03/
    validation/epoch-{01,02,03}/
    source_snapshot/
    run.json
    summary.json
    progress.json
    events.jsonl
  verify_backup.py
  backup-verification.json
  backup-verification.log
  regression-tests.log
  verify_colab_run.py
  colab-verification.json
```

The archive contains the selected/final epoch-03 model, tokenizer, metadata,
all three epochs' validation predictions/metrics, event log and source snapshot.
The epoch-01/02 model weights and full optimizer checkpoints are not in this ZIP.

## Local checks passed

| Check | Result |
| --- | --- |
| ZIP file size, SHA-256, CRC, duplicate/unsafe paths | Passed; 28 archive entries |
| Run identity, configuration and archived/current source hashes | Match |
| Three epoch counters and best-epoch selection | 3 epochs / 10,968 updates; best epoch 3 |
| Model/tokenizer file hashes | All 6 match the export manifest |
| Independent hash of actual model tensors | `487a8c7c38d05787f8527eca642b4d2a80bb94a814b960479b9668a87492cd96` |
| All model tensors finite | Passed |
| Offline model load | No missing, unexpected or mismatched weights |
| All 10,968 event records | Step/epoch/row/word counters, finite loss/gradient/timings and learning rates valid |
| Word-weighted training losses | Recomputed from events; all epochs match |
| All validation targets | Match pinned raw validation tags by unique source row ID |
| All epoch validation predictions | 14,453 unique records / 179,164 words per epoch, expected ordering and supports |
| Recomputed validation metrics | All three epochs' P/R/F1, confusion matrices and diagnostics match saved reports |
| Offline CPU model inference | 31 records / 806 word predictions, all identical to saved GPU predictions |
| Existing offline regression suite | 30 tests passed |

The CPU sample combines short, long and length-spread records. It uses local
PyTorch 2.14.1+cpu; the recorded training environment used 2.11.0+cu128. This is a
reload/inference check, not a repeat of training or a separate model-quality run.

## Local preprocessing provenance

The local manifest hash differs from the Colab training manifest. The local
preparation contains train/validation/test split metadata and records local
PyTorch/datasets versions; Colab prepared train/validation with its own versions.
The validation Arrow data and dataset-info hashes match exactly, while
`state.json` and the overall manifest differ.

The audit therefore checked the pinned raw validation hash and re-tokenized
**all 14,453 validation records using the archived model's tokenizer**. Every
saved input/alignment column matched. No prepared files or manifests were
overwritten. A local three-split preparation must not be substituted byte-for-byte
for the recorded Colab training cache when resuming an exact run.

## Test-helper repair

The first regression-suite invocation exposed three errors: the preparation
fixture passed the unittest runner's command-line arguments into the production
CLI parser. `tests/test_pipeline.py` now isolates that helper's `sys.argv` to
`["prepare_data.py"]`. All 30 existing tests then passed. Training, preprocessing,
metric implementations and the completed run's configuration were not changed.

## Colab checks passed

The recorded T4/package environment matched the original run. All six files in
each of the three epoch exports matched their saved hashes. Both retained
optimizer checkpoints passed file hashing and restricted CPU loading; model
tensors, optimizer steps/moments, scheduler/cursor, row permutation and saved
Python/NumPy/Torch/CUDA RNG states were consistent. Both were at update 10,968;
checkpoint 28 records three completed epochs.

The selected model then repeated **the complete clean validation inference**
using the original prepared inputs and recorded GPU settings. All **14,453
records / 179,164 word predictions** and all metrics matched. The predictions
file itself was byte-identical, with SHA-256:

`4eb360cc7751dab9cba7cf721acff52fd7f9d1fb668581b3e5885c75546c3861`

Replay took 37.46 seconds. This was inference only: no optimizer updates and no
held-out test access. Loading checkpoint state does not establish that a later
training update after interruption has been exercised.

## Storage budget and archive policy

The observed FP32 retention is about **4 GB per completed arm**: two resumable
optimizer states total 2.642 GB, and three epoch model exports about 1.32 GB.
Shared initialization adds about 0.44 GB per seed; input caches, predictions,
logs and temporary write/archive space need additional headroom. Figures use
decimal GB and are estimates for the current architecture/settings.

If approximately 12–13 GB is actually free, this is adequate headroom for the next
arm at the observed sizes, but is not a guarantee for every planned seed/ablation.
Three retained arms alone use about 12 GB, before shared assets and transient
writes. The local project filesystem currently has about 59.92 GB free.

Keep active-run resumable checkpoints on Drive. After each completed run, make
a local archive, check its hash and contents, then consider removing the
specific Drive copies included in that verified archive. Keep all epoch models,
predictions, metadata and sources to audit selection; include full checkpoint
states if exact training continuation is to remain possible. The existing
449 MB recovery ZIP is sufficient for selected-model inference and result
inspection, but **is not a complete archive of all run weights/checkpoints**.
Do not use it as justification to delete unarchived epoch/checkpoint files.

Before the flush attempt, the mounted quota read 11,179,663,360 free bytes.
A subsequent `drive.flush_and_unmount(timeout_ms=60_000)` returned a failure;
the following check found Drive unmounted. Reconnecting succeeded without a new
permission prompt. All three epoch exports and both checkpoint hashes passed
again after reconnecting, but the refreshed mount reported only about **0.16 GB
free** and Colab displayed another quota warning. Google One independently showed
98% used, consistent with the refreshed low-space reading rather than the earlier
mount estimate or the user's reported 12–13 GB free. The reason for that
discrepancy is not established. Trash contents/personal files were not inspected.
The user confirmed permanent deletion from Trash; a page refresh still showed
the same 98% usage. Delayed quota propagation is possible according to the
[official Colab FAQ](https://research.google.com/colaboratory/faq.html#drive-full),
but has not been established as the cause here. The user is checking the account's
storage screen before further large runs.

Cloud-upload completion is therefore not confirmed. The downloaded local
selected-model backup remains verified. The notebook was restored to its normal
completed-run summary with full training disabled. No project model/checkpoint
files or personal files were deleted by this audit. Confirm adequate actual
free space before another large run.

## Remaining scope

Drive quota and cloud-upload completion remain separate from local backup
integrity. A local model copy cannot confirm every Drive upload has completed.
Actual optimizer updates after recovering an interrupted run have not been
performed. The verification logs and scripts are stored with ignored artifacts;
model weights are not committed to Git.

## Completed-run model archive — locally verified 2026-10-19

The user requested moving the completed clean run's necessary artifacts locally.
Optimizer states are not needed for the planned inference/evaluation; both final
`state.pt` files are deliberately excluded from the archival payload. Removing
those states later would give up exact optimizer/RNG continuation, without
removing the trained model or its evaluation results.

`scripts/archive_training.py` created a package after verifying source manifests:

- Name: `clean-seed586-models-5ddecc9b8bfa.zip`.
- Size: **1,772,093,748 bytes** (approximately 1.77 decimal GB / 1.65 GiB).
- SHA-256: `e9483c84e3e43bd57a1c8cf88f8a4bbf813b2159759cd8e8a3ba9ceb741023a4`.
- Contents: 49 source files plus `archive-manifest.json`.
- `run/models/epoch-{01,02,03}/`: all three model exports, each with its own
  configuration/tokenizer/manifest.
- `initialization/`: the exact shared initialization, configuration and tokenizer.
- `run/validation/`, `run/source_snapshot/`, run/summary/progress metadata and events.
- No optimizer checkpoints and no raw dataset or held-out test access.

The package was built on the Colab VM. Because the in-app Colab download route
did not previously save a file, a temporary verified copy was made under the
project's Drive `archives/` folder to use Drive's browser download route. A small
`summary.json` download succeeded and matched the original file byte-for-byte.
The large in-app download could not be confirmed; the user downloaded the ZIP
through a normal browser instead. The original Downloads file matched the source
size and SHA-256. It was copied into the ignored run artifact directory and
safely extracted under `completed-archive/`:

```text
artifacts/training/training-v1/train-clean-seed-586-v1/
  clean-seed586-models-5ddecc9b8bfa.zip
  completed-archive/
    initialization/
    run/models/epoch-{01,02,03}/
    run/validation/
    run/source_snapshot/
    run/{run.json,summary.json,progress.json,events.jsonl}
    archive-manifest.json
  models-archive-receipt.json
  models-archive-verification.json
  models-archive-model-load.json
```

All 49 source-file hashes/sizes and ZIP CRC/path/entry checks passed. Extracted
run identity, completion counters, best-epoch selection, all epoch export and
validation hashes and shared-initialization manifests passed revalidation.
All four models then loaded offline with no missing/unexpected/mismatched weights;
their tensor digests matched, all tensors were finite, label mappings matched
the project and all tokenizers loaded correctly. No optimizer update or held-out
test access was performed. **The full intended model/results archive is now
confirmed locally.** The earlier selected-model recovery backup also remains.

The raw dataset is intentionally absent from this ZIP; it is separately present
in local `data/raw/kurt_cayir_2023/`. Full project code is retained in Git/GitHub;
the ZIP also includes the run's training-source snapshots and environment record.
Models/results plus those shared data/code assets are the preservation scope;
the archive is not an exact optimizer-resume backup.

The user prefers one manual ZIP download after each successful training. Prepare,
download, verify locally, then clean only identified run files. Proposed cleanup
for this run: both optimizer checkpoints, epoch-01/02 Drive model exports and the
temporary transfer ZIP. Keep the selected epoch-03 model ready for the upcoming
validation, alongside shared initialization/input assets and small reports.
**No Drive cleanup has yet been performed.**

The visible Drive storage summary now shows 5.41 GB of 15 GB used, consistent with
the user's report of approximately 9.6 GB free before the temporary archive copy.
This supersedes the earlier low-space observation for that point in time;
the temporary archive itself consumes additional Drive space until cleaned up.
