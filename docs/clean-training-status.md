# Clean training status

## Completion verified: 2026-10-19

**The three-epoch clean baseline has completed.** `summary.json` and
`progress.json` report **10,968 optimizer updates** and **best epoch 3**.
Completion time: **2026-10-19 04:40:52 Europe/Istanbul**.

Run identity:
`5ddecc9b8bfaf0b9442a7942caa45a64173e037b0ea7309692c12e5767603f37`.

The measured loop took **72.1855 minutes**, including validation, checkpoint
writes and model exports; initial input/model loading is excluded. Time inside
checkpoint calls was 5.0936 minutes. The training cell's wall time also includes
its initial loading/checks.

| Epoch | Train word loss | Clean validation word loss | Punctuation macro-F1 (%) |
| --- | ---: | ---: | ---: |
| 1 | 0.188665 | 0.130181 | 83.2543 |
| 2 | 0.121353 | 0.129169 | 83.4822 |
| 3 | 0.107470 | 0.131499 | **84.0365** |

Selection follows the prespecified clean punctuation macro-F1 rule. Epoch 2's
slightly lower validation cross-entropy does not change that selection.

Best-epoch punctuation macro P/R/F1 is **86.0306 / 82.4744 / 84.0365%**;
punctuation micro P/R/F1 is **85.6606 / 80.4760 / 82.9874%**.

| Label | Precision (%) | Recall (%) | F1 (%) | Support |
| --- | ---: | ---: | ---: | ---: |
| PERIOD | 94.6335 | 97.3512 | 95.9731 | 15,252 |
| COMMA | 70.3312 | 57.6593 | 63.3680 | 11,894 |
| QUESTION_MARK | 93.1271 | 92.4126 | 92.7685 | 1,173 |

All epochs evaluated the same 179,164 validation words with fixed class supports.
Against the published reference's clean validation macro-F1 83.6366%, this run's
selected checkpoint is higher by **0.39995 percentage points**. This is one
training seed and a descriptive validation comparison; no significance or noisy
robustness claim follows from that difference.

The selected model is `models/epoch-03` under the Drive run directory below.
After verification, the active Colab training switch was returned to `False`;
the enhanced completion summary remains visible. The runtime was left connected
for model preservation.
Original numeric JSON from the visible notebook was preserved and metadata
copies saved under ignored
`artifacts/training/training-v1/train-clean-seed-586-v1/`:
`metadata-bundle.json`, `summary.json`, `run.json` and `progress.json`.
The run identity was recomputed and matched; recorded source/config hashes match
the local sources. Update counts, epoch ordering/selection and class supports
also agree. Before publishing its summary, the runner checked epoch model-file
and validation-report hashes. The selected weights were subsequently downloaded
and independently verified locally; see [backup verification](backup-verification.md).
Checkpoint recovery with subsequent optimizer updates remains unexercised.
The subsequent Colab audit loaded both retained checkpoint states and verified
all epoch exports. Full validation replay reproduced all 179,164 word predictions
and metrics exactly, with no training updates or test access.

### Storage issue observed after completion

At approximately 04:47 Europe/Istanbul, Colab displayed **Google Drive storage
quota has been exceeded**. A project-scoped check reported total 16,106,127,360
bytes, used 15,403,937,792 bytes and free 702,189,568 bytes. Active checkpoint
states were `checkpoint-000027/state.pt` (1,321,522,291 bytes) and
`checkpoint-000028/state.pt` (1,320,587,955 bytes). No personal files or Drive
Trash contents were inspected or deleted. The cause of the quota pressure was
not established. See the [official Colab storage-quota FAQ](https://research.google.com/colaboratory/faq.html#drive-full).

Training completion and successful reads from the mounted Drive do not establish
that all pending cloud uploads have completed. A later flush/unmount attempt
returned an error and left the mount disconnected. Reconnecting succeeded without
new authorization, and all epoch/checkpoint hashes passed again. The refreshed
mount reported only about 0.16 GB free with another quota warning; Google One
independently showed 98% used. Cloud-upload completion is not confirmed. The
runtime was not reset. Before another large run, confirm adequate actual free
space; selected-model preservation is already verified locally below.

The selected epoch-03 model's six files were verified against `export.json`,
copied to a temporary recovery folder and verified again. Together with run
metadata, events, sources and validation outputs, an uncompressed archive was
created at `/content/cmpe586-recovery/clean-seed586-5ddecc9b8bfa.zip`:

- Size: 448,623,671 bytes.
- Archive SHA-256: `9a45deb24d8db5e7cf6c0cf99d81ea22c6a0c8119d39e56f5786a35ee680e6d5`.
- Selected model tensor SHA-256: `487a8c7c38d05787f8527eca642b4d2a80bb94a814b960479b9668a87492cd96`.

The in-app download initially produced no confirmed local file. The user then
downloaded the ZIP through a normal browser into Downloads. The original ZIP's
size/SHA-256, all model-file hashes, independent tensor digest and all three
epochs' metrics recomputed from predictions passed verification. A copy and
extracted selected model now reside under the ignored project artifact directory.
**The selected-model local backup is confirmed.** A subsequent 1.77 GB archive
also preserves all three epoch models, shared initialization and run results.
All 49 source files and four offline model/tokenizer reloads passed verification.
After explicit user confirmation, the two optimizer states, epoch-01/02 Drive
exports and temporary transfer ZIP were permanently deleted. The selected
epoch-03 model, run reports and shared inputs/initialization remain on Drive;
the UI now shows 5 GB of 15 GB used (about 10 GB free). Exact optimizer-state
continuation is deliberately no longer preserved for this completed run.
Full scope and limits are in [backup verification](backup-verification.md).

**Next planned step after storage preservation:** evaluate the selected clean checkpoint on a matched clean
control and the same four medium/seed-586 validation noise conditions as the
published-reference pilot. Then implement/audit the planned augmentation arm and
train it from the identical shared initialization. No noisy evaluation of this
fine-tuned model, augmented training or held-out test evaluation was started as
part of this completion check.

## Launch observed: 2026-10-19

The user authorized starting formal clean training after persistent preparation.
The Colab training cell started at approximately **03:27 Europe/Istanbul** using
repository commit `7351f9f651d7cba02b3846c7e08df875ee4dd532` and a Tesla T4.
The GPU PyTorch version guard passed; train/validation and initialization checks
passed again in the new runtime. Drive was mounted after the user completed
the authorization flow.

| Setting | Value |
| --- | --- |
| Arm / seed | clean / 586 |
| Epochs / planned optimizer updates | 3 / 10,968 |
| Batch / gradient accumulation | 8 / 4, effective batch 32 |
| Precision / learning rate | FP32, TF32 disabled / 2e-5 with warmup and linear decay |
| Training records / words | 116,969 / 1,463,897 |
| Validation records / words | 14,453 / 179,164 |
| Selection | Clean validation punctuation macro-F1 at completed epoch ends |

Shared initial tensor SHA-256:
`f26849ec8c687a095ec702cd81c3522e4e0e58a4ed52c4aeff88b0d6715cbf5e`.
The earlier calibration's updated weights are not reused. Test data is not
opened by this runner.

Run outputs are under the designated persistent project directory:

```text
MyDrive/cmpe586-turkish-punctuation/training-v1/train-clean-seed-586-v1/
```

## Initial observation

The first 525 optimizer updates completed without an error. Observed individual
accumulation-window losses were 1.4371 at step 25, 0.6851 at 100, 0.4648 at 125,
0.1941 at 500 and 0.1609 at 525. These are training losses on different windows,
not validation metrics or evidence of final model quality.

The runner saves a starting checkpoint and then checkpoints every 500 optimizer
updates and at epoch boundaries. The step-525 log was observed after the runner's
synchronous step-500 checkpoint returned successfully, so the initial and first
periodic checkpoint writes have been exercised. No separate file-browser read
or resume attempt was performed at that handoff. Full-epoch validation/selection
and recovery from an interruption were still unobserved then. The completion
verification above supersedes this initial observation.

## Notebook handoff

The training cell streams the CLI's output live and forwards notebook interrupts.
The Colab copy enables the training switch for this authorized run; the source
notebook in Git retains `RUN_FULL_TRAINING=False` as its default.
The following summary cell is queued behind training and prints
**CLEAN TRAINING COMPLETE**, completed epoch
and update counts, best epoch, clean validation macro-F1 and model location when
`summary.json` is present. The summary is only published after the runner's final
artifact checks pass.

At the initial handoff, the user watched the visible notebook and returned after
execution ended. No scheduled monitoring or automatic follow-up was configured.
