# Augmented training — first formal run

## Completion verified — 2026-10-22

The first formal augmented run completed **3 epochs / 10,968 optimizer updates**.
`run.json`, `summary.json`, `progress.json`, all event counters, source snapshots,
shared initialization and the matched clean recipe/runtime passed checks.
The event steps are contiguous from 1 through 10,968 and all recorded losses/LRs
are finite. All three validation reports contain 179,164 word targets.

| Epoch | Updates | Train word loss | Clean validation macro-F1 | Micro-F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3,656 | 0.197345 | 0.833312993 | 0.825531760 |
| 2 | 7,312 | 0.131136 | 0.833883254 | 0.827567037 |
| 3 | 10,968 | 0.119515 | 0.839880566 | 0.830789938 |

Selected and final model: **epoch 3**. Clean validation punctuation macro-F1 is
**83.9881%**, versus clean training's **84.0365%** (difference **−0.0485 percentage
points**). This small clean-validation difference is descriptive; augmentation's
robustness contribution has not yet been evaluated on noisy inputs.
No held-out test evaluation was performed.

The loop including validation/checkpoint/export work took **82.8746 minutes**;
checkpoint writes accounted for **9.6889 minutes**. Initial data/model checks are
outside this loop time. Completion timestamp is **2026-10-22 05:37:38 Europe/Istanbul**.
The user's later disconnect notice does not establish the exact disconnect cause
or timing; the persisted run completion and model/results were verified afterward.

Run identity:
`3cdf62cc3be4bbd21e48dc17aee7b450f90e6b1e2b6041dea185e3d8f0ceb01e`.
There are two complete checkpoint manifests, serials **27 / 28**, with state files
of 1,321,522,291 / 1,320,587,955 bytes. These optimizer-state payloads were not
reloaded/rehashed in this completion check; exact optimizer resume remains
unexercised and is not needed for the completed-run inference archive.

## Model/results archive — local verification passed

All three epoch exports, their validation outputs and shared initialization
passed the archive builder's manifest/hash checks. The VM ZIP's entries/hashes
were verified, then its Drive copy passed a full SHA-256/size readback check.

- File: `augmented-seed586-models-3cdf62cc3be4.zip`.
- Size: **1,772,193,862 bytes** (about 1.77 GB decimal / 1.65 GiB).
- SHA-256: `b93de2fd2c0335c487332fcdb2d8ab10e9aaaf3eba11c9e43d93a7b5cfb44378`.
- Contents: **58 source members**, plus `archive-manifest.json`; all three models,
  tokenizers/configurations, initialization, validation predictions/metrics,
  event/run metadata, training/noise source snapshots and augmentation audits.
- Optimizer checkpoint payloads are excluded, as agreed for completed runs.
- Drive folder: [project archives](https://drive.google.com/drive/u/0/folders/1rp2-j-2Hk-nwzgCTIq0hx2yX7TujQyGU).
- Status: the user's manual download, the project ZIP copy, all member hashes,
  safe extraction and all four offline model loads **passed on 2026-10-22**.
- Local ZIP:
  `artifacts/training/training-v1/train-augmented-seed-586-v1/augmented-seed586-models-3cdf62cc3be4.zip`.
- Extracted copy: the same run directory's `completed-archive/`, containing
  `initialization/` and `run/`. The original Downloads copy was retained.
- All four models loaded with no missing, unexpected or mismatched weights;
  tensor identities, finite weights, tokenizers and label mappings matched.
  All three epoch reports were recomputed from archived validation predictions:
  14,453 records / 179,164 words each, with every metric matching.
  No training, new model inference or held-out test evaluation was performed.

Drive showed **7.69 GB of 15 GB used** with the temporary archive present.
After local verification and action-time user confirmation, only these four
identified items were permanently deleted from Drive:

1. This run's `checkpoints/`, containing the two optimizer-state slots.
2. This run's `models/epoch-01/`.
3. This run's `models/epoch-02/`.
4. `archives/augmented-seed586-models-3cdf62cc3be4.zip`.

All four item IDs were matched before deletion and were absent from Trash
afterward. The selected epoch-03 model and the run's validation, source snapshots
and metadata remain; shared inputs/initialization and the small archive receipt
were preserved. Optimizer history was intentionally not backed up: this completed
run cannot be resumed with that optimizer state. Account-wide Trash was not emptied.
After the display updated, Drive showed **2.75 GB of 15 GB used**
(about **12.25 GB free**), down from 7.69 GB used before cleanup.

Completion receipts, 16 metadata files, audit/archive helper sources and screenshots
are saved outside Git under
`artifacts/training/training-v1/train-augmented-seed-586-v1/completion-2026-10-22/`.
This directory also contains `local-archive-verification.json`,
`local-model-verification.json`, `completion-verification.json`, the scoped
`drive-cleanup-plan.json` / `drive-cleanup-result.json` and before/after screenshots.
All transferred metadata hashes, the run identity and source snapshot hashes
also matched the local project. The active notebook's full training/calibration
switches were reset to false; the archive switch is also saved false.

## Launch and initial handoff — 2026-10-22

Formal augmented training was started on Colab **2026.07 / Tesla T4**. At the
initial handoff it was still in progress. The user requested checking the initial
updates/checkpoint and then leaving the run to continue visibly.

- Executed clean checkout: `de9345abb02a54dccea8e394172752a0f07af6d5`.
- Configuration: `configs/training-augmented.json`, training seed **586**,
  FP32/eager, batch **8 × accumulation 4**, **3 epochs / 10,968 updates**.
- Existing Drive views were reused; no noisy data or calibration was regenerated.
  Preparation identity:
  `cf0acd2af581f9e145b53ece0578e52bb847ec6a955621c31892c8014f70e019`.
- Shared initialization, prepared inputs and clean numerical recipe/runtime
  checks passed before optimizer creation. All three views are checked and
  re-tokenized by the formal runner, retaining original word targets.
- The initial checkpoint was written before the update loop. Progress was
  observed through step **825** without an error; visible losses were finite.
  The synchronous step-500 checkpoint write/readback-hash check completed and
  training continued beyond step 525. This exercises the first optimizer-state
  write, not an interrupted-run optimizer resume.
- The completion-summary cell is queued after training. Full training is enabled
  only in the active Drive notebook; both switches stay false in Git.

Working notebook:
[03_augmented_training_colab](https://colab.research.google.com/drive/1KKuqXRHKn4xr0nA7mXjtZexL4FrKFsV8).
Run outputs are under
`MyDrive/cmpe586-turkish-punctuation/training-v1/train-augmented-seed-586-v1/`.
The source checkout is not modified during the run.

The Drive project page showed **2.34 GB of 15 GB used** before launch. Preparation
performed zero optimizer updates. The previous separate 30-update calibration
is recorded in [augmentation-preparation.md](augmentation-preparation.md); its
weights are not reused. Formal-run timing, full-epoch validation/selection,
model-quality results and real checkpoint resume remain unobserved at launch.

Initial checks completed and control was handed to the user while training
continued. No recurring monitor was created. The visible log, screenshot and
`launch-observation.json` are saved outside Git under
`artifacts/training/training-v1/train-augmented-seed-586-v1/launch-2026-10-22/`.

## Local verification and cleanup policy

Verify `summary.json`, epoch histories/model exports and matched inputs before
reporting a completed result. Follow the agreed storage process: build the
model/results archive, let the user download it manually, verify its local files
and model loading, then clean only identified obsolete Drive copies. No Drive
cleanup or automatic long-running monitoring was started with this launch.
