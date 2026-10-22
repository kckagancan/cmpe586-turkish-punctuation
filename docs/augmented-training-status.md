# Augmented training — first formal run

## Current state — 2026-10-22

Formal augmented training has started on Colab **2026.07 / Tesla T4**. It is
**in progress**, not a completed model-quality result. The user requested checking
the initial updates/checkpoint and then leaving the run to continue visibly.

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

Initial checks are complete and control is handed to the user while training
continues. No recurring monitor was created. The visible log, screenshot and
`launch-observation.json` are saved outside Git under
`artifacts/training/training-v1/train-augmented-seed-586-v1/launch-2026-10-22/`.

## After completion

Verify `summary.json`, epoch histories/model exports and matched inputs before
reporting a completed result. Follow the agreed storage process: build the
model/results archive, let the user download it manually, verify its local files
and model loading, then clean only identified obsolete Drive copies. No Drive
cleanup or automatic long-running monitoring was started with this launch.
