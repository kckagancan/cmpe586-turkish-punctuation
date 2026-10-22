# Current project state

Updated **2026-10-22**. Read this first, then only the relevant linked details.
This document tracks current work; it does not authorize starting a new experiment.

## Objective and confirmed method

**Robustness of Turkish Punctuation Restoration under Informal Writing Noise**
(CMPE 586). Compare a published punctuation BERT, clean BERTurk fine-tuning and
the same initialized BERTurk with synthetic-noise augmentation. Labels: O,
PERIOD, COMMA, QUESTION_MARK; first-subword supervision. Preserve word targets.

Training-v1 uses three epochs, 10,968 updates, batch 8 × accumulation 4, LR 2e-5,
FP32/eager, seed 586; select by clean validation macro-F1. Noise-v1 has case loss,
Turkish diacritic loss and internal adjacent typo swaps, 2/5/10% component word
budgets. Combined coverage is larger. Corruption seeds are not training seeds.

## Completed evidence

- Clean and augmented seed-586 training completed; both selected epoch 3.
  Their complete model/result archives were locally verified. The retained Drive
  exports and shared initialization remain; disposable copies were cleaned with
  scoped approval. [Clean](docs/clean-training-status.md),
  [augmented](docs/augmented-training-status.md).
- Matched GPU validation grid completed: **3 × 39 = 117 evaluations**; raw-target
  metrics, local archive hashes and seed aggregates verified. Clean macro-F1:
  clean model 84.0365%, augmented 83.9881%. Combined-heavy mean F1:
  clean 76.2023%, augmented 80.0646% (**+3.8623 pp**).
  [Full report](docs/robustness-grid-results.md).
- One training seed only; held-out test and the training ablation remain unrun.
  Dataset license and exact source/split provenance remain unresolved.

## Active outcome: case-loss ablation preparation

User asked to finish preparation and **stop before formal ablation training**;
they want to make a separate request before starting the study.

- Three local omitted-family epoch views are prepared, re-tokenized and audited.
  Selected rows and retained word masks/operations match original augmentation.
  No budget redistribution; changed words 4.8153 / 4.8538 / 4.8402%, versus
  7.1204 / 7.1817 / 7.1510% originally. Maximum length 234 subwords.
- Plan: `configs/ablation-case-loss.json`; unchanged numerical config:
  `configs/training-augmented.json`. Input transfer:
  `configs/ablation-case-loss-inputs.json` (42.8 MB, 31 files).
- Orchestration: `notebooks/05_case_loss_ablation_colab.ipynb`;
  `scripts/prepare_ablation.py` and `src/turkish_punctuation/ablation.py`.
  [Preparation details](docs/case-loss-ablation-preparation.md).
- **Still pending:** Drive ZIP transfer/restore, a separate 30-update T4
  calibration whose weights are discarded, and final readiness verification.
  No formal ablation output or performance result exists yet.
- Current working Colab:
  <https://colab.research.google.com/drive/1KKuqXRHKn4xr0nA7mXjtZexL4FrKFsV8>.
  Inspect live runtime state before using it; prior evaluation finished.

## Locations and next action

- Local ablation views:
  `data/processed/training-augmentation-v1/seed-586-without-case_loss/`.
- Local prep receipts/ZIP:
  `artifacts/training/training-v1/ablation-without-case-loss-preparation-seed-586/`.
- Drive input destination: `inputs/training-augmentation-v1/seed-586-without-case_loss`.
- Planned calibration: `training-v1/calibration-augmented-without-case-loss-seed-586-v1`.
- Planned formal output: `training-v1/train-augmented-without-case-loss-seed-586-v1`.

Finish the pending preparation, update this status with actual receipts, and
leave formal training/calibration/archive switches off for handoff. Do not
start the three-epoch run until the user explicitly asks.
