# Case-loss training ablation: preparation

Prepared **2026-10-22** under the minimum ablation already specified in
[training-v1](training-protocol.md#7-seeds-ablations-and-affordable-execution).
The completed [117-evaluation grid](robustness-grid-results.md) is preserved;
its noisy scores do not select a new learning rate, mixture or checkpoint rule.

## Fixed comparison

Omit **case_loss** from augmented training. Retain the original typo swaps and
diacritic-loss word plans, the same epoch row masks and source-row permutations,
seed 586, exact shared initial tensors and all numerical hyperparameters.
The surviving operation order is `typo_swap → diacritic_loss`. The omitted
budget is not redistributed. This estimates removal of a training component
with lower total noise exposure, not an equal-dose intrinsic importance score.

Use the unchanged `configs/training-augmented.json` and existing
`train.py --arm augmented --without-family case_loss`. The new
`configs/ablation-case-loss.json` freezes reference-run identities, paths,
the initialization and five planned post-training validation conditions.
There are three full epochs / 10,968 optimizer updates, physical batch 8,
accumulation 4, LR 2e-5, FP32/eager and clean-validation checkpoint selection.
Retain final-epoch comparison if selected epochs differ across arms.

The clean and augmented runs remain fixed references. No held-out test is used.
Only one training seed is planned in this ablation. Dataset licensing and exact
source/split provenance are still open, as recorded in the project README.

## Prepared local views

The existing view preparer generated the three omitted-family views locally,
then reloaded and re-tokenized every saved word view against its original labels.
The additional paired audit verified identical selected row masks, full-split
retained word masks, and the positions/parameters of all retained operations.
No model/optimizer was created during view preparation.

| Epoch | Selected row views | Changed words | Changed words (%) | Original augmented (%) | Max subwords |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 58,484 | 70,491 | 4.8153 | 7.1204 | 227 |
| 2 | 58,484 | 71,054 | 4.8538 | 7.1817 | 234 |
| 3 | 58,484 | 70,856 | 4.8402 | 7.1510 | 229 |

Every epoch retains 116,969 records / 1,463,897 word targets. Row-view selection
is 50%; it is not the changed-word percentage. Selected views with no actual
edit are retained. Maximum length is 234, below 512; omitting lowercasing can
still increase subword counts for some inputs. A separate calibration must
include the longest prepared batch rather than assume the old peak is a bound.

Preparation identity:
`c51d595912e144ab4f65233311299067590974077954504193f5abbf249a3a33`.
The verified input ZIP is **42,800,098 bytes**, 31 files, SHA-256
`e0fe9e767406c75702f112c068a404276c4d41661231e3376e2862654dfdca53`.
It contains the existing local training views and their audits/snapshots, no
weights or held-out test text. Colab restores these bytes instead of regenerating
noise on Drive. `configs/ablation-case-loss-inputs.json` records the transfer
hash and serialization-independent epoch hashes.

## Execution and readiness

`notebooks/05_case_loss_ablation_colab.ipynb` separates setup, input restoration,
30-update calibration, readiness, formal training, status and archive creation.
All calibration/training/archive switches default to **False**.

- `scripts/prepare_ablation.py audit` compares the saved view pair.
- `package` builds and verifies the input ZIP; `restore` checks transfer/member
  hashes, safely restores new views and repeats the paired audit.
- `ready` checks the separate calibration, finite event losses, all training
  source/config hashes, both reference runs, shared initial files and current
  GPU environment. It requires formal training to remain unstarted.
- The original training runner independently checks/re-tokenizes all epoch views
  and word targets before optimizer creation. Its training and inference
  implementations are unchanged in this preparation.

The new calibration is a short technical run with **30 optimizer updates**.
It saves timing/memory/events, not trained weights or model-quality scores.
The formal ablation starts again from the shared initialization. A passed
calibration is not a completed ablation experiment.

Formal run path: `training-v1/train-augmented-without-case-loss-seed-586-v1`.
Calibration path: `training-v1/calibration-augmented-without-case-loss-seed-586-v1`.
Use two fixed optimizer slots and preserve all three candidate epoch exports.
After successful formal training, follow the existing manual archive download,
local model/metric verification and explicitly scoped cleanup workflow. No
automatic or account-wide Drive deletion is added.

## Current status

Local view preparation, complete re-tokenization, paired masks/operations audit,
ZIP/member hash verification and notebook/config preparation passed.
Drive transfer and GPU calibration are pending. Formal ablation training has
**not started**. No ablation performance result exists yet.

Operational receipts are outside Git under
`artifacts/training/training-v1/ablation-without-case-loss-preparation-seed-586/`.
