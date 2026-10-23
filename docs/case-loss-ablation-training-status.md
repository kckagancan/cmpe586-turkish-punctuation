# Case-loss ablation: formal training

## Completion verified — 2026-10-22

The formal run completed **3 epochs / 10,968 optimizer updates** at
**18:53:28 Europe/Istanbul** (15:53:28 UTC). The selected and final model is
**epoch 3**, selected only by clean-validation punctuation macro-F1:
**84.1310%**. Training loop time was **94.75 minutes**, including validation,
exports and checkpoints; checkpoint work accounted for 10.32 minutes. This
excludes startup input/model verification.

| Epoch | Update | Train word loss | Validation word loss | Clean macro-F1 (%) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3,656 | 0.191944 | 0.130628 | 83.1426 |
| 2 | 7,312 | 0.124982 | 0.128714 | 83.5030 |
| 3 | 10,968 | 0.113060 | 0.129739 | **84.1310** |

| Selected seed-586 model | Epoch | Clean validation macro-F1 (%) |
| --- | ---: | ---: |
| Clean fine-tuning | 3 | 84.0365 |
| Full augmentation | 3 | 83.9881 |
| Augmentation without case loss | 3 | **84.1310** |

The ablation's clean macro-F1 difference is **+0.0945 pp** versus clean
fine-tuning and **+0.1430 pp** versus full augmentation. Its noisy-validation
performance has not been measured yet. These small clean-score differences do
not establish significance or robustness gains. This is one training seed;
the omitted noise budget was not redistributed. All three selected epochs are
the final epoch, so selected/final comparisons currently coincide.

Completion checks passed on Colab and transferred metadata hashes/identity were
checked locally. The audit verified the complete event sequence, all 4,391,691
training word presentations, finite losses/gradients/LRs, fixed calibrated
specification, source snapshots, all three model export file hashes, shared
initialization file hashes and clean-validation selection. For every epoch,
all 14,453 saved rows / 179,164 word targets matched the original raw validation
targets and stable length-sorted evaluation order; every saved metric was
recomputed and matched. These checks perform no new training or model inference.

The first audit attempt incorrectly assumed source-row prediction order. It
stopped before publishing a receipt. The corrected audit reproduces the
runner's stable `(input_length, source_index)` order from hash-verified prepared
validation data and checks original targets by row ID. Neither prediction files
nor training source/configuration were changed. Both audit source versions and
the correction record remain in the ignored completion directory.

Both checkpoint manifests were complete (serials 27/28); state file presence and
sizes were checked. Optimizer checkpoint hashes/tensors were not reloaded in
this completion audit. Strict offline model loads subsequently passed locally
after the manual ZIP download, as recorded below. No test suite
was added or run, and held-out test text was not accessed.

## Archive and next action

The existing archive workflow produced a verified model/results ZIP containing
all three candidate epoch models, their validation predictions/reports,
source/config snapshots, augmentation audits/events and the shared initialization.
Optimizer resume states are intentionally omitted for this completed run.

- File: `ablation-without-case-loss-seed586-models-9a74ed0ee701.zip`.
- Size: **1,772,191,681 bytes** (1.77 GB); **58 source members** plus
  `archive-manifest.json`.
- SHA-256: `d95815cfa02ec880ff99fb9d0f09630821ab1199fec8fb86e2a6da730c6fe7b9`.
- Temporary Drive file ID: `1wYa8b4SXHS4yUI4pGYFldx_vXM4NddCY`;
  subsequently deleted after local verification, as recorded below.
- VM ZIP member hashes/sizes/paths and the complete Drive mount readback
  hash/size passed. Drive connector folder listing and file metadata also
  confirmed the remote ZIP's exact name, size and archives-folder parent.
  A provider-side SHA-256 was not returned by the connector.
- Local archive download/extraction and four offline model loads **passed**.
  The project ZIP and extracted `completed-archive/` preserve all epoch models.

After the archive's ZIP/member and Drive readback hashes pass, the user downloads
it manually. Verify the local archive and four offline model loads before the
agreed scoped cleanup: this run's two optimizer slots, epoch-01/02 Drive exports
and temporary transfer ZIP. Retain the selected epoch-03 Drive model, metadata,
inputs, shared initialization and all locally archived epoch models/results.
The scoped cleanup plan lists four targets totaling **5,296,606,031 bytes**
(the two optimizer slots are grouped under `checkpoints/`). This is the sum of
current file sizes, not a guarantee of immediate Drive quota reclamation.
The cleanup subsequently completed after local verification, as recorded below.
No account-wide Trash cleanup was performed.

Then run the prespecified clean + four medium/seed-586 noisy validation
comparison. Held-out test evaluation remains separate and requires its frozen
run matrix. Local completion evidence is under
`artifacts/training/training-v1/train-augmented-without-case-loss-seed-586-v1/completion-2026-10-22/`.
The clean curve containing three epochs for each of the three arms is saved in
[`case-loss-ablation-clean-validation.csv`](case-loss-ablation-clean-validation.csv).

## Local verification and scoped cleanup — 2026-10-22

The user's Downloads ZIP matched the source's **1,772,191,681 bytes** and SHA-256.
The shared archive verifier checked all 58 member hashes/sizes, duplicate/unsafe
paths, extracted bytes, completed-run manifests, shared initialization and all
epoch exports. A verified ZIP copy is retained at the local run artifact root;
the safely extracted models/results are in `completed-archive/`. The original
Downloads copy is retained too.

All four models (initialization + epochs 1/2/3) loaded offline with no missing,
unexpected or mismatched weights. Finite tensor digests, vocabulary/tokenizer
configuration, four-label mappings and sequence limits matched their manifests.
Each epoch's 14,453 rows / 179,164 references matched the original local raw
validation IDs/targets, and every recomputed metric matched the saved report.
No new inference, training or test-suite execution occurred. Receipts:
`local-archive-verification.json` and `local-model-verification.json` in the
ignored completion directory.

Following the user's explicit request to repeat the archive/verify/scoped-cleanup
workflow after all checks, the Google Drive connector removed only these scopes:

| Scope | Files deleted | Bytes |
| --- | ---: | ---: |
| Both optimizer slots under this run's `checkpoints/` | 4 | 2,642,110,754 |
| This run's `models/epoch-01/` | 7 | 441,151,798 |
| This run's `models/epoch-02/` | 7 | 441,151,798 |
| Temporary Drive transfer ZIP | 1 | 1,772,191,681 |
| **Total** | **19** | **5,296,606,031** |

The connector rejected the folder URL before deleting anything. The cleanup
therefore used the 19 individually inspected file IDs; empty folders remain.
Every file deletion returned success. Fresh listings confirmed all four
slot/epoch scopes empty, the ZIP absent from archives, all seven selected
epoch-03 files unchanged in ID/name/size and run metadata retained. No Trash
or account-wide cleanup was used; actual provider quota reduction was not read.
The exact identity/target listings, deletion responses and readback are in
`drive-cleanup-candidate.json`, `drive-cleanup-execution.json` and
`drive-cleanup-result.json`.

Selected epoch-03 and shared inputs/initialization stay on Drive. Every epoch
model and result stays in the verified local archive. Optimizer/RNG continuation
states were intentionally not archived, so this completed run cannot resume
from those exact optimizer states. Final local-verification and cleanup receipts
were also persisted under the project's ablation preparation folder on Drive.
The current Colab handoff is cell 30; all execution switches are off. Next is the
planned clean + four medium-noise validation comparison; no noisy ablation or
held-out test score is claimed yet.

## Historical launch record

The user explicitly authorized the formal run on **2026-10-22** after the
separate [preparation and calibration](case-loss-ablation-preparation.md).
This is the prespecified seed-586 augmented arm without `case_loss`; the
remaining noise budget is not redistributed. It measures removal of this
training component with lower total noise exposure, not equal-dose importance.

## Launch checks

- The live T4 VM was connected, Drive was mounted, no experiment process was
  active and the formal output directory did not exist.
- Colab source commit `2cf928fa8608066971dc7da05444685c619af6fd` was clean.
  The newer local `7501d31` commit only changes documentation; all 16 calibrated
  source/config hashes still match. No training code or frozen config changed.
- The current readiness gate passed again. The complete recorded environment
  equals the calibration environment: Python 3.12.13, Torch 2.11.0+cu128,
  Tesla T4, FP32/eager, deterministic execution and TF32 disabled.
- Paired row/retained-operation audits, reference-run metadata, shared
  initialization and prepared-file hashes passed. The previous 30-update
  calibration was reused; its weights were not reused.
- The live readiness receipt is
  `preparation/case-loss-ablation-v1/seed-586/readiness-launch-20261022T141257Z.json`
  under the project Drive root. SHA-256:
  `94c6810fda858eee10d1c65ca9c9cdfdfc62df65c397b6d6d8c4ebaaf95fe7f6`.
  It is byte-identical to the previously verified receipt.

The original runner performs the complete train/validation and three-view
re-tokenization/target checks again before optimizer creation. It uses the same
initial tensors, row permutations, batch 8 × accumulation 4, LR 2e-5 and three
epochs / 10,968 updates. Selection remains clean-validation punctuation macro-F1
with earliest-epoch tie-breaking; all three epoch exports and two fixed optimizer
slots are retained.

## Live location and handoff

The existing [Colab notebook](https://colab.research.google.com/drive/1KKuqXRHKn4xr0nA7mXjtZexL4FrKFsV8#scrollTo=oe3Izxyt9jJF)
still has the historical display name `03_augmented_training_colab.ipynb` and
old augmented-run outputs. The current ablation launcher is appended **cell 26**;
it uses `training_command(..., mode="train", resume=False)` from the ablation
module. Do not run the historical augmented training cell or Run all. The launch
switch is consumed before starting the subprocess; calibration/archive switches
remain off, and an existing-output guard prevents a second fresh launch.
After completion, cells 27–29 were added for auditing, the existing archive
workflow and a download/handoff card with all execution switches off.

Formal Drive output:
`training-v1/train-augmented-without-case-loss-seed-586-v1`.
Local launch evidence:
`artifacts/training/training-v1/train-augmented-without-case-loss-seed-586-v1/launch/`.

The subprocess started at **17:14:51 Europe/Istanbul** (14:14:51 UTC), PID
`16912`. At **17:19:00**, updates 1–25 were checked: consecutive steps, exactly
800 records, finite loss/gradient/LR values and an exact formal specification
match to calibration except `mode=train` and the removed calibration step limit.
The training runner reached these updates only after its input checks and
initial checkpoint write/readback passed. The training log subsequently reached
step 300 without an error; this is startup evidence, not an ablation quality score.

| Initial observation | Value |
| --- | ---: |
| First / update-25 word loss | 1.505801 / 1.461627 |
| Loss range across updates 1–25 | 1.421071–1.567117 |
| Gradient norm range before clipping | 13.607080–15.698046 |
| Formal run identity | `9a74ed0ee70110b75111f9e5a43e0cba1236cca8dafaee22eb3a371da34e6928` |

The local launch observation records the source/run/readiness hashes and update
summary. Its expected run identity was independently derived from the verified
local calibration specification and matched. The full formal `run.json`, event
stream and checkpoint tensors remain on Drive; they have not been downloaded as
a completed-run archive. The new readiness and observation receipts are separate
from the previous preparation/calibration records.

At the initial handoff, training was running and the user would report
completion; continuous monitoring was not requested. The completion section
above supersedes that historical running state. No test suite was added or run.

After completion, inspect `progress.json`, `summary.json`, all three validation
records and model exports. Follow the existing manual archive/download/local
verification workflow. The planned clean + four medium/seed-586 validation
conditions are recorded in `configs/ablation-case-loss.json`; they are not
automatically launched here. Held-out test evaluation and Drive cleanup have not
been performed. Report the single-training-seed limitation and unresolved data
license/split provenance.
