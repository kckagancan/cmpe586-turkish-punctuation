# Augmented training preparation

Prepared on **2026-10-19**. This step implements the existing training-v1 recipe;
it does not change noise severity, mixture, optimizer or model selection after
observing the clean-model pilot. Full augmented training has not started.

## Epoch views prepared locally

`scripts/prepare_augmentation.py` reads only the pinned train split, reproduces
the clean encodings, builds the full-split word plans and selects row views by
the prespecified label-free hashes. Exactly 58,484 of 116,969 rows are selected
for noisy views per epoch; selected rows with no edit are not resampled.

| Epoch | Noise seed | Selected noisy-view rows | Actually edited rows | Changed words | Changed words (%) | Max subwords |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 705861 | 58,484 | 46,625 | 104,235 | 7.1204 | 227 |
| 2 | 705862 | 58,484 | 46,770 | 105,132 | 7.1817 | 230 |
| 3 | 705863 | 58,484 | 46,732 | 104,683 | 7.1510 | 224 |

Every epoch retains **116,969 records / 1,463,897 supervised word targets** and
the original source IDs/order. All changes, protected characters, first-subword
targets and length limits passed audit. Saved views were then reloaded and every
encoding/target checked against re-tokenized saved words and original labels.
No model weights, optimizer, validation scores or test text were used.

The three views plus edit logs/examples/snapshots use **221,310,174 bytes** locally
under `data/processed/training-augmentation-v1/seed-586/` (outside Git).
Preparation identity:
`cf0acd2af581f9e145b53ece0578e52bb847ec6a955621c31892c8014f70e019`.

Serialization-independent encoded-content hashes:

| Epoch | SHA-256 |
| --- | --- |
| 1 | `6d82e08a7ad9aa633ee6c47f99c6345eac61c7ecbfb5307b44355fd2a4df650f` |
| 2 | `60f3c22ea6d25369a40a1378da49a74dfcc88b8231d912b882e2dbe1094aaae9` |
| 3 | `c1227f3153e7ef483efc242a6d18556c1befcb2125063a032d68e56c42fd6368` |

The Colab notebook requires the same content hashes even if Arrow file bytes
differ across package versions. Validation inputs remain clean and unchanged.

## Training integration

- `configs/training-augmented.json` copies the frozen numerical recipe and adds
  the view location, paired clean-run identity and two-slot storage policy.
  The historical clean configuration is retained.
- `scripts/train.py --arm augmented` verifies the completed clean run's numerical
  settings, budget, prepared inputs, initialization and required runtime fields.
  It rejects a mismatched pair before creating an optimizer.
- All three view files are verified/re-tokenized before updates. Training uses
  the appropriate epoch view and the same seeded source-row permutation as clean.
  One record presentation and one target per original word remain fixed.
- Resume binds the complete view manifest into run identity; a different noise
  realization/code/configuration cannot resume the run. Audits/manifests are also
  copied into the training run for the completed-run archive.
- The existing word-normalized loss, optimizer/scheduler, update loop and clean
  validation selection are reused. The optional `--without-family` preparation
  path keeps the same row masks and remaining family plans; it has not been run.

## Two fixed checkpoint slots

New augmented runs use `checkpoints/slot-01` and `slot-02`. A write stages the
payload on the VM, marks the other slot incomplete, overwrites its state in
place, verifies the readback hash and publishes its complete manifest last.
The last verified slot is protected throughout the write. The loader selects
the newest valid complete serial and can fall back to the other slot.

There is no checkpoint directory deletion in this layout. This replaces the
legacy create-and-delete rotation responsible for discarded checkpoint folders
accumulating in Drive Trash. The old layout is retained for historical code paths;
the new augmented configuration explicitly selects two reused slots.

`scripts/inspect_checkpoint_storage.py` exercises tiny writes/restricted loads
without a model or optimizer update. This checks filesystem behavior, not actual
model optimizer-resume or provider-side cloud durability/version retention.
The tiny probe passed both locally and on the mounted Drive project folder on
2026-10-19: four serial writes alternated between two slots, restricted loads
reproduced each payload, and each slot retained its inode. Drive ended with two
16,643-byte state files. No file was permanently deleted. This observes the
mounted filesystem; provider-side version retention, VM-loss durability and
actual model/optimizer resume remain unverified.
No account-wide Trash cleanup or permanent deletion is implemented here.

## Execution handoff

`notebooks/03_augmented_training_colab.ipynb` separates project/input preparation,
tiny storage inspection, 30-update GPU calibration and full training. Both
`RUN_CALIBRATION` and `RUN_FULL_TRAINING` default to `False`. Full training also
requires a matching calibration/source/view identity. Calibration starts from
the shared initialization, includes the longest prepared view and its weights
are not reused for the formal run.

```bash
python scripts/prepare_augmentation.py
python scripts/prepare_augmentation.py --verify-only
```

Colab uses `--output-directory` under the designated project input folder and
the existing persistent clean input/initialization assets. After a successful
formal run, the agreed manual ZIP download, local verification and scoped Drive
cleanup process remains in effect.

## Colab preparation and GPU calibration

Executed on **2026-10-19**, clean code checkout
`6497e82b226390bf9a4120d55c9e40befd0ff75e`. The working notebook is
[03_augmented_training_colab](https://colab.research.google.com/drive/1KKuqXRHKn4xr0nA7mXjtZexL4FrKFsV8).
Select Colab runtime **2026.07 / T4** for the matched arm. Python 3.12.13,
Torch 2.11.0+cu128, Transformers 4.57.6, datasets 4.0.0, numpy 2.0.2,
scikit-learn 1.6.1, tokenizers 0.22.2 and safetensors 0.8.0 matched the clean
training environment. Precision stayed FP32/eager, TF32 off, deterministic Torch
operations on. The original shared initial tensor hash also matched.

Persistent clean inputs and initialization passed verification. Three noisy
views were generated on Drive and reloaded/re-tokenized successfully. Their
preparation identity, every epoch's audit and encoded-content hashes exactly
matched the local outputs above. View generation/writes before final reload
took 227.37 seconds; the notebook's complete preparation cell took longer.
Preparation itself performed zero optimizer updates and did not access test data.

The separately named calibration passed all clean-pair/source/input checks and
completed **30 optimizer updates / 960 records** from the shared initialization.
It used epoch-02's view and explicitly put the longest 32 prepared records first
(maximum length 230 subwords). All 30 recorded losses were finite. Its updated
weights were discarded; formal training starts fresh from the shared initialization.

| Observation | Value |
| --- | ---: |
| Microbatch × accumulation | 8 × 4 = effective 32 |
| Calibration loop, including startup | 25.573 seconds |
| Mean update, excluding first five startup updates | 0.360647 seconds |
| Median update after startup | 0.318099 seconds |
| Peak allocated GPU memory | 2.434 GiB |
| Peak reserved GPU memory | 2.838 GiB |
| Projected 10,968-update work | 65.93 minutes |

The projection excludes data/model preparation, validation and checkpoint/model
export I/O; it is not a promised full-run duration or a model-quality result.
No batch/optimizer/noise settings were changed after calibration.
Calibration identity:
`e7a0bc7ce94118fae25b48df95ce2e9a33270de3a3e38de2c9f6d74f65f098cf`.

The final preparation-only check exercised the full-training prerequisites
without entering its training branch: calibration/source/view identities,
checkpoint implementation hash and finite event count passed. The formal
`train-augmented-seed-586-v1/` output directory did not exist. Both execution
switches were set back to **False** and the notebook displayed the completed
calibration separately from the unstarted full run.

Drive's project-page sidebar displayed **2.34 GB of 15 GB used** at the capacity
check, approximately 12.66 GB free. This supports the current three-epoch plan
and a temporary transfer ZIP; capacity can change before launch. No account-wide
file/Trash cleanup was performed. The existing manual download → local archive
verification → scoped cleanup agreement remains in effect.

Raw receipts are saved outside Git under
`artifacts/training/training-v1/augmented-preparation-seed-586/`: calibration
JSON/events/progress/run metadata, Colab view manifest/readiness, tiny Drive
inspection report and screenshots. Transferred file hashes, calibration run
identity, all source hashes and local-versus-Colab epoch audits passed local
verification. The JSON config's historical `status` string is retained because
the calibration pins its bytes; this document records the current execution state.

## Current execution status

- CPU and Colab view generation, persistence, reload, tokenization/target audits: passed.
- Matched clean-run inputs, initialization, numerical recipe and runtime: passed.
- Tiny local and Drive two-slot storage inspection: passed within the stated scope.
- Separate 30-update augmented T4 calibration and final source/view checks: passed.
- Both notebook execution switches: **False**; formal augmented training not started.
- Actual model/optimizer resume, expanded evaluation and held-out test: not run.
- Static lint and whitespace checks: passed; no unit test suite was added or run.
