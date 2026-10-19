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
Real Drive behavior and the augmented GPU calibration remain to be observed.
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

## Current execution status

- CPU view generation, persistence, reload, tokenization and target audits: passed.
- Static lint and whitespace checks: passed.
- Colab preparation, tiny Drive slot inspection and 30-update augmented calibration:
  pending execution.
- Full augmented training, actual model-state resume and test evaluation: not run.
