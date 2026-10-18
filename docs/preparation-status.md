# Persistent preparation status

Observed **2026-10-18**. Preparation is complete; formal clean training has
**not started**. The user requested stopping before training this session.

## Observed preparation

`scripts/prepare_training.py` completed in Colab with exit code **0** in
**102.444 seconds**, using clean code commit
`a1211940c1c12300efcf6f4b5b66a2d46b1b2cc5`.

| Item | Observed result |
| --- | --- |
| Prepared training records | 116,969 |
| Prepared validation records | 14,453 |
| Optimizer updates during this preparation | **0** |
| Test data accessed by preparation | No |
| Shared initialization tensor SHA-256 | `f26849ec8c687a095ec702cd81c3522e4e0e58a4ed52c4aeff88b0d6715cbf5e` |
| GPU / Python / PyTorch | Tesla T4 / 3.12.13 / 2.11.0+cu128 |

Input files were copied with byte hashes checked, then loaded and re-encoded
against the configured tokenizers. The saved shared initialization was loaded
on CPU and checked against its tensor hash. No forward/backward pass, optimizer
or training checkpoint was created by preparation.

The earlier calibration remains a separate reference: code `fc17d41`, identity
`add5395ed2d5b2ed1a9bb6c04d60e4cdb65882d53cee367cd9d1a6548211794a`,
30 historical optimizer updates. Its updated model weights were discarded and
were not reused. Calibration was not rerun during persistent preparation.

## Saved assets

All persistent writes were confined to the designated project folder:

```text
MyDrive/cmpe586-turkish-punctuation/
  inputs/clean-v1/
  initializations/training-v1/seed-586/
  training-v1/calibration-clean-seed-586-v1/
  preparation/training-v1/seed-586/
    readiness.json
    pip-freeze.txt
```

Local metadata copies are under the ignored directory
`artifacts/preparation/training-v1/seed-586/`: `readiness.json`, `pip-freeze.txt`,
`initialization.json` and `input-cache-files.json`. Recorded source/config hashes
match the local files. Model weights and data caches remain in Drive.

The actual environment snapshot contains 704 `pip freeze` entries. The main
dependencies are constrained in `requirements/colab.constraints.txt`; these
constraints are not a portable lock of the full Colab environment. The notebook
checks the observed GPU PyTorch version instead of replacing it with CPU PyTorch.

After preparation, `drive.flush_and_unmount()` completed successfully in 2.321
seconds. This flushed pending writes and closed the runtime's mounted Drive
access; it does not revoke the user's Google account permissions.

## Notebook state

`notebooks/02_clean_training_colab.ipynb` and its Colab copy contain 12 cells.
Old standalone permission retries, calibration reruns and temporary export cells
were removed. One conditional Drive mount remains. In the already connected
runtime, that cell completed without opening another authorization prompt.
A new runtime may require authorization again.

The full training switch remains **false**. Full-epoch validation/selection,
training checkpoint save/resume, augmented training and noisy-input training
memory have not been exercised. No new model-quality or test result is claimed.
