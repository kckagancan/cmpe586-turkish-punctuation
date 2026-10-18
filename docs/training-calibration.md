# Clean GPU calibration

Observed **2026-10-18** on Google Colab. This is a timing/memory measurement,
not a model-quality experiment. Full clean training has not started.

## Run identity and inputs

- Code commit: `fc17d41473e623c3be02101341675019e0e8d3ae`, clean checkout.
- Run identity: `add5395ed2d5b2ed1a9bb6c04d60e4cdb65882d53cee367cd9d1a6548211794a`.
- Configuration: the archived `training-v1` configuration, seed **586**,
  batch **8**, accumulation **4**, effective batch **32**, FP32/eager attention,
  TF32 disabled and deterministic Torch operations enabled.
- Verified eligible inputs: **116,969 train records / 1,463,897 word targets**;
  **14,453 validation records / 179,164 word targets**. Maximum encoded lengths
  were 227 and 196 respectively. Only train/validation were downloaded/prepared.
- Shared initial tensor SHA-256:
  `f26849ec8c687a095ec702cd81c3522e4e0e58a4ed52c4aeff88b0d6715cbf5e`.

The first update includes the 32 longest clean records. The remaining records
follow the seed's permutation with those records removed. Exactly **30 optimizer
updates / 960 records** were processed. Calibration uses the full experiment's
LR schedule: all 30 updates remain inside its 1,097-update warmup. Excluding five
startup updates from timing does not mean the LR warmup has ended.

## Environment

The Colab UI selected runtime version **2026.07** and a **T4 GPU**. Recorded values:

| Component | Observed value |
| --- | --- |
| GPU | Tesla T4, 15,637,086,208 bytes total memory |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA / cuDNN | 12.8 / 91900 |
| Transformers | 4.57.6 |
| Datasets | 4.0.0 |
| NumPy | 2.0.2 |
| scikit-learn | 1.6.1 |
| Tokenizers | 0.22.2 |
| Safetensors | 0.8.0 |

These are observed versions, not a complete dependency lock or a guarantee that
a later Colab session provides the same environment.

## Measurements

| Measurement | Observed value |
| --- | ---: |
| Optimizer updates | 30 |
| Updates used for steady timing | 25, after excluding the first 5 |
| Mean time per update | 0.475685 s |
| Median time per update | 0.356765 s |
| Peak allocated GPU memory | 2,609,035,264 bytes = 2.430 GiB |
| Peak reserved GPU memory | 3,047,161,856 bytes = 2.838 GiB |
| Calibration loop elapsed time | 20.538 s |
| Projected three-epoch update work | 5,217.317 s = 86.96 min |

The projection multiplies the mean by **10,968 updates**. It includes batch
preparation, transfer and training computation inside the update timer, and
excludes data/model preparation, validation, logging and checkpoint/export I/O.
It is an estimate from a short run, not an observed full-training duration.
Colab performance can vary. No memory fallback was required; retain **8/4** for
the planned clean run. Noisy-input peak memory still needs its own check.

## What this established

- The pinned published/base tokenizers passed vocabulary, normalization,
  special-token and alignment checks.
- Every eligible train/validation record re-encoded identically to the prepared
  inputs; input files and source CSVs passed the runner's hash checks.
- The pinned BERTurk encoder loaded with only the fresh four-class classifier
  missing. The shared initialization was saved and its tensor hash verified.
- FP32 forward/backward, gradient accumulation, clipping, AdamW and the scheduler
  completed 30 updates under the declared deterministic settings. Loss and
  gradient checks did not stop the run.
- The downloaded report has 30 consecutive event steps, a matching run identity,
  and source snapshots matching their recorded hashes.

This did **not** exercise full-epoch validation/selection, checkpoint save/restore,
interruption recovery or augmented training. There is no new F1 score. Updated
calibration weights were discarded; formal runs must start from the shared
initialization.

## Storage and remaining preparation

Drive mounting failed with `credential propagation was unsuccessful`, including
after the user reported granting file/document access. The error does not identify
a missing permission category. Calibration therefore used temporary storage at
`/content/cmpe586-calibration`, without a mounted Drive. Resolve durable storage
before formal training; this run did not establish Drive checkpoint I/O.

The report archive was retrieved from the notebook's visible download link and
saved locally under:

```text
artifacts/training/training-v1/calibration-clean-seed-586-v1/
  calibration.json
  run.json
  events.jsonl
  progress.json
  source_snapshot/
```

These files are ignored by Git. The archive contains metadata/logs/source, no
dataset or model weights. The temporary runtime's shared initialization and
prepared-input cache are not part of that archive and will not survive runtime
deletion. Create or copy and verify the initialization and input cache in durable
storage before formal training.

### Later storage preparation

After the user completed the authorization flow, `Mounted at /content/drive`
was observed. This does not establish which subset of requested permissions is
the minimum required by Colab. The original calibration measurements and source
snapshot above remain unchanged.

The cleaned notebook now uses one mount-if-needed cell and a separate preparation
command for persistent assets. Old permission retry, calibration rerun and export
cells were removed. It does not run a new calibration; `RUN_FULL_TRAINING`
remains `False`. Preparation verifies assets without any optimizer update.
