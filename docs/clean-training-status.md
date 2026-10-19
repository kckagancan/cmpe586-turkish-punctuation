# Clean training status

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
or resume attempt was performed. Full-epoch validation/selection and recovery
from an interruption remain unobserved. This launch record does not claim that
the three-epoch run has completed.

## Notebook handoff

The training cell streams the CLI's output live and forwards notebook interrupts.
The Colab copy enables the training switch for this authorized run; the source
notebook in Git retains `RUN_FULL_TRAINING=False` as its default.
The following summary cell is queued behind training and prints
**CLEAN TRAINING COMPLETE**, completed epoch
and update counts, best epoch, clean validation macro-F1 and model location when
`summary.json` is present. The summary is only published after the runner's final
artifact checks pass.

The user will watch the visible notebook and return when execution ends. No
scheduled monitoring or automatic follow-up was configured. Training continues
in Colab independently of this chat, while the runtime remains connected.
