# Supplemental transfer execution

Prepared **2026-11-04**, continuing the immutable
[protocol](transfer-evaluation-protocol.md) and
[13-view input preparation](transfer-evaluation-preparation.md).
This page describes execution wiring; actual runtime/readiness/launch evidence
is recorded below when observed. No completed original experiment is rerun.

## Execution and preserved numerical rules

`configs/transfer-execution.json` binds the exact protocol, freeze/input
identities, manifest/complete bytes, model/condition order and52-forward matrix.
It keeps scoring off by default. Separate execution wiring does not edit the
frozen protocol's historical preparation-only fields.

`transfer_execution.py` reads the immutable CPU-prepared artifacts on Colab.
The recorded preparation Python/packages must still match their historical
clean-input reference. The separate GPU runtime gate requires the original
T4/Python3.12.13/Torch2.11.0+cu128/CUDA12.8/cuDNN91900 and recorded package
versions; no CPU, mixed-precision or newer-runtime substitution is allowed.
The original CPU preparation gate and shared numerical code remain unchanged.

Readiness rechecks all saved inputs, raw targets, masks, morphology and fresh
tokenization. It verifies the four actual selected checkpoints' files and tensor
digests, then runs validation-only technical forwards: fixed clean-length batches
of clean/each new mechanism/control, plus validation words padded to the maximum
planned shape. It computes no test predictions/F1 or optimizer updates.

The separate resumable evaluator uses unchanged shared inference, metrics and
condition-publication functions. Each model has a fresh clean anchor and12
noisy conditions. All models/conditions share clean-length/source-ID batch
membership, batch16, FP32, SDPA/math and deterministic numerics. Sensitivity
filters the same predictions; per-class P/R/F1/support/confusions are saved.
Interrupted uncommitted outputs may be retried; committed outputs require exact
hash, original-target and run-identity checks before reuse. No qualitative
sampling was specified for this supplement, so its example arrays are empty.

## Transfer and notebook

Bundle `transfer-execution-v1.zip`: **22,133,412 bytes /205 files**,
SHA `d13f88d06fd97f5a74a891fefa0f33248bb7e9be4f7363063570d57f4905ed80`.
It includes existing audited inputs, the explicitly referenced clean encoding,
raw test CSV, freezes and exact sources. It contains no model weights or
predictions. Archive membership and every member's SHA were checked on readback.

The existing Drive validation archive is reused by exact SHA
`579a6a6ff5969ce0406ef7bbd057a5fe56c5058ebb17e6cb4918e6dc360c97eb`;
only its pinned validation CSV is extracted. Fine-tuned exports remain on Drive.
A cold VM may cache the published model's pinned revision from Hugging Face;
its file hashes are bound before actual model tensor verification.

[08 Colab notebook](https://colab.research.google.com/drive/1QWib5-KYb7FkSC49KB5NuD2iA5N4ktZm)
is in the project Drive preparation area. Its public/local source has all
execution switches off, empty outputs and null counts. Live authorized switches
are enabled one stage at a time. The notebook streams logs to Drive and remains
running during evaluation; it does not disconnect the VM or delete artifacts.

| File | Purpose |
| --- | --- |
| `src/turkish_punctuation/transfer_execution.py` | Frozen-input reconciliation and execution/readiness bindings |
| `scripts/ready_transfer_evaluation.py` | Actual model/runtime verification and validation-only GPU probes |
| `scripts/evaluate_transfer.py` | Explicit52-condition scoring, raw-target reconciliation and resume |
| `scripts/package_transfer_execution.py` | Hash-audited input/source bundle, no weights |
| `scripts/restore_transfer_colab.py` | Base-runtime gate, pinned packages, safe restore and cache bootstrap |
| `notebooks/08_corpus_transfer_colab.ipynb` | Disabled-by-default orchestration and visible logs |

Private preparation/check/upload/runtime evidence and full file inventory:
`artifacts/evaluation-preparation/transfer-noise-v1/launch-2026-11-04/`.
Drive output namespace: `artifacts/evaluations/transfer-noise-v1/` under the
existing project storage root. Readiness:
`preparation/transfer-noise-v1/readiness-v1/`.

## Actual runtime and launch

The actual VM reports Python3.12.13, Torch2.11.0+cu128, CUDA12.8, cuDNN91900 and
Tesla T4, matching the original base-runtime receipt. Drive is mounted after
the user's explicit authorization of this notebook's Drive access. The pinned
restore completed; validation-only readiness is running. No completed readiness
or launched evaluation is claimed yet. After readiness,
start the explicit runner, inspect its first committed clean/noisy conditions,
and hand control back without waiting for all52 conditions. On completion,
recompute saved-prediction metrics and all specified contrasts, archive/verify
outputs, then incorporate the verified supplement into the **final report**.
Proposal work is outside this task.
