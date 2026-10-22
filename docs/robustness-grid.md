# Full validation robustness grid — GPU

## Scope fixed before broader noisy scores — 2026-10-22

Extend the completed CPU medium/seed-586 pilot using the unchanged `noise-v1`
validation variants. The 2/5/10% levels and corruption seeds 586/587/588 were
specified before the pilot. No new noise is generated, no model is retrained and
no checkpoint is selected from noisy scores.

- Models: pinned published BERT and selected epoch-03 clean/augmented seed-586
  exports. Their tensor identities are fixed in the three grid configs.
- Per model: 4 families × 3 severities × 3 corruption seeds = 36 stochastic
  conditions, plus clean and two full-loss endpoints = **39 evaluations**.
- Total: **117 model/input evaluations**, all on 14,453 validation rows /
  179,164 original word targets. Test text is excluded from the transfer bundle.
- Single-family budgets: 2/5/10% of all original words. Combined reuses three
  component masks; its measured union coverage is larger and reported separately.
- Full lowercasing and full Turkish folding are deterministic, unequal-coverage
  endpoints, not extra heavy doses. They have no corruption-seed SD.
- Means and **sample SD (ddof=1)** summarize three corruption realizations.
  Paired augmented-minus-clean deltas use matching seeds. This does not estimate
  training-seed variability: only one clean/augmented training pair exists.

## GPU controls and publication

The CPU reference cannot simply be reused as the GPU clean score. The GPU mode
requires a new clean control for each model, with the same batching, package
versions, GPU type and numerical policy across all three runs. Historical CPU
or training predictions remain diagnostic references. Their differences are
recorded by the clean-anchor paired analysis, not attributed to input noise.

Use batch 16, clean-length ordering, FP32, SDPA math, no TF32, deterministic
algorithms and fixed seed 586. Pin the tokenizer and retain first-subword labels.
The calibration receipt pins runtime/config/source hashes, verifies the matched
training initialization/recipe/budget and measures representative batches for
all models. It computes no validation scores or optimizer updates.

The published run audits all inputs before full inference. Fine-tuned runs must
match its complete preparation, rows, batches, packages and GPU numerical policy.
Every model independently checks raw targets, all encodings and export identity.
Before each condition is published, staging files are read back and hashed;
`complete.json` is written last. Resume retains verified commits and retries
incomplete conditions. Progress files are advisory and overwritten in place.
No model or checkpoint is deleted. Code/config changes require new provenance;
committed but modified condition files cause an error.

## Files and execution

- `configs/robustness-grid.json`: frozen matrix and transfer SHA.
- `configs/{published,clean,augmented}-robustness-grid.json`: model pins and
  identical GPU evaluation settings.
- `notebooks/04_robustness_evaluation_colab.ipynb`: setup, existing-input restore,
  separate disabled calibration and grid switches, status view.
- `scripts/package_evaluation_inputs.py`: existing validation-only ZIP and safe
  checksum-verified restoration. The verified ZIP is 71,888,756 bytes, SHA-256
  `579a6a6ff5969ce0406ef7bbd057a5fe56c5058ebb17e6cb4918e6dc360c97eb`.
- `scripts/calibrate_evaluation.py`: fixed representative batches, peak memory,
  advisory timing. Timing extrapolation excludes preparation/publication overhead.
- `scripts/run_robustness_grid.py`: reference → clean → augmented, then analysis.
- `scripts/analyze_robustness_grid.py`: recompute all metrics from saved word
  predictions against raw targets; verify cross-model inputs; produce seed
  summaries and severity PNG/PDF figures.
- `src/turkish_punctuation/gpu_evaluation.py`: numerical policy and verified
  publication/resume markers, shared by calibration and inference.

Persistent results are under the existing project's `artifacts/evaluations/`
directories. The aggregate `robustness-grid-gpu-v1/analysis/` includes per-seed
comparisons, mean/SD tables, deterministic endpoints and figures. Large outputs,
text and weights stay outside Git. Inputs, models and earlier CPU pilots remain
preserved. Results will be reported only after completion and verification.

## Current status

Prepared and locally checked; GPU timing and full-grid execution are pending.
The existing T4 session has reconnected. The input ZIP reuses 285 existing files
and passed full archive/member-hash checks. No broader noisy model scores have
been computed yet. Held-out test evaluation remains unperformed.
