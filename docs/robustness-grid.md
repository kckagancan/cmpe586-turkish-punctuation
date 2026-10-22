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

GPU calibration completed on Tesla T4, Python 3.12.13, Torch 2.11.0+cu128,
Transformers 4.57.6, datasets 4.0.0, NumPy 2.0.2 and scikit-learn 1.6.1.
All three exact model tensor identities and the paired training recipe passed.
Sample peak allocated GPU memory was **0.4872 GiB**, not a full-run memory bound.
Calibration took 126 seconds including model/tokenizer loading and preparation.
Representative-batch extrapolations were 57.6 / 38.0 / 22.4 minutes per model
(117.9 minutes total). These vary despite the same architecture and are advisory
timing observations, not evidence that one model architecture is faster.
Input audit, model loading, Drive publication and final analysis add overhead.

The validation ZIP reused 285 existing files and passed full ZIP/member hash
verification after its Drive transfer and Colab extraction. Source commit:
`085065f5203c63467321db7f5e5bbfb5b2158c45`.
The complete grid started **2026-10-22 around 12:30 Europe/Istanbul**, beginning
with the published model. Completion and aggregate results remain pending.
Held-out test evaluation remains unperformed.

Initial full-condition observations were **33.5–35.8 seconds** for the clean
control and first seven noisy conditions (14,453 records each). At roughly
35 seconds per condition, 117 inference passes alone would take about 68
minutes. Heavy/full-loss conditions, input audits, Drive publication and final
analysis can change the total; the earlier two-hour figure is not a fixed ETA.
The first eight conditions were published successfully, including their
verified `complete.json` markers. The job is left running visibly for the user;
this launch observation does not assert that the full grid has completed.

Live execution reuses the reconnected T4 in the existing
[Colab notebook, grid cell](https://colab.research.google.com/drive/1KKuqXRHKn4xr0nA7mXjtZexL4FrKFsV8#scrollTo=BMwam3sZ8fY3).
Appended cells 16–18 restore the new evaluation setup, calibrate, and run the
grid; original completed training cells are not executed. The clean standalone
reproduction notebook is `04_robustness_evaluation_colab.ipynb` in Git.
Calibration/source pins stay frozen during the running evaluation.
