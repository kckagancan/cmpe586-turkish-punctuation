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
preserved. Results are reported after completion and verification.

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
with the published model, and finished around **13:58 Europe/Istanbul** according
to the notebook execution display. All **117 evaluations and saved-metric
verification completed successfully**. The source and publication history are
preserved. [Results, class behavior and figures](robustness-grid-results.md).
Held-out test evaluation remains unperformed; one training seed was evaluated.

Initial full-condition observations were **33.5–35.8 seconds** for the clean
control and first seven noisy conditions (14,453 records each). At roughly
35 seconds per condition, 117 inference passes alone would take about 68
minutes. Heavy/full-loss conditions, input audits, Drive publication and final
analysis can change the total; the earlier two-hour figure is not a fixed ETA.
The first eight conditions were published successfully before handoff. The
job then completed all three models, recomputed every condition's metrics from
saved predictions and raw targets, checked cross-model inputs/runtime, and
published the aggregate analysis with verified `complete.json` markers.

Live execution reuses the reconnected T4 in the existing
[Colab notebook, grid cell](https://colab.research.google.com/drive/1KKuqXRHKn4xr0nA7mXjtZexL4FrKFsV8#scrollTo=BMwam3sZ8fY3).
Appended cells 16–18 restore the new evaluation setup, calibrate, and run the
grid; original completed training cells are not executed. The clean standalone
reproduction notebook is `04_robustness_evaluation_colab.ipynb` in Git.
Calibration/source pins stayed frozen throughout the evaluation. Export and
archive-persistence cells 19–20 were added after completion.
The live calibration/grid switches were then set to `False` without rerunning
either cell. Their completed outputs remain visible. The standalone notebook's
status cell now reads saved result tables and figures; its calibration/grid
switches remain off and committed notebook outputs are cleared.

## Local archive verification

The user downloaded `robustness-grid-gpu-v1-results.zip` through a normal browser
after the in-app browser did not save the download. The ZIP contains **533 files**
and is **15,740,950 bytes**, SHA-256
`30712c7dcd36684d20708eddf5c6671dfafe569737e87f3079004b37ad4caf32`.
It contains all three evaluation directories and the shared calibration,
progress, execution log and analysis; no model weights or optimizer states.

Locally, the transfer hash, ZIP membership, every member's hash/size, safe paths
and extracted bytes passed. All 117 condition publication markers, prediction
hashes, original targets, record order and recomputed metrics passed again.
All three cross-model comparisons passed, as did the 13 committed analysis
files and their source/summary hashes. The verification receipt is stored in
`artifacts/evaluation-preparation/robustness-grid-gpu-v1/completed-result-verification.json`.
An additional local aggregation check reproduced all 36 model/family/severity
summaries, 12 paired seed deltas and six deterministic model endpoints, including
sample SD (ddof=1). Its receipt is `aggregate-verification.json` in the same folder.
The original Drive results and its small archive are retained. No Drive file
was deleted. The completed source snapshot is preserved alongside every run.
