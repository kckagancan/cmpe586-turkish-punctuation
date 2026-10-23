# Case-loss ablation: matched GPU validation

The fixed post-training comparison in `configs/ablation-case-loss.json` evaluates
one frozen selected ablation model on clean validation and four medium-noise
conditions, corruption seed 586: case loss, diacritic loss, typo swaps and combined.
It reuses the corresponding 15 completed results from the published, clean-trained
and augmented-trained models; the completed 117-result grid is preserved.

## Scientific question and limits

Does omitting case loss from augmented training reduce robustness to case loss
and combined noise, while retaining benefits from the remaining components?
The ablation keeps initialization, selected row masks, retained word operations,
training recipe and checkpoint selection unchanged. The removed corruption
budget is not redistributed, so total training noise exposure is lower. This
estimates removal of a component under that recipe, not an equal-dose ranking.

All three fine-tuned models selected epoch 3 by clean validation. No noisy score
selects a checkpoint or new hyperparameter. This comparison has one training seed
and one corruption seed; the larger reference grid's three corruption seeds do
not turn this ablation into a multiple-training-seed experiment. All scores are
validation results. Held-out test remains unrun; dataset licensing and exact
split provenance remain unresolved.

## Versioned orchestration and gates

- `configs/ablation-case-loss-evaluation.json` pins the selected training identity
  `9a74ed0ee70110b75111f9e5a43e0cba1236cca8dafaee22eb3a371da34e6928`,
  tensor hash `ed61be05e2545fe4bb6b6ab9c4308cbe871e93109a6bfac469a6e1378ed66b89`,
  frozen plan hash and full published GPU summary hash. Batch 16, seed 586,
  four CPU threads, CUDA FP32 SDPA/math and clean-length ascending order match
  the completed GPU grid.
- `scripts/evaluate_noise.py` keeps its existing default exact-condition-list
  requirement. The explicit `verified_ablation_subset_v1` policy allows only the
  frozen ablation subset, after verifying the original full summary/config,
  source snapshots and condition commit hashes. Shared numerical/data code must
  match all 11 original source files; only this orchestration script is versioned.
  Input bytes, row order and batch membership must match the corresponding
  original conditions. Historical summaries/snapshots are not rewritten.
- `scripts/prepare_ablation_evaluation.py` verifies the selected model export,
  training/initialization identity, runtime/packages, input manifest and unchanged
  shared code against the completed GPU calibration. It probes the selected
  ablation model on 16 evenly spaced whole clean-length batches plus every
  planned condition's longest-input batch. The five technical probes compute no
  quality scores and perform zero optimizer updates. Its own source, runner,
  config and shared code are snapshotted before readiness is published. A
  matching existing receipt is reused without repeating probes.
- `scripts/run_ablation_evaluation.py` verifies readiness, invokes the shared
  evaluator and retains only verified completed conditions on resume. After
  inference it audits all 20 selected saved results against original validation
  word targets, checks source/input/batching/model/configuration provenance and
  recomputes their metrics. It publishes comparison JSON/CSV/Markdown and writes
  the analysis hash marker last. Reference inference is not repeated.
- `notebooks/06_case_loss_ablation_evaluation_colab.ipynb` provides setup, audited
  input restoration, separate readiness/evaluation switches and read-only status.
  Both switches are off; outputs/counts are cleared in the saved notebook.

The existing validation input bundle is restored byte for byte from Drive, not
regenerated or downloaded again. No weights are uploaded/downloaded for this
step: the retained epoch-03 export is used directly on Drive. Existing local
training ZIP and all three epoch models remain the completed-training backup.

## Locations

Drive paths are relative to `MyDrive/cmpe586-turkish-punctuation/`:

- Inputs: `inputs/validation-inputs-noise-v1.zip`.
- Preparation: `preparation/case-loss-ablation-evaluation-v1/seed-586/`.
- New output: `artifacts/evaluations/finetuned-augmented-without-case-loss-seed586-medium-gpu-v1/`.
- Analysis: new output's `analysis/{comparison.json,comparison.csv,comparison.md,complete.json}`.
- Local execution receipts/inventory:
  `artifacts/evaluation-preparation/ablation-case-loss-medium-gpu-v1/`.

The working Colab remains the historical notebook named
`03_augmented_training_colab.ipynb`:
<https://colab.research.google.com/drive/1KKuqXRHKn4xr0nA7mXjtZexL4FrKFsV8>.
Use only the appended evaluation cells. Do not Run all or rerun training cells.
The runtime checkout starts at `2cf928f`; the four new/changed evaluation files
are deployed with exact hashes and preserved by readiness/run source snapshots.
The new evaluation therefore records a dirty checkout truthfully; historical
training/grid recipes and archived source snapshots remain unchanged.

## Execution record — 2026-10-22

The user authorized execution without reducing checks for account usage.
Runtime inspection found a connected Tesla T4, no active experiment and the
retained selected export/reference grid. The audited validation bundle was
restored with full ZIP/member verification. Five technical probes passed:
245 records per condition, about 0.48–0.52 seconds each; peak allocated memory
at most 519,292,928 bytes. These are probe timings, not full-evaluation duration.
Readiness identity:
`d7cd9857f943a6c93911f44dc78db463b73a1aef632b7fa21ca373f607f5fcc5`.

The wrapper started **19:43:54 Europe/Istanbul**, PID **56025**, in working Colab
cell 34; cells 31–33 contain runtime inspection, deployment/restoration/readiness
and the saved readiness readback. Existing readiness was verified and probes
were not repeated. All four noisy conditions passed full re-tokenization and
word-target verification. At handoff, clean/case-loss/diacritic-loss/typo-swap
completed and combined inference was running. The clean anchor reproduced
84.1310%; final cross-model analysis was pending at that handoff. The launch switch was
turned off in the saved live cell while its existing process continued; Colab
was left visible with output in view.
A readback-only cell initially contained an incomplete hand-copied model hash;
it was corrected to read the exact pinned config. No model/config/receipt hash
or scientific gate was relaxed, and no inference was repeated for that correction.

Local source hashes match the persisted Colab readiness. Python/notebook syntax,
changed-script lint and whitespace checks passed. No test suite was run.

## Completion and local verification

The wrapper finished **19:48:59 Europe/Istanbul** on 2026-10-22. All five new
conditions and the 15 reused reference results passed its raw-target audit.
On resumption, the Drive connector independently read the final comparison
marker and downloaded 62 exact result/preparation files (7,921,211 bytes).
The transient `.run.lock` was omitted. No weights were transferred and no Drive
data was deleted. Existing verified training archives were reused.

`scripts/finalize_ablation_evaluation.py` verified transfer sizes/hashes, all
condition commits, full readiness/evaluation/config/model identity, 16 preflight
source snapshots and the final four-file analysis commit. It invokes the frozen
runner's read-only verification path to recompute all 20 saved metrics locally,
then checks comparison JSON/CSV values against those audited metrics. Evaluation
commit identities use compact JSON; readiness uses the existing training helper's
JSON format. The new local helper initially used the latter for a condition commit;
it was corrected to the frozen evaluator's exact format before the audit passed.
No historical identity or verification gate was changed.

The locally created results-only archive has **62 source files plus manifest**,
**730,799 bytes**, SHA-256
`5a2e803a4a08ec13816320765d53ace4b24d0915b51c44ab3a29531af5e4bfd0`.
Its complete member set, safe paths, file sizes and hashes passed readback.
Archive, downloaded bytes, receipts and complete inventory are in the local
preparation directory's `completion-2026-10-22/` subdirectory. The canonical new
run is also restored under `artifacts/evaluations/`.

`scripts/report_ablation_evaluation.py` reads only verified aggregate values and
publishes the [results report](case-loss-ablation-results.md), exact comparison
CSV and PNG/PDF figure. The original run and its analysis remain byte-identical.
No training, inference or test suite was run during finalization. The experiment
is complete; no job remains running. Next is a separately frozen held-out test
protocol/matrix, not another ablation training run.
