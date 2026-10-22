# Augmented BERT: matched medium-noise validation pilot

## Scope fixed before inference — 2026-10-22

Evaluate the completed augmented run's selected **epoch-03** model on a fresh
clean control and the same four medium/seed-586 validation conditions already
used by the [clean fine-tuned model](clean-noise-pilot.md) and the
[published reference](noise-pilot.md). Reuse the existing noise words, masks and
encodings; do not regenerate noise or select a checkpoint using noisy scores.

- Run identity:
  `3cdf62cc3be4bbd21e48dc17aee7b450f90e6b1e2b6041dea185e3d8f0ceb01e`.
- Selected tensor digest:
  `f7e03bcf899af866cd7711a2747974c84fe6448b2deab002c3834811fe9c648b`.
- Same validation population: 14,453 rows / 179,164 word targets.
- Case loss, diacritic loss, internal adjacent swap and combined noise; medium
  dose, corruption seed 586. This pilot does not cover the severity/seed grid.
- Same CPU FP32 inference, batch size 16, four PyTorch threads, seed and
  clean-length ordering/batch membership as both completed reference pilots.
- Same PERIOD/COMMA/QUESTION_MARK macro-F1 primary score, micro and per-class
  P/R/F1, confusion matrices, paired outcomes and source-order error examples.
- Before inference, verify shared initialization tensors, training budget,
  optimization, selection policy, training runtime and the paired clean identity.

The principal comparison is augmented versus clean fine-tuning: both trained
from identical initialization with the same record/word/update budget. Report
absolute noisy F1 **and** each model's drop from its own clean score. The
published checkpoint is a separate reference with uncontrolled training history.
Also retain clean-score trade-offs and class-specific changes, including losses.

This is a descriptive validation experiment with one training seed and one
corruption seed. There is no significance or held-out generalization claim.
These validation data also served clean checkpoint selection. No training
updates, Drive writes/deletion or held-out test access are part of this step.

## Reproduction

The existing evaluator verifies the completed export, hashes, raw validation
targets, prepared inputs, tokenization, package versions and batch membership.
Its inference/metrics implementation is reused unchanged.

```bash
python scripts/evaluate_noise.py --config configs/augmented-noise-evaluation.json
python scripts/analyze_noise.py \
  --run-directory artifacts/evaluations/finetuned-augmented-seed586-noise-pilot-v1 \
  --comparison-run-directory artifacts/evaluations/finetuned-clean-seed586-noise-pilot-v1
python scripts/analyze_noise.py \
  --run-directory artifacts/evaluations/finetuned-augmented-seed586-noise-pilot-v1 \
  --output-directory artifacts/evaluations/finetuned-augmented-seed586-noise-pilot-v1/analysis-published \
  --comparison-run-directory artifacts/evaluations/published-bert-noise-pilot-v2
```

For an interrupted evaluation, repeat its identical command with `--resume`.
Completed condition outputs are retained and verified; the interrupted condition
is recomputed. Existing completed output directories are never overwritten.

## Completed and verified — 2026-10-22

Inference started from clean Git commit
`3526ba954ca5bfb5eb05e334f1a50ddefb3a4304` on 2026-10-22 after 45 paired-training,
runtime, source and configuration checks passed. The fresh clean control
reproduced all 179,164 training-time predictions and metrics exactly
(macro-F1 0.839880566). All five conditions then completed successfully, each
covering 14,453 rows / 179,164 word targets. Saved prediction/example hashes,
raw targets, row order, source snapshots and every recomputed metric passed the
post-run check. Both matched cross-model analyses passed input, runtime, source,
batching, target-support and prediction-hash checks.

Runtime: Python 3.11.17, Torch 2.14.1+cpu, Transformers 4.57.6, datasets 4.8.5,
NumPy 2.4.6 and scikit-learn 1.9.1. Successful inference loops including output
I/O totaled **27.2464 minutes**; preflight and postprocessing are excluded.
Completion: **2026-10-22 11:43:25 Europe/Istanbul**. No new training, Drive writes
or held-out test evaluation occurred.

## Three-model comparison

F1 values are percentages. Delta and drop are **percentage points (pp)**.
Drop is measured from each model's own fresh clean control.

| Condition | Published F1 | Clean fine-tuned F1 | Augmented F1 | Augmented − clean (pp) | Clean drop | Augmented drop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean control | 83.6366 | 84.0365 | 83.9881 | −0.0485 | 0.0000 | 0.0000 |
| Case loss | 81.2256 | 81.2344 | 82.2428 | +1.0084 | 2.8021 | 1.7453 |
| Diacritic loss | 83.4347 | 83.7975 | 83.9694 | +0.1719 | 0.2390 | 0.0186 |
| Typo swap | 82.9044 | 83.2000 | 83.5331 | +0.3331 | 0.8366 | 0.4550 |
| Combined | 80.3439 | 80.0965 | 81.8435 | +1.7470 | 3.9400 | 2.1446 |

For this seed pair and these medium conditions, augmentation improves absolute
noisy macro-F1 and reduces the clean-to-noisy drop in all four cases. Its largest
absolute gain is combined (**+1.7470 pp**), followed by case loss (**+1.0084 pp**).
Clean macro-F1 is **0.0485 pp lower**; this is an observed trade-off, not a
statistical claim of equivalence. Among single families, case loss remains most
damaging. Combined changes more words, so its larger drop does not by itself
identify a noise-interaction/synergy effect.

## Augmented precision, recall and class behavior

Each single family changes exactly **8,958 words (5%)**; combined changes
**25,673 distinct words (14.3293%)** with the same component masks used for the
other models. Labels remain O 150,845; PERIOD 15,252; COMMA 11,894;
QUESTION_MARK 1,173. O positions count toward false insertions/omissions but are
excluded from the primary three-class macro/micro averages.

All P/R/F1 values below are percentages.

| Condition | Macro P | Macro R | Macro F1 | Micro P | Micro R | Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean control | 85.79 | 82.60 | 83.99 | 85.66 | 80.65 | 83.08 |
| Case loss | 84.36 | 80.35 | 82.24 | 83.69 | 78.33 | 80.92 |
| Diacritic loss | 85.89 | 82.51 | 83.97 | 85.67 | 80.43 | 82.97 |
| Typo swap | 85.61 | 81.98 | 83.53 | 85.59 | 79.93 | 82.66 |
| Combined | 84.23 | 79.79 | 81.84 | 83.75 | 77.56 | 80.53 |

Per-class cells are **precision / recall / F1**.

| Condition | Period P/R/F1 | Comma P/R/F1 | Question P/R/F1 |
| --- | ---: | ---: | ---: |
| Clean control | 94.58 / 97.40 / 95.97 | 70.55 / 58.00 / 63.66 | 92.26 / 92.41 / 92.33 |
| Case loss | 94.76 / 93.44 / 94.09 | 66.66 / 57.84 / 61.94 | 91.64 / 89.77 / 90.70 |
| Diacritic loss | 94.43 / 97.40 / 95.89 | 70.58 / 57.46 / 63.35 | 92.67 / 92.67 / 92.67 |
| Typo swap | 94.34 / 97.36 / 95.83 | 70.33 / 56.35 / 62.57 | 92.16 / 92.24 / 92.20 |
| Combined | 94.37 / 93.64 / 94.00 | 66.66 / 55.70 / 60.69 | 91.67 / 90.03 / 90.84 |

- **Comma is the largest contributor to the gains:** versus clean fine-tuning,
  comma F1 rises by 2.5189 pp under case loss, 0.6695 under diacritic loss,
  1.1069 under swaps and 4.2927 under combined noise. Combined comma recall rises
  from 48.5623% to 55.7004%, with precision 67.2410% to 66.6566%: more recovered
  punctuation comes with a precision trade-off.
- **Benefits are not uniform across classes.** Compared with clean fine-tuning,
  question F1 is 0.4346 pp lower on clean text, 0.1001 lower under diacritic loss
  and 0.1380 lower under swaps. Period F1 under diacritic loss is 0.0536 pp lower.
  A higher macro-F1 does not imply every class improved.
- **A near-zero macro drop can hide changes:** augmented diacritic noise lowers
  comma F1 from 63.6582% to 63.3452%, while question F1 increases from 92.3339%
  to 92.6684%. The aggregate 0.0186 pp drop is partly cancellation, not unchanged
  predictions or evidence that diacritics never matter.

## Paired errors and contextual sensitivity

These compare each model's noisy output with **its own clean output**. Counts
include all four labels; new minus recovered errors is a word-error change,
not an F1 change. The two models' initially correct word sets differ, so their
new-error counts are not a direct count of baseline errors fixed by augmentation.

| Condition | New errors | Recovered errors | Changed wrong class | Net added word errors |
| --- | ---: | ---: | ---: | ---: |
| Case loss | 856 | 217 | 65 | 639 |
| Diacritic loss | 266 | 226 | 16 | 40 |
| Typo swap | 502 | 334 | 25 | 168 |
| Combined | 1,415 | 601 | 99 | 814 |

| Condition | Error word edited | Unchanged error word, edited context | Entire record unchanged |
| --- | ---: | ---: | ---: |
| Case loss | 72 | 784 | 0 |
| Diacritic loss | 73 | 193 | 0 |
| Typo swap | 166 | 336 | 0 |
| Combined | 399 | 1,016 | 0 |

Case loss still creates 784/856 new errors (**91.59%**) at unchanged words in
edited records. Of 77 new period omissions, 70 occur at an unchanged word with
an edited following word. No predictions change in wholly unedited records.
This is consistent with continuing contextual casing sensitivity; it is not a
causal isolation of the following word or a measure of all contextual effects.

Revisit the source-order example already discussed in the clean-model report:
row 22, word index 4, `degil` followed by `Kara` → `kara`. The target is PERIOD;
under case loss the clean-trained model predicts O, while the augmented model
predicts COMMA. The error changes type but remains wrong. Other words are also
edited in the record. This residual case illustrates why average improvement
does not mean the boundary problem is solved. Automatically selected examples
are stored with full inputs/predictions; these inspection cases are not
representative samples or controlled single-word interventions.

## Artifacts and next experiment

Run directory:
`artifacts/evaluations/finetuned-augmented-seed586-noise-pilot-v1/`.
It contains the run/summary/progress records, frozen source/config snapshots,
five prediction/metric/example directories, `analysis/` versus the clean model
and `analysis-published/` versus the published reference. Both analyses contain
`analysis.json`, `comparison.md`, `model-comparison.json`,
`model-comparison.md` and the analysis source snapshot.

Preflight receipt, execution log, the saved-metric verification helper and its
passed receipt are under
`artifacts/evaluation-preparation/augmented-seed586-noise-pilot-v1/`.
Model weights, text and raw predictions remain outside Git.

Next expand the frozen evaluation to mild/medium/heavy and the existing
corruption seeds to check whether these observations persist beyond this pilot.
The prespecified training ablation without case loss is still outstanding;
additional training-seed confirmation is optional under the recorded compute
budget. These are future experiments, not completed results. Do not retune or
select checkpoints against the pilot's noisy scores and then present them as
independent evidence. Held-out test evaluation remains unperformed.
