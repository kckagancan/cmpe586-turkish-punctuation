# Clean fine-tuned BERT: medium-noise validation pilot

## Prespecified scope

Evaluate the completed clean training run's selected epoch-03 model on a fresh
clean control and the same four medium/seed-586 validation conditions as the
[published-reference pilot](noise-pilot.md). This is validation analysis, with
one training seed and one corruption seed. It is not held-out test evidence or
an augmentation experiment. No optimizer or training update is involved.

- Training run identity:
  `5ddecc9b8bfaf0b9442a7942caa45a64173e037b0ea7309692c12e5767603f37`.
- Pinned selected model tensor digest:
  `487a8c7c38d05787f8527eca642b4d2a80bb94a814b960479b9668a87492cd96`.
- Validation: 14,453 records / 179,164 word targets; same quarantine policy.
- Noise: case loss, diacritic loss, internal adjacent swap and combined, medium,
  corruption seed 586. Existing records/masks are reused without regeneration.
- CPU, batch 16, four PyTorch threads, FP32; fresh clean control and all variants
  share the published pilot's clean-length row ordering and batch membership.
- Main score: PERIOD/COMMA/QUESTION_MARK macro-F1. Also retain micro and per-class
  P/R/F1, confusion matrices, clean-to-noisy paired outcomes and error examples.

## Implementation and reproduction

`configs/clean-noise-evaluation.json` pins the archived training run and selected
tensor digest. `evaluation_model.py` verifies the completed run/epoch selection,
export files, original validation outputs and model loading. The existing
`scripts/evaluate_noise.py` also handles this verified local model, preserving
the same tokenization, alignment, prediction and metric functions used by the
published pilot. It re-tokenizes all five inputs before scoring and rejects
condition-specific row drops, changed targets or mismatched batching/packages.

The original training-time GPU predictions/metrics are retained as historical
references. Fresh CPU clean predictions are used for this pilot's paired noisy
comparisons; any differences from the training-time predictions are recorded in
the clean-anchor paired analysis. Training-time prepared-file hashes differ from
the local Arrow cache because the preparation environments differ; raw validation
hashes, every source target and every local encoding are checked explicitly.

```bash
python scripts/evaluate_noise.py --config configs/clean-noise-evaluation.json
python scripts/analyze_noise.py \
  --run-directory artifacts/evaluations/finetuned-clean-seed586-noise-pilot-v1 \
  --comparison-run-directory artifacts/evaluations/published-bert-noise-pilot-v2
```

After interruption, use the identical evaluation command with `--resume`.
Completed condition files are checked and retained. Existing completed runs are
never overwritten; code/input/config changes require a new output directory.

The model ZIP/extracted archive and prepared noise data must already exist locally.
Results, progress and source snapshots are saved under the ignored evaluation
directory named above. This step writes no Drive files and deletes no models.

## Completed results — 2026-10-19

All five conditions completed successfully. Every condition evaluated the same
14,453 records / 179,164 word targets. The fresh CPU clean control reproduced
**all 179,164 training-time word predictions and all metrics exactly**, including
prediction-file SHA-256 `4eb360cc7751dab9cba7cf721acff52fd7f9d1fb668581b3e5885c75546c3861`.

Runtime: Torch 2.14.1+cpu, Transformers 4.57.6, datasets 4.8.5, NumPy 2.4.6 and
scikit-learn 1.9.1. Shared inference, metrics, preprocessing, data, tokenization
and robustness source hashes match the published pilot. Input hashes, row/batch
membership, package versions and per-class supports also match.

P/R/F1 values below are percentages. Drop is clean minus noisy in **percentage
points (pp)**; O remains in the confusion matrix, but is excluded from the
primary three-class punctuation macro/micro averages.

| Condition | Changed words (%) | Macro P | Macro R | Macro F1 | Drop (pp) | Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean control | 0.00 | 86.03 | 82.47 | 84.04 | 0.00 | 82.99 |
| Case loss | 5.00 | 84.95 | 78.11 | 81.23 | 2.80 | 80.15 |
| Diacritic loss | 5.00 | 86.28 | 81.93 | 83.80 | 0.24 | 82.84 |
| Typo swap | 5.00 | 85.98 | 81.09 | 83.20 | 0.84 | 82.30 |
| Combined | 14.33 | 84.98 | 76.28 | 80.10 | 3.94 | 79.17 |

Each cell below is **precision / recall / F1**:

| Condition | Period P/R/F1 | Comma P/R/F1 | Question P/R/F1 |
| --- | ---: | ---: | ---: |
| Clean control | 94.63 / 97.35 / 95.97 | 70.33 / 57.66 / 63.37 | 93.13 / 92.41 / 92.77 |
| Case loss | 94.85 / 93.16 / 94.00 | 67.28 / 53.20 / 59.42 | 92.72 / 87.98 / 90.29 |
| Diacritic loss | 94.59 / 97.35 / 95.95 | 71.11 / 56.03 / 62.68 | 93.13 / 92.41 / 92.77 |
| Typo swap | 94.48 / 97.15 / 95.80 | 70.24 / 54.63 / 61.46 | 93.22 / 91.47 / 92.34 |
| Combined | 94.69 / 92.89 / 93.78 | 67.24 / 48.56 / 56.40 | 93.01 / 87.38 / 90.11 |

Fixed supports: O 150,845; PERIOD 15,252; COMMA 11,894; QUESTION_MARK 1,173.

## Matched published-reference comparison

| Condition | Published macro F1 | Clean fine-tuned macro F1 | Fine-tuned minus published (pp) | Published drop | Fine-tuned drop |
| --- | ---: | ---: | ---: | ---: | ---: |
| Clean control | 83.6366 | 84.0365 | +0.4000 | 0.0000 | 0.0000 |
| Case loss | 81.2256 | 81.2344 | +0.0088 | 2.4110 | 2.8021 |
| Diacritic loss | 83.4347 | 83.7975 | +0.3628 | 0.2019 | 0.2390 |
| Typo swap | 82.9044 | 83.2000 | +0.2955 | 0.7321 | 0.8366 |
| Combined | 80.3439 | 80.0965 | -0.2474 | 3.2927 | 3.9400 |

The clean gain of **0.4000 pp** does not persist across all noisy conditions.
Case-loss absolute F1 is nearly equal to the published model (+0.0088 pp), while
combined F1 is **0.2474 pp lower**. The fine-tuned model has a larger drop from its
own clean score in each noisy condition. Absolute noisy F1 and relative loss
answer different questions; better clean performance alone does not establish
better robustness.

This is a descriptive, single-training-seed/single-corruption-seed validation
comparison. Both checkpoints may have validation-selection exposure; the
published training recipe is not controlled by our experiment. The upcoming
clean-versus-augmented comparison will share initialization and training budget.
No significance, held-out generalization or augmentation benefit is established here.

## Paired errors and interpretation

| Condition | New errors | Recovered errors | Changed wrong class | Net added word errors |
| --- | ---: | ---: | ---: | ---: |
| Case loss | 1,425 | 598 | 77 | 827 |
| Diacritic loss | 385 | 366 | 22 | 19 |
| Typo swap | 770 | 488 | 31 | 282 |
| Combined | 2,352 | 1,123 | 114 | 1,229 |

These include all four target classes; their net difference is a word-error count,
not an F1 change.

| Condition | Error word edited | Unchanged error word, edited context | Entire record unchanged |
| --- | ---: | ---: | ---: |
| Case loss | 161 | 1,264 | 0 |
| Diacritic loss | 104 | 281 | 0 |
| Typo swap | 324 | 446 | 0 |
| Combined | 743 | 1,609 | 0 |

- **Case loss:** 1,264/1,425 new errors (88.70%) occur at an unchanged word in an
  edited record. Among 659 newly wrong period predictions, 524 become comma, 130
  become O and five become question marks. Thus boundary-type confusion is a
  major component, alongside omissions. Of 130 new period omissions, 96 occur
  at an unchanged word whose following word was lowercased.
- **Diacritic loss:** macro-F1 falls only 0.2390 pp, while 385 new errors and 366
  recovered errors coexist. Period and question recall remain at their clean
  values; comma recall falls from 57.6593% to 56.0282%. Small aggregate changes
  therefore hide substantial changes in which words are correct.
- **Typo swap:** comma F1 falls from 63.3680% to 61.4613%, a larger drop than
  either period or question F1. Subword fragmentation alone is not an explanation
  for the noise ranking: case loss causes a larger aggregate F1 drop.
- **Combined:** comma F1 falls to 56.3952% (6.9728 pp below clean), with recall
  down to 48.5623%. Combined coverage is 14.3293%, versus 5% for each single
  family, so this result does not isolate a noise-interaction/synergy effect.
- No predictions change in wholly unedited records in any noisy condition.

One preselected source-order inspection example is row 22, word index 4:
`degil` stays unchanged, while the following `Kara` becomes `kara`; the model
changes PERIOD to O. Other words in that record are also edited. This is
consistent with contextual casing acting as a boundary cue, rather than an
isolated causal test of that one following word. Tokenization/position changes
can also affect contextual representations. All sampled examples remain in
local `error_examples.json` files, with both complete inputs and predictions.
The limited source-order examples are inspection cases, not representative samples;
the paired/context tables above count the complete validation population.

## Artifacts and next step

Local directory: `artifacts/evaluations/finetuned-clean-seed586-noise-pilot-v1/`.
It contains `run.json`, `summary.json`, completed `progress.json`, source snapshots,
five condition directories with predictions/metrics/examples, and verified
`analysis/{analysis.json,comparison.md,model-comparison.json,model-comparison.md}`.
Large/raw artifacts remain outside Git; this document retains aggregate results.

Successful inference loops including prediction I/O totaled **27.65 minutes**; preflight and postprocessing are excluded.
No training, model deletion or Drive writes occurred in this step.

The matched augmented training and its medium-noise pilot subsequently completed
on 2026-10-22 with the shared initialization, budget and clean selection rule.
Its combined macro-F1 is 81.8435%, versus this clean model's 80.0965%; full
results and class-specific trade-offs are in the
[augmented comparison](augmented-noise-pilot.md). The broader severity/corruption
seed evaluation, training ablation and held-out test evaluation remain outstanding.
