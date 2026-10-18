# Fine-tuning protocol: controlled clean versus augmented BERT

Specified **2026-10-18**, version **training-v1**. Settings are in
[configs/training.json](../configs/training.json). The clean runner completed a
**30-update T4 calibration**; see [observations](training-calibration.md).
Full fine-tuning, full-epoch validation/selection and checkpoint/resume remain
unexecuted. Train augmentation preparation is still planned. No test evaluation
has run as part of this training work.

## 1. What the comparison measures

Train two four-class punctuation models from the same Turkish BERT initialization:

- **Clean:** every training row uses its released, unmodified words.
- **Augmented:** the same rows and targets, with a controlled subset presented
  through a noisy view. Each row still appears exactly once per epoch.

Share architecture, initial parameter tensors, row order, loss, optimizer,
learning-rate schedule and full training budget. The changed training input is
the intended experimental difference. The published punctuation checkpoint stays
a separately reported reference, with its own training provenance.

The protocol follows development inspection of the published model's
[medium-noise pilot](noise-pilot.md). It is not a preregistration made before
seeing those validation results. Final-test text and predictions will not inform
these decisions. Gains, including clean/noisy trade-offs, remain unknown.

## 2. Sources and project decisions

Checked on 2026-10-18. Sources provide starting points, not an established optimum
for this dataset or evidence of future gains.

| Primary source | What it supports | Our choice |
| --- | --- | --- |
| [Devlin et al., BERT, Appendix A.3](https://aclanthology.org/N19-1423.pdf) | Fine-tuning ranges include batch 16/32, learning rates 2e-5/3e-5/5e-5 and 2/3/4 epochs; dropout is 0.1. | Start with effective batch 32, peak LR 2e-5, three epochs and dropout 0.1. No exhaustive search is scheduled. |
| [Kurt & Çayır, official training code](https://github.com/uygarkurt/Turkish-Punctuation-Restoration/blob/main/main.py) | Uses BERTurk with a four-class token head, three epochs and train batch 64. Its declared LR and custom optimizer LR differ. | Retain the task/model starting point; define our optimizer and budget explicitly. This is not an exact reproduction. |
| [BERTurk model card](https://huggingface.co/dbmdz/bert-base-turkish-cased) | Provides a cased Turkish pretrained encoder. | Start both arms from the pinned base encoder; initialize a new punctuation head. |
| [Karpukhin et al., W-NUT 2019, section 2](https://aclanthology.org/D19-5506/) | Mixes clean and synthetic inputs and refreshes noise during training. | Keep clean views and refresh train noise each epoch, using our Turkish operators and project-specific row mixture. |
| [Hugging Face scheduler documentation](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup) | Defines linear warmup followed by linear LR decay to zero. | Use this schedule with a project-chosen 10% warmup. |
| [PyTorch serialization documentation](https://docs.pytorch.org/docs/2.14/notes/serialization.html) | Describes tensor state dictionaries and restricted weights-only loading. | Checkpoints store tensors/primitives, including serialized RNG state; loading explicitly uses `weights_only=True`. |

### Why the upstream LR needs care

The public `main.py` inspected at `main` declares `TrainingArguments.learning_rate`
as **2e-5**, then constructs an `AdamW` with **lr=1e-3** and passes that optimizer
to `Trainer` through `optimizers=(optimizer, None)`. A supplied optimizer controls
its parameter-group LR; the declaration alone is not a trustworthy account of
the effective recipe. The [Trainer API](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)
documents custom optimizer injection. This code observation does not establish
which recipe produced the released checkpoint or what the paper actually ran.

Our choice of **2e-5** is a conservative starting candidate within BERT's published
fine-tuning range. It is not claimed to be optimal. The 50% row mixture, 5%
component dose, 10% warmup and gradient clipping below are project decisions.

## 3. Initialization and data

Use `base_model_id` and `base_model_revision` from
[tokenizer.json](../configs/tokenizer.json):

- Encoder: `dbmdz/bert-base-turkish-cased` at
  `b6e1de16c983e0f2c70664591ea3f22810072608`.
- Architecture: `BertForTokenClassification`, all encoder/head parameters trainable.
- Fresh four-class head: `O=0`, `PERIOD=1`, `COMMA=2`, `QUESTION_MARK=3`.
- Hidden, attention and classifier dropout: 0.1.

For each training seed, construct and save the entire initialized model **once**.
Start clean, augmented and any training ablations from those identical tensors;
record the initialization hash. A shared seed alone is not the evidence of shared
weights. Reset run RNGs before each arm; separate row-order/noise generators from
dropout RNG use. Do not initialize from the published punctuation head.

Reuse the pinned tokenizer assets already named in `tokenizer.json`, after
verifying their vocabulary IDs, special tokens and tokenization configuration
against the pinned base model's tokenizer and embedding dimensions. This
compatibility check is implemented but has not run yet: the pinned base assets
are not cached locally. Reproduce all prepared train/validation encodings with
the base tokenizer before training; fail on incompatibility instead of silently
changing token IDs.

Use fixed `clean-v1` IDs: **116,969 train rows / 1,463,897 word labels** and
**14,453 validation rows**. Preserve the released split and existing whole-row
quarantine policy. Load train and validation explicitly; training must not read
test text or iterate a DatasetDict containing test. The original overlap and
document-provenance limits remain as recorded in [data-audit.md](data-audit.md).

Retain first-subword supervision, `-100` elsewhere, dynamic padding, the 512-position
limit and complete word coverage. Do not truncate, silently drop rows or infer
sentence endings from CSV row boundaries.

## 4. Optimization recipe and its meaning

| Setting | Initial value | Reason |
| --- | --- | --- |
| Optimizer | `torch.optim.AdamW`, betas (0.9, 0.999), epsilon 1e-8 | Explicit adaptive optimization settings shared by both arms. |
| Peak learning rate | 2e-5 | A cautious full-encoder fine-tuning starting point. |
| Weight decay | 0.01; zero for biases and LayerNorm parameters | Define regularization groups explicitly. |
| Training budget | Three complete epochs, no early stopping | Match data exposure and scheduled updates. |
| Physical / effective batch | 8 records × accumulation 4 = 32, one GPU | Start with manageable activation memory. |
| Schedule | Linear warmup for ceil(10% of updates), then linear decay to zero | Introduce updates gradually and reduce step size later. |
| Gradient norm clipping | 1.0 after accumulation | Bound unusually large accumulated updates. |
| Loss | Unweighted four-class cross-entropy, no label smoothing | Keep the baseline simple and isolate augmentation. |

`O` participates in training loss, so false punctuation insertions have a learning
signal. Macro-F1 gives punctuation classes equal evaluation weight; it does not
require class-weighted training. Class weighting or focal loss would introduce an
additional intervention and would need a separate experiment.

**Accumulate by word count:** sum cross-entropy over supervised positions across
an accumulation window, then divide by the window's total non-`-100` count.
Equivalently, scale each microbatch's summed loss by that shared denominator.
Do not simply average four per-microbatch mean losses: record lengths vary, and
the short last window needs its actual denominator. This objective gives every
original word equal weight within the update, irrespective of subword splitting.
Step the optimizer and scheduler once per completed accumulation window; flush
the partial window at each epoch end. Evaluation loss is also word-weighted.

For the current train count, batch 8 and accumulation 4 imply:

```text
microbatches/epoch = ceil(116969 / 8) = 14622
updates/epoch     = ceil(14622 / 4)   = 3656
total updates     = 3 * 3656         = 10968
warmup updates    = ceil(.10 * 10968) = 1097
```

The final update each epoch contains nine records. Keep them. Both arms see
350,907 record presentations and 4,391,691 supervised word presentations over
three epochs. Extra noisy subwords change compute, so this is an equal
record/word/update budget, not an equal FLOP or wall-clock budget.

## 5. Augmented training views

This first augmentation recipe is intentionally small enough to audit and ablate:

1. For each epoch, build the existing `noise-v1` **combined-medium** plans on the
   entire clean training split: 5% of all word positions selected per family,
   with the existing eligibility and composition rules.
2. Choose exactly `floor(0.50 * N_train_rows)` row IDs by stable SHA-256 rank,
   without looking at labels, predictions or the training arm.
3. Present the combined noisy view for those rows and the clean view for the
   rest, in the same shuffled row order as the clean arm. Replace in place;
   never append extra copies. This selects **58,484 noisy-view rows** and
   **58,485 guaranteed clean-view rows** per epoch.
4. Refresh both word plans and row selection once per epoch. Retokenize and
   realign before use; the targets and supervised word count stay unchanged.

Row rank key: compact UTF-8 JSON
`[training_version, training_seed, epoch_1_based, row_id, "noisy_view"]`, using
`ensure_ascii=False` and `separators=(',', ':')`. Sort by full SHA-256 digest,
then row ID. Training word plans use split `"train"` and seed
`700000 + 10 * training_seed + epoch_1_based`; seed 586 therefore uses
705861/705862/705863. Evaluation corruption seeds remain 586/587/588. Reuse the
existing noise selection/operation code and do not mix train and validation text.

**50% is a row-view fraction, not a changed-word rate.** A selected view may have
no edited word. Because word plans precede row selection, the realized exposure
after mixing is lower than the full combined view, with no exact 5% guarantee
per component. Do not force edits or resample. Audit selected and actually edited
rows, component/final changed words, overlap, per-class exposure, subword counts
and maximum sequence length for each epoch. Exceeding 512 or losing a word stops
preparation before training. All comparisons keep the same eligible row IDs.

The clean views preserve exposure to original text; the medium combined views
teach the three disturbances jointly without training separate models per test
severity. Evaluate mild/heavy/full-loss conditions to examine generalization
beyond this recipe. These are hypotheses, not promised outcomes. Equal component
budgets do not equalize information loss or eligible-word exposure.

## 6. Checkpoint selection and evaluation

Run all three epochs. Evaluate **clean validation** at each epoch end using
the existing word-level metrics. Select the epoch checkpoint with highest
`punctuation_macro.f1`, breaking exact ties in favor of the earlier epoch.
Use the same rule for every arm. Noisy validation scores do not choose the LR,
mixture or checkpoint in this initial comparison. Clean selection makes the
clean-performance criterion explicit; a robustness-based selector would be
a different experiment.

Keep the last epoch model as well. The selected epochs can differ between arms,
so report selected epoch/update alongside the common full training budget.
Use final-epoch checkpoints for a matched-update sensitivity comparison if
selection differs; do not describe different selected epochs as equal-update
models. Save the complete clean validation curve for all epochs, including losses,
macro/micro/per-class P/R/F1 and confusion matrices.

After selection, score the frozen validation severity/seed grid on GPU for both
arms and the published reference, with paired row IDs and identical saved inputs.
Existing pilot scores remain development observations. The current evaluation
runner enforces its original CPU reference settings, so GPU evaluation needs a
matching clean reference and documented runner integration first.

For final-test experiments, freeze the model-selection rule, run matrix, metrics,
seeds and severity conditions before scoring any test condition. Generate noise
from test alone and evaluate the fixed eligible test IDs. Report clean/noisy
absolute F1, degradation, augmentation-minus-clean gains, per-class P/R/F1 and
qualitative errors. Synthetic stress performance does not establish robustness
to all real informal Turkish writing.

## 7. Seeds, ablations and affordable execution

- **First milestone:** seed 586 clean/augmented pair, six dataset passes in total.
- **Training-seed confirmation:** optional matched pairs for 587 and 588 after GPU
  timing. Decide inclusion before inspecting those results; keep all completed
  seeds. With one pair, explicitly report a single-training-seed result.
- **Minimum training ablation:** after the first pair, plan an augmented arm
  without case loss at seed 586, motivated by the observed contextual casing
  sensitivity in the published-model pilot. This is part of the project scope;
  execution follows GPU budgeting. The initial evaluation can use clean plus the
  four medium/seed-586 conditions. Removing folding and removing swaps are
  optional additional arms. Reuse the same initialization, row views, shuffle,
  remaining word plans and update budget. Omit one operator without reallocating
  its word budget. Report lower realized exposure; removal effects do not
  estimate equal-dose intrinsic importance.
- **Evaluation corruption seeds:** 586/587/588 measure noise-realization variation.
  They do not substitute for independent training runs. Report these two sources
  of variation separately, retaining paired per-seed results and all three class
  scores. Any mean/std must identify what varied; one seed has no estimated std.

Training ablations are retrained models; input ablations in `noise.json` are
modified evaluation inputs. Keep the two claims separate. Extra training runs
are scheduled explicitly after GPU calibration. There is no LR grid in v1. If clean learning
curves justify a revised recipe, version it and rerun the compared arms; do not
tune one arm after seeing a favorable/unfavorable augmentation score.

## 8. GPU calibration, saved state and next implementation

Start on a single Colab GPU with FP32, TF32 disabled and batch 8/accumulation 4.
Use a short, separately named calibration run from the common initialization to
measure peak allocated/reserved GPU memory and seconds per optimizer update.
Exclude the first five startup updates from the steady-state timing and include
evaluation and checkpoint overhead separately in a full-run estimate. This timing
exclusion does not skip the LR warmup. Calibration is not a
quality result; discard its trained weights before formal runs.

If memory requires batch 4/accumulation 8 or batch 2/accumulation 16, resolve it
before formal runs, save the resolved configuration and use it for both arms.
The effective batch and update count stay the same, but dropout/numerical behavior
can differ; do not switch midway and call the runs identical. Mixed precision
is a possible later runtime revision after GPU measurement, with its dtype and
scaling behavior recorded for both arms. The observed clean calibration retained
8/4 at 2.430 GiB peak allocated memory and 0.476 s/update mean; projected update
work is about 87 minutes, excluding validation/checkpoint overhead. Full training
time and noisy-input memory remain unobserved.
Nonfinite loss/gradients stop the run with diagnostic state.

The first implementation uses eager BERT attention, disables TF32 and requests
deterministic Torch operations, with `CUBLAS_WORKSPACE_CONFIG=:4096:8` established
before CUDA initialization. Unsupported deterministic operations will fail during
calibration rather than silently changing the declared policy. The saved row
permutation uses a separate CPU generator seeded with `10 * training_seed +
epoch_zero_based`; noise seeds retain the one-based formula in section 5.

Save configuration and source snapshots, data/tokenizer/noise hashes, shared
initialization hash, package versions, hardware, precision, all RNG seeds, row
permutations and view plans. Record actual optimizer/scheduler updates, word
exposure, LR, losses, validation curves and timing. Store model/optimizer/
scheduler/RNG/sampler state in resumable checkpoints every 500 updates and at
epoch ends, keeping two recent resumable states. Retain all three candidate epoch
model exports, including best/final, to make selection auditable. FP32 model and
optimizer states require roughly 4 GB per completed arm with this retention,
plus approximately 0.44 GB for the shared initialization per seed.
Mid-epoch resumable checkpoints are not extra model-selection candidates.

Use a new local directory under `artifacts/training/training-v1/` per seed/arm;
persist Colab checkpoints to configured durable storage. Resume only the same
run identity and exact saved data/order/state, without silently restarting the
schedule or duplicating a partially processed epoch. Existing runs are not
overwritten. Data text and model weights stay outside Git. The advisory run lock
coordinates processes within one host; use one active Colab runtime for these
shared output paths. It is not a lock across independent Drive mounts/VMs.

## 9. Implementation and first Colab run

| File | Responsibility |
| --- | --- |
| `training_model.py` | Validate pinned tokenizers, load only the base encoder, save/reuse a fresh-head initialization, export HF-loadable epoch models. |
| `training_data.py` | Open train/validation split directories explicitly; check raw/prepared hashes, eligible IDs and all re-encoded columns. |
| `training.py` | AdamW groups, schedule, deterministic row permutations, summed-word gradient accumulation, clean validation loss/metrics. |
| `training_io.py` | Tensor hashes, atomic JSON/checkpoint publication, RNG save/restore, restricted `weights_only=True` loading. |
| `scripts/train.py` | Resolve configuration, archive provenance, orchestrate calibration/full training, checkpoints, selection and resume. |
| `scripts/prepare_training.py` | Copy/verify persistent inputs, initialization and calibration metadata; save preparation provenance with zero training updates. |
| `02_clean_training_colab.ipynb` | Thin setup/storage/preparation cells and a separate disabled full-training switch. |

The first reviewed clean calibration completed on 2026-10-18 using the clean
GitHub commit `fc17d41`, runtime 2026.07, Python 3.12.13 and a T4 GPU. Its 30
updates established clean GPU fit and exercised the update path; they do not
establish full-epoch validation/selection or deterministic resumption. See the
[calibration report](training-calibration.md) for measurements and remaining work.

The notebook clones the public repository, so the reviewed files need to be
synced to GitHub before that checkout can run them. The reviewed files were
committed and pushed before calibration. Notebook commands download/prepare only train and
validation using the new `--splits train validation` option. Existing all-split
commands keep their defaults and existing clean-v1 outputs are not rewritten.
Prepared inputs are cached byte-for-byte in the designated Drive project folder.
Raw train/validation files are reverified by source hash. Initial authorization
attempts failed; a successful mount was later observed. The cleaned notebook has
one mount-if-needed cell and removes the old retry, calibration rerun and report
export cells. Preparation preserves the existing calibration report, restores
cached inputs, checks every eligible encoding and verifies the saved shared
initialization on CPU. It performs no forward/backward or optimizer update.

`requirements/colab.constraints.txt` pins the main observed dependencies while
retaining Colab's GPU PyTorch installation. Preparation saves source/config
hashes, input identities, the initialization hash, actual package versions and
`pip-freeze.txt` under `preparation/training-v1/seed-586/`. This is a preparation
record, not evidence of training-checkpoint recovery or full-epoch validation.
All persistent writes are confined to `MyDrive/cmpe586-turkish-punctuation`.

```bash
# CUDA required. Default mode is calibration, not a full experiment.
python scripts/train.py --mode calibrate --download-model

# Start again from the shared initialization for formal clean fine-tuning.
python scripts/train.py --mode train --download-model

# Use identical settings and the same output/initialization paths.
python scripts/train.py --mode train --download-model --resume
```

Use `--output-directory` and `--initialization-directory` for persistent paths.
`--batch-size 4 --gradient-accumulation 8` and `2/16` are the allowed memory
fallbacks; calibration retries require new output paths. Calibration runs 30
updates by default, includes the longest clean batch, excludes five startup
updates from timing and discards its trained weights. Its estimate covers update
work including batch preparation/transfer, not validation/checkpoint/export I/O
or logging. It does not establish the augmented input memory requirement.

Full runs write `run.json`, `events.jsonl`, `progress.json`, `checkpoints/`,
`models/epoch-XX/` and `validation/epoch-XX/{metrics.json,predictions.jsonl}`.
`summary.json` appears only after the full budget, artifact verification and epoch
selection complete. Checkpoints store the next row index, saved permutation,
epoch loss/word counters, validation/selection history and RNG states along with
model/optimizer/scheduler. A failure during validation can resume without
repeating that epoch's training. Updates after the last committed checkpoint are
replayed, with corresponding uncommitted event entries removed.

The augmented view preparer, augmented runner integration and training ablations
follow review/calibration of this clean path. No augmented training command is
advertised as implemented yet.
