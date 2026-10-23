# Held-out test protocol: controlled Turkish writing noise

Specified **2026-10-22**, version **held-out-test-v1**, freeze revision **2**, before any project test
predictions or scores. Machine-readable decisions and full model/file identities
are in [configs/held-out-test.json](../configs/held-out-test.json).
This is a validation-informed final evaluation plan, not a preregistration made
before development: clean/noisy validation and the case-loss ablation were
already inspected. The present step freezes the protocol and structural row
masks only; it does not generate noise, load models or score test data.

## Research question and contribution

Does the fixed synthetic-noise augmentation improve Turkish word-level
punctuation restoration on an untouched scoring split under controlled casing,
Turkish diacritic and internal typo perturbations, and does removing case loss
change that robustness/clean-performance trade-off?

The primary contrast is **augmented minus clean fine-tuning**, holding initial
parameters, examples, order and training budget fixed. The external published
checkpoint is a reference with an uncontrolled training history. The case-loss
ablation isolates removal under the recorded recipe, with less total noise;
it does not estimate equal-dose intrinsic importance.

Punctuation restoration with noise-aware augmentation already exists, including
medical ASR work by [Sunkara et al. (2020)](https://aclanthology.org/2020.nlpmc-1.8/).
Our contribution should be described as this Turkish controlled-writing-noise
comparison and reproducible ablation, not as the first combination of punctuation
restoration and robustness. This targeted literature check does not establish
that no prior work uses an identical design.

## Literature used, and the project-specific adaptation

| Primary source checked | Relevant evidence | Decision for this project |
| --- | --- | --- |
| [Nagy, Bial & Ács (2021), §4 Results](https://arxiv.org/pdf/2101.07343) | BERT punctuation restoration reports class P/R/F1 and an overall F1 excluding EMPTY; validation selects models before test evaluation. | Retain our already implemented three-punctuation macro-F1 and clean-validation checkpoint rule. Do not copy their corpus lowercasing, model variants or multi-prediction/token setup. |
| [Sunkara et al. (2020), §3.3](https://aclanthology.org/2020.nlpmc-1.8.pdf) | Augmentation with ASR hypotheses aims to improve punctuation/truecasing robustness. | Retain the clean/noisy evaluation principle. Our controlled spelling edits preserve every word slot and label; do not copy ASR deletion/alignment/label-transfer rules or add a truecasing task. |
| [Pruthi, Dhingra & Lipton (2019), §1](https://aclanthology.org/P19-1561.pdf) | Internal character alterations can damage downstream NLP predictions. | Keep the existing deterministic adjacent internal-swap operator. Our fixed random-word budgets are not an adversarial worst-case attack or a reproduction of their defense. |
| [Ribeiro et al. (2020), CheckList](https://aclanthology.org/2020.acl-main.442/) | Behavioral evaluation exposes failures hidden by a single aggregate score. | Report each fixed noise family/dose and edited-word versus contextual errors. Do not add a new CheckList dependency or silently equate fixed targets with semantically verified invariance for every edited word. |
| [Lee et al. (2022)](https://aclanthology.org/2022.acl-long.577/) | Duplicated text and train/evaluation overlap can distort evaluation. | Preserve the released eligible cohort as primary and add a label-independent exact-input overlap sensitivity view. This simple row filter is our choice, not their LM deduplication pipeline or proof of document separation. |
| [Dodge et al. (2020)](https://arxiv.org/abs/2002.06305) | Initialization and data order contribute to fine-tuning variation. | Retain the one-training-seed limitation. Three corruption seeds describe input realizations, not independent trained models. |

The Turkish release/model source remains Kurt & Çayır's pinned repository and
our [data manifest](data-manifest.json). The publisher page did not provide
readable full text in this check; no new licensing or split-provenance conclusion
is inferred from it. No new published benchmark score is used as a target.

## Frozen models and selection

| Model | Fixed checkpoint | Planned test conditions |
| --- | --- | ---: |
| Published punctuation BERT | Existing pinned model revision and tensor digest | 39 |
| Clean-trained BERTurk | Seed 586, epoch 3 / 10,968 updates | 39 |
| Augmented-trained BERTurk | Seed 586, epoch 3 / 10,968 updates | 39 |
| Augmented without case loss | Seed 586, epoch 3 / 10,968 updates | 5 |

Each fine-tuned checkpoint was selected by highest clean-validation
`punctuation_macro.f1`, earliest epoch on an exact tie. All selected epochs are
also the final epoch, so no separate final-epoch sensitivity inference is needed.
No test result may select a checkpoint, learning rate, noise ratio, seed or epoch.
No extra training, normalization, spell correction or checkpoint averaging is
introduced. Model/export identities are pinned in the config and freeze receipt;
full weight/tensor readback remains a future runtime-readiness gate.

## Two fixed cohorts from one test release

**Primary:** the existing `clean-v1` eligible test split: **14,600 rows /
181,485 original word targets**. The same 61 whole rows containing empty words
remain quarantined; preserve original split-local CSV row IDs. Scores describe
this filtered release, not a reproduction of the paper's unfiltered test score.
Within-test duplicate inputs are retained as released.

**Sensitivity:** remove a primary test row when its entire original token list
is identical to any released train **or validation** input. Use case-sensitive
compact UTF-8 JSON token-list SHA-256, with no normalization, lowercasing, labels,
predictions or length threshold in membership decisions. Check both reference
splits because validation influenced model selection and experiment development.
The excluded set is the union, not the sum of two overlapping counts.

Read the exact cohort counts and ID digests in the generated
[aggregate cohort summary](held-out-test-cohort-summary.json). Full row masks are
Git-ignored in the protocol-freeze directory. The shared reader parses/validates
labels, but cohort selection uses words and the existing eligibility rule only.

Generate all noise on the **full primary cohort first**, then filter its saved
predictions by the frozen sensitivity IDs. Do not regenerate noise, reallocate
budgets, retokenize a reduced split or rerun inference for sensitivity. Report
realized edited-word coverage for both cohorts. Exact-input exclusion is a
sensitivity analysis; absent document/talk IDs, document-level independence and
near-duplicate separation remain unverified. The published model's own prior
exposure to these test examples is unknown.

## Fixed noise and evaluation matrix

For published/clean/augmented: one fresh clean control, **4 families × 3 levels ×
3 corruption seeds = 36 noisy conditions**, and two full-loss stresses: **39 per
model**. Families are case loss, diacritic loss, typo swap and combined; component
word budgets are 2/5/10%, with seeds 586/587/588. Reuse `noise-v1` operations and
hash rules, with the split key explicitly **test**. The existing `noise.json`
validation-development field is historical orchestration; the future test
preparer must pass the new split explicitly, without changing that frozen file.

For the case-loss ablation: clean plus all four medium/seed-586 conditions,
mirroring its fixed validation scope. This five-condition confirmation is not a
full ablation severity/seed study. Do not expand or reduce its scope after seeing
test scores. Combined reuses the single-family plans in order typo swap → case
loss → diacritic loss. Full-loss endpoints remain separate deterministic
stresses; their realized coverage is not an equal heavy dose.

Total: **3 × 39 + 5 = 122 model-condition forward evaluations**. Two reporting
cohorts produce **244 score sets from those same predictions**, not 244 inference
runs. Preserve word IDs/order/counts and targets; retokenize changed inputs and
retain first-subword alignment / ignored `-100` padding. Any missing word,
insufficient eligible edit budget or length above 512 stops the condition before
scoring; no condition-specific row drops, truncation, budget reduction or resampling.

## Inference and readiness

Use the recorded T4 GPU policy: batch 16, four CPU threads, FP32 SDPA/math,
TF32 off, deterministic algorithms, CUBLAS `:4096:8`, inference seed 586. Derive
one order from primary clean token lengths, breaking ties by original row ID;
share that order and batch membership across models and conditions. Every model
gets a fresh test clean anchor in the same runtime. Validation scores cannot
substitute for that anchor. Record hardware, packages, actual model tensors,
config/source hashes and all prepared input hashes before scoring. No CPU or
mixed-precision fallback should silently replace the frozen comparison.

Current `prepare_noise.py`, `evaluate.py` and `evaluate_noise.py` intentionally
accept validation only. **Do not relax those historical guards or run them with
only a changed split setting.** The next implementation must provide separate
test orchestration using the shared `.py` noise/inference/metric functions,
plus source snapshots, model/input/runtime gates and committed resumable outputs.
A short inference-only calibration may use existing validation inputs and planned
maximum tensor shapes, with zero optimizer updates and no test predictions/scores.
This protocol freeze is not evidence of runtime readiness.

## Analysis fixed before scoring

Report every planned condition on both cohorts, including unfavorable results.
Primary metric: word-level macro-F1 over PERIOD, COMMA, QUESTION_MARK; include O
positions in false-positive/negative counts. Keep zero-division at zero and the
fixed class order. Also retain macro/micro P/R/F1, per-class P/R/F1/support and
confusion matrices, own-clean-to-noisy drops, augmented-minus-clean differences
and augmented-minus-without-case-loss differences on its five shared conditions.
No global pooled robustness score hides the family/dose results.

For the 36 stochastic conditions, show mean and sample SD (`ddof=1`) across the
three corruption seeds per family/dose. Compute paired model deltas separately
for each corruption seed, then aggregate deltas. Do not pool predictions across
seeds to pretend they are independent examples. Clean/full-loss controls have no
corruption-seed SD; the seed-586 ablation has no estimated seed SD. These are
fixed-model descriptive results: no p-values, equivalence or training-seed
stability claims. Report achieved corruption coverage, including combined union
and overlap, and primary versus sensitivity support changes.

For the prelisted clean/case-loss/combined diagnostic conditions, use up to two
examples per punctuation class and paired outcome category, selected by ascending
original row ID then word index. Separate edited words from unchanged words with
edited context. Examples are diagnostic, not a representative sample; keep raw
texts/examples out of Git. A new test failure can motivate future work, but must
not retroactively tune this experiment and remain described as untouched testing.

## Freeze, resume and retention

`python scripts/freeze_heldout_protocol.py` verifies existing source hashes and
checkpoint-selection metadata, derives label-independent row masks and writes
`freeze.json`, `cohorts.json`, `aggregate.json`, exact source/config snapshots and
a last-written hash commit under
`artifacts/evaluation-preparation/held-out-test-v1/protocol-freeze-r2/`.
Its identity covers protocol/config bytes, sources, model metadata and row masks.
Repeated use verifies matching committed bytes; a changed specification must
preserve the old freeze and use a new version.

Before the first test inference, the future runner must bind this freeze to
prepared input/model/runtime identities. Resume retains only verified completed
conditions. Implementation corrections require an explicit new version and
preserved evidence; do not edit frozen snapshots or tune on a favorable score.
After completion, recompute metrics from raw targets, verify archives locally,
and retain only aggregate reports/figures in Git. No Drive cleanup is part of
this preparation.

Dataset licensing, exact source/split construction, document independence,
published-model prior exposure, single training seed and synthetic-noise coverage
remain reporting limitations. The test split has already had structural audits
and clean input preparation; it has not been scored in this project.

Revision 2 corrects only the Nagy Results-section citation number and active
freeze metadata/path. The initial revision remains preserved in `protocol-freeze/`.
No model, cohort rule, noise recipe, metric or run-matrix decision changed; no
test scoring occurred under either revision.
