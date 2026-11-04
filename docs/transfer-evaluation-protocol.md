# Corpus-anchored supplemental transfer evaluation, v1

Specified **2026-11-04**, after instructor feedback and the completed
[corpus depth study](informal-corpus-depth-results.md), before any new model
scores. This is a post-feedback supplemental controlled evaluation on the
original, already scored test text. It is not independent natural-text test
data or a blind preregistration. No original recipe, model or score is replaced.

## Question, scope and fixed matrix

Does training-v1 augmentation transfer to three word-preserving mechanisms
outside its explicit case-loss/full-fold/interior-swap recipe? Existing encoder
pretraining and the published checkpoint's historical exposure are uncontrolled;
do not claim those models have never encountered such strings.

Use the existing selected published, clean, augmented and without-case-loss
checkpoints from the exact original held-out model bindings. Training-v1
selection remains highest clean-validation punctuation macro-F1, earliest epoch
on an exact tie; all fine-tuned arms retain epoch3/update10968. No training,
checkpoint search or new optimizer step is planned.

Each of the four models has a fresh clean anchor plus four conditions at each
corruption seed586/587/588: elongation, one internal omission, partial folding,
and matched full folding. Thus **13 input views / 52 model-condition forwards /
104 primary+sensitivity score sets**. There is one empirical anchor per family;
no unsupported three-way natural severity grid or mixed-noise condition is added.
All specified outcomes, favorable and unfavorable, must be reported.

Reuse the frozen original cohorts: primary14,600 rows/181,485 word targets;
sensitivity14,445/181,116 excludes the previously identified155 exact input
overlaps. Filter the same generated primary inputs/predictions for sensitivity;
do not rerank, resample or recompute budgets. The original61 quarantined rows
stay outside both. Document independence, dataset terms/exact provenance and
published-model exposure remain unresolved.

## Corpus profile, opportunities and exact word dose

Reuse the completed MultiLexNorm Turkish **train-only** calibration run
`f8b903a57451c895712c7a7e71bb7061a02dbeeff123cf3efe4b58605f15039f`.
Do not read the corpus test file or collect new model scores to choose profiles.
Compile **aggregate anonymous morphology templates**, not lexical replacements,
from its verified affected-slot features. Store no source word, tweet, user,
sentence ID or original pair hash in the public compiled profile. Bind the
compiler, inputs, original source SHA, frozen folding rules and profile bytes.

| Mechanism | Minimum corrected/source word length | Corpus anchored hit/opportunity rate |
| --- | ---: | ---: |
| Elongation | 2 | 89/4512 |
| One strictly internal omission | 3 | 50/4174 |
| Partial folding | 2, at least2 foldable occurrences, casing-compatible | 43/617 |

Both sides of the corpus pairs must be alphabetic in the frozen Turkish/ASCII
alphabet and one-to-one. On target text, eligible clean words use that alphabet
and the stated length/operation constraints. Counts include unchanged eligible
corpus pairs. These are conditional finite-corpus anchors transported to a
different domain, not estimates of Turkish population noise prevalence.

The completed case-neutral corpus profile has partial folding43/621. Applying
the frozen surface map while preserving casing exposes Turkish `I/İ`
noncommutation: Turkish-lowercase(fold(word)) can differ from
fold(Turkish-lowercase(word)). Exclude such **words from the partial/control
opportunity pool** to isolate lexical folding. On the source train split this
removes4 opportunities and zero partial-fold hits: **43/617 (6.9692%)** is the
separate operational rate. This is not the initial min4 pool of617; membership
differs. Preserve the earlier43/621 profile and its original interpretation.

For each family, select exactly `floor(N_eligible * numerator / denominator)`
positions from clean primary inputs by ascending full SHA-256 rank. The
new `transfer-noise-v1` namespace uses version, seed, split, original row ID,
word index, family and purpose. Serialization is the existing explicit UTF-8
JSON-array convention; ties use original row ID/word index. Labels, model ID,
condition display name and corpus vocabulary never enter selection. Family
plans are independent; each condition edits only its own family. This uniform
whole-split selection preserves the pooled anchor but **simplifies the observed
tweet clustering/co-occurrence**; it is not a natural tweet simulator.

## Per-word morphology and declared adaptation

All profile draws use a purpose-separated SHA-256 integer modulo the sum of
positive integer weights. Retain the full measured tails; no clipping,
post-score resampling or retry-until-favorable edits are allowed. Existing
surface characters keep their case except the explicit frozen folding map.

- **Elongation:** aggregate joint templates of ordered added-character counts
  and their source relative run-end positions. The89 source occurrences include
  77 one-run and12 multi-run words, up to25 added characters. At a target word,
  condition templates only on having no more changed runs than target
  Turkish-case-neutral character runs. Map ordered source relative positions
  onto distinct ordered target run ends, minimizing summed absolute relative
  position distance. Repeat the original surface character at each chosen run
  end by its template's extra count. Do not lowercase the word itself.
- **Omission:** retain exactly one strictly internal deletion. Represent all50
  source events in the position distribution. The47 uniquely aligned events
  have weight2 each; each candidate of the3 two-way ambiguous events has
  weight1, total100. Uniform ambiguity marginalization is an explicit modeling
  assumption, not recovered writer intent. Map the drawn relative index to the
  nearest legal target interior index, preserving first/last characters.
- **Partial folding:** retain joint source available-count, folded-count and
  ordered relative changed-position templates. If the target foldable-count
  has source support, draw only that conditional group. Otherwise draw from
  all43 events and map its folded fraction to the nearest nonempty proper
  target subset size. Adapt ordered source positions by midpoint-quantile
  resampling when subset cardinality changes, then minimum-distance ordered
  assignment to distinct foldable target positions. Record this extrapolation;
  never silently draw an incompatible folded/available combination.

Ordered-assignment primary costs use exact rational arithmetic. Equal primary
costs minimize the sum of purpose-separated candidate SHA integers, then the
candidate-index tuple. Nearest-integer ties likewise use a dedicated hash.
Template conditioning and mapping can change the realized target distribution;
report donor usage, mapping/extrapolation counts and actual edit histograms.
No claim is made to reproduce the full joint natural error process.

## Seen-mechanism control and inference plan

For each seed, the full-folding control uses **exactly the partial-folding
selected words**, applying unchanged `WordNoise.fold` to all eligible characters
in each. It has a new input identity; old full-fold predictions are not
substitutes. Word exposure is matched; character dose differs. A performance
difference cannot be attributed solely to whether a mechanism was seen in
training. Existing case/swap/combined results remain historical references;
do not rerun the complete old grid.

Later scoring must retain the original T4, batch16, FP32, SDPA/math, deterministic
algorithms, TF32-off, inference-seed586, four CPU threads and shared clean-length
batch ordering. Require a separately verified matching runtime and model-file/
tensor identities before loading/scoring; metadata checks here are not weight
or runtime readiness. A technical calibration may use validation inputs only.
Scoring remains disabled by default and is not performed by the preparer.

The primary contrast is augmented minus clean for each new mechanism; secondary
contrasts include augmented minus without-case-loss and each model's partial
minus matched-full-fold result. Report per-seed paired differences, means and
sample SD (`ddof=1`) across the three corruption seeds for **all four models**.
Those SDs do not measure training uncertainty. Clean anchors have one value.
Report own-anchor drops, PERIOD/COMMA/QUESTION_MARK precision/recall/F1/support,
four-class confusion matrices, and realized dose by cohort. The primary score
is the three punctuation-class macro-F1; retain O for false positives/negatives,
zero_division=0. No p-values, equal-dose novelty claims, global pooled robustness
score or post-score recipe selection is specified.

## Preparation, audits, freeze and stop point

Reusable code resides in Python; old noise/audit/tokenizer/inference files are
unchanged. The old corruption audit deliberately requires equal codepoint
length; a **separate transfer audit** validates the new insertion/deletion
semantics rather than weakening that guard. Preserve every original row ID,
word boundary/order and punctuation target. New words must be nonempty/NFC;
unchanged words, whitespace and punctuation are protected. Inspect edit logs,
per-class support, selected-position hashes, character dose and row exposure.

Retokenize with the pinned locally cached tokenizer and unchanged first-subword
alignment, ignoring continuations/padding with-100. Record actual tokenizer
file hashes. Any missing word or sequence>512 stops preparation; no truncation,
condition-specific filtering or budget reduction is allowed. Reuse the audited
clean encoding via an explicitly hash-bound read-only reference; do not silently
copy or regenerate the old39 conditions.

A real validation-prefix data-only probe may precede the final freeze; its
source IDs/limit/code hashes are recorded and it computes no model metrics.
After that, freeze the exact protocol/config/profile/code/model metadata/cohort
bindings **before generating the new test views**. Refuse completed-output
overwrites; retain failed preparation evidence and version material changes.
Saved complete inputs must be independently reconciled with raw targets and
edit semantics. Keep all text, encodings and receipts ignored; public outputs
are code/configuration and aggregate preparation reports only.

Successful preparation means audited inputs exist. GPU readiness, inference,
predictions and F1 results remain separate subsequent work. Preserve all
original experiments, corpus analyses, final PDFs and source snapshots.
