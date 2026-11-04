# Real informal-text evaluation: feasibility and next gate

Reviewed **2026-11-04**, after the completed
[MultiLexNorm training-split audit](informal-corpus-audit-results.md).
The user proposed directly evaluating the existing models on authentic informal
Turkish instead of relying solely on new synthetic perturbations. This is a
feasible research direction **conditional on suitable punctuation references**.
No new evaluation protocol has been frozen, no model has run, and no new score
or punctuation annotation is claimed here.

## What direct evaluation would establish

Keep the published reference and the existing clean, augmented, and case-loss
removal checkpoints fixed. In particular, our models were fine-tuned on the
original punctuation dataset with clean or synthetic-noisy views; they were
**not trained on MultiLexNorm**. No retraining is needed to measure their
transfer to authentic informal text.

A separate, suitably labeled corpus could measure whether the augmentation
advantage carries to naturally occurring spelling and casing variation. It
would complement the original controlled experiments. Natural text also changes
topic, syntax, vocabulary, sentence length and style, so its performance
difference cannot be attributed solely to one unseen noise operator. It also
does not by itself establish that particular error mechanisms never occurred
in model pretraining or earlier fine-tuning.

The instructor requested advance severity specification, empirical calibration
where possible, distinct enough train/test corruptions, and class support/F1.
Direct natural evaluation strengthens external validity but does not
retroactively calibrate the original 2/5/10% synthetic doses. The completed
corpus audit supplies separate evidence about operator coverage and conditional
error shapes. Both contributions should retain their actual chronology.

## Controlled transfer alternative and test reuse

The user also proposed applying the three corpus-supported shapes (elongation,
internal omission, partial Turkish folding) to our existing labeled test text.
This is a valid **supplemental controlled transfer evaluation** of fixed models.
It uses existing punctuation targets and requires no new natural-text labels or
retraining. These mechanisms differ from the explicit training-v1 augmentation
operators; no claim is made about their absence from encoder pretraining or the
published model's unknown training history.

Re-evaluation alone does not train the model on test data or invalidate the
original results. The new conditions contain changed inputs, but share original
texts and targets with the earlier study, so they are not an independently
sampled test corpus. Merely conducting the extension after earlier test scoring
does not establish adaptive bias. The risk is choosing mechanisms, severities,
checkpoints or reported outcomes to favor results seen on that test set.
[Dwork et al. (2015)](https://arxiv.org/abs/1506.02629) analyze adaptive holdout
reuse; we do not implement their reusable-holdout mechanism or claim its formal
guarantees.

For this project, the recommended immediate extension is therefore to specify
the three operators and their corpus-conditioned dose rationale **before new
scores**, preserve the old cohorts/models/targets and every original result,
then evaluate all specified conditions and report favorable and unfavorable
outcomes. The old 2/5/10% recipe stays frozen. Publish the chronology as a
post-feedback supplemental study; do not retrospectively call it preregistered
or claim natural-text performance. A separate human-validated natural set is
an optional additional contribution. This recommendation is not an execution
authorization or a completed protocol freeze.

## Verified label obstacle

| Resource inspected | What its reference supports | Consequence for punctuation F1 |
| --- | --- | --- |
| MultiLexNorm Turkish | Word-level lexical normalization, including casing; the task guidelines generally preserve punctuation | A corrected spelling is not an annotation of every missing/incorrect punctuation boundary |
| #Turki$hTweets | Misspelling correction and error categories; the paper describes replacing punctuation and emojis with spaces, except apostrophes and hashtags | Its correction column does not supply complete punctuation-restoration gold; the released raw column does not establish intended punctuation either |
| ITU Web Treebank | Manual normalization, morphology and dependency annotation | The annotation paper does not establish complete restoration gold for omitted punctuation; source availability/terms and exact layers need checking |
| Turkish-GEC resources | Human evaluation references exist alongside synthetic training data | Human grammatical correction alone does not verify complete punctuation coverage, raw-word alignment, or suitability for our four-label task |

These are findings of a bounded primary-source review, **not proof that no
suitable Turkish informal punctuation corpus exists**. No additional corpus
test content was read. No new large dataset was downloaded.

Sources: [MultiLexNorm task](https://noisy-text.github.io/2021/multi-lexnorm.html)
and [paper](https://aclanthology.org/2021.wnut-1.55/);
[#Turki$hTweets, Section 3.1](https://aclanthology.org/2020.findings-emnlp.374/);
[ITU annotation procedure](https://aclanthology.org/W15-1610/);
[Turkish-GEC author repository](https://github.com/asimokby/Turkish-GEC).

Treating the punctuation present in an uncorrected tweet as exhaustive gold
would reward reproducing its omissions. Model predictions, another punctuation
model's output, or an unreviewed LLM draft are not independent human gold.
An automatic-reference study would need an explicit silver-label claim and
could not be presented as definitive human-reference accuracy.

Creating our own annotation set is methodologically possible. Labels drafted
by the assistant are **LLM-generated/LLM-assisted**, even if produced one item at
a time; they are not human manual annotation. An unvalidated automatic-reference
evaluation must retain a silver-label interpretation. Human review can improve
the references; independent annotation on a declared subset and adjudication
would provide stronger reliability evidence. A review of a subset alone does
not make every unreviewed label human gold. Punctuation can admit multiple
reasonable readings, so annotation guidelines must handle ambiguity rather
than promise a unique answer from Turkish rules alone. Keep annotators blind
to the models' predictions and record instructions, assistant/model identity,
drafts, human edits and the final reference hash before scoring.
[Pangakis and Wolken (2024)](https://arxiv.org/abs/2409.09467) provide primary
evidence for task-dependent LLM annotation quality and human validation; that
study is not a validation of our Turkish punctuation labels.

## Concrete next step before scoring

For the optional direct natural evaluation, complete the following
**data/annotation feasibility check** before scoring:

1. Verify a candidate's annotation instructions and actual reference format,
   source identity, access/terms, and original-word alignment. A corpus being
   public or having a corrected-text column is insufficient. Preserve the
   already inspected MultiLexNorm train split as the mechanism-analysis source;
   its official test contents remain unread at this handoff.
2. If complete punctuation references are unavailable, design a small native
   Turkish human annotation study on separately reserved authentic text.
   Freeze selection and annotation rules before predictions are exposed.
   Annotate punctuation at the **original noisy word boundaries**, preserving
   spelling, casing, word order and IDs. The task is punctuation restoration,
   not lexical correction. Two independent annotators plus adjudication would
   strengthen reliability; a single-annotator or LLM-assisted alternative must
   disclose its actual process and limitations.
3. Specify handling of ambiguous boundaries, URLs, emojis, standalone marks,
   and marks outside O/PERIOD/COMMA/QUESTION_MARK before scoring. Do not silently
   force unsupported marks into PERIOD or treat uncertainty as O. Record all
   exclusions and class supports. Set any sample-size or rare-class coverage
   rule before inspecting model outputs; a targeted sample is not a population
   prevalence estimate.
4. Freeze source/annotation hashes, cohort membership, exact-input overlap
   audit, model identities, the existing first-subword/-100 alignment and
   retokenization policy, runtime checks, and reporting plan. Report per-class
   precision/recall/F1/support and confusion counts alongside the three-class
   macro-F1; do not hide zero or scarce QUESTION_MARK support. Compare augmented
   minus clean on identical words; label other arm comparisons secondary.
   Do not select checkpoints or adjust noise/annotation rules using these scores.

Raw-text punctuation annotation does **not** require a one-to-one lexical
normalization pair. The 354 boundary-changing slots excluded from the earlier
operator-coverage denominator are not an automatic exclusion from a future
direct raw-text evaluation. A paired noisy-versus-normalized analysis would
separately require verified word/target mapping; do not project labels through
splits/merges by simply zipping columns.

Actual fine-tuning overlap can be audited, but document independence and the
published checkpoint's historical exposure remain unresolved. A previously
unread source split alone does not prove either. Reusing already scored original
test text with new corruptions would instead be a **post hoc supplemental
controlled study**, not fresh independent text.

## Current stop point

The planned MultiLexNorm train audit is complete: 570 sentences / 6,443 slots;
source test content remains unread. The original experiments, frozen recipes,
archives and final-report PDFs are preserved. No natural-text punctuation gold,
new corruption set, transfer F1, runtime-readiness receipt or running job exists
from this step. A possible synthetic transfer matrix remains a design option,
not a frozen or executed experiment. Corpus-grounded synthetic evaluation stays
available if reliable natural references cannot be established; that fallback
must be described separately from real-noise evaluation.

The subsequent user-authorized [depth analysis](informal-corpus-depth-results.md)
is now complete. It adds opportunity denominators, per-message density,
character-amplitude/placement distributions, duplicate sensitivity and conditional
sentence-bootstrap stability. It supports the three-mechanism supplemental
direction plus a small matched seen-folding control, rather than arbitrary
identical mild/medium/heavy rules. Original scores and recipes remain intact;
the new evaluation recipe still needs its own implementation and freeze.
