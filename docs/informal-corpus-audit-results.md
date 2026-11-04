# Informal Turkish corpus audit: verified data-only results

Completed **2026-11-04**, after instructor feedback. The
[candidate comparison](informal-corpus-selection.md) selected **MultiLexNorm,
monolingual Turkish, official training split**. This adds empirical evidence
about writing-noise coverage; it replaces no original experiment or report.
No model, tokenizer, training, inference, or source test file was used.

[Protocol](informal-corpus-audit-protocol.md) ·
[Config](../configs/informal-corpus-audit.json) ·
[Complete aggregate JSON](informal-corpus-audit-summary.json).

## Source and denominators

- Pinned author repository commit: `e58e946375299a04355fa2d7fa9e1c769234e68a`.
- `data/tur/train.norm`: **570 sentences / 6,443 source token slots**,
  99,174 bytes; SHA-256
  `17793b8db8f41ae387cfcdf4b7a1c061f2ccdc677418b3c2cec912c4e570695d`.
- 5,803 original slots contain letters. These normalization slots
  include standalone punctuation and are not automatically our original
  punctuation-free word denominator.
- 6,089 slots retain one source/one corrected word; 5,453 of
  those corrected words contain letters, and 4,557 pairs are purely alphabetic
  within the frozen Turkish/ASCII alphabet. No duplicate sentence excess.
- 2,385 exact source/correction differences: **37.0169% of all slots**.
  This agrees with the shared-task paper's rounded train-table figure; it is a
  property of this selected resource, not a population-wide Turkish error rate.

The source's test file remains unread. Source selection/domain, annotation
scope and unresolved per-language terms limit reuse and generalization. Raw
text is retained only in ignored local storage; published outputs are aggregates.

## Boundary changes are explicitly outside the word-preserving comparison

| Disjoint structural exclusion | Source slots |
| --- | ---: |
| Split only | 191 |
| Merge anchor only | 52 |
| Merge continuation | 109 |
| Both split and merge anchor | 2 |
| Total | **354** |

These slots are counted in the full-source audit, not silently dropped. In
particular, a merge anchor is not misclassified as a one-word insertion. The
remaining **2,031 changed one-word pairs** define the exact coverage denominator.

## Coverage of the original operators

The direction is **corrected -> observed**. Exact reachability uses the existing
frozen Turkish-aware lowercasing, full-word Turkish folding and adjacent interior
swap functions. We also enumerate eligible subsets in their frozen order.
This is a word-level **upper bound** over possible operations. It ignores row
masks, hash-selected swap locations, budgets and actual training exposure; it
is neither model accuracy nor evidence that training generated those pairs.

| View / weighting | Changed | Boundary excluded | One-word changed | Reachable | Reachable / one-word changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| exact / occurrence_weighted | 2,385 | 354 | 2,031 | 1,109 | 54.6036% |
| exact / unique_pair_weighted | 1,986 | 269 | 1,717 | 893 | 52.0093% |
| case_neutral / occurrence_weighted | 1,958 | 354 | 1,604 | 872 | 54.3641% |
| case_neutral / unique_pair_weighted | 1,547 | 250 | 1,297 | 667 | 51.4264% |

For exact occurrence weighting, **1,075** pairs are reachable by a single
operator, **34** additional pairs only by combinations, and **922** are
unreachable. Single-family membership is case loss **263**, full folding **809**,
and internal swap **3**. Subset memberships overlap; do not add the raw
combination-membership counters to get a disjoint total.

The unique-pair view reduces weighting of repeated pairs. It is a descriptive
reweighting, not an independent sample. Case-neutral results lowercase both
sides using Turkish rules to separate lexical spelling from casing differences;
this changes no stored input or historical recipe.

## Empirical shapes not covered by the old operators

The following are deterministic, nonexclusive **case-neutral** shapes among
unreachable changes, on alphabetic pairs with corrected length at least four.
They are our classifications of correction pairs, not human-provided error-type
labels and not proof of the writer's intention.

| Shape | Occurrences |
| --- | ---: |
| character_run_elongation | 55 |
| one_character_insertion | 28 |
| one_character_substitution | 123 |
| one_internal_character_omission | 49 |
| partial_turkish_folding | 42 |

The strongest compact next-scope candidates are **character-run elongation
(55)**, **one internal omission (49)** and **partial Turkish folding (42)**.
Elongation may express emphasis; a run-length change does not establish intent.
Partial folding preserves some Turkish letters while changing others, unlike
our old full-word folding operator. Counts are conditional on the stated
eligibility, exclude shorter words, and must not be added as an exhaustive error
partition. No new corruption dataset or transfer score was generated.

## Casing is observable, with a conditional denominator

Of **1,323** one-word corrected slots eligible for the old case-loss operator,
**263** observed forms exactly equal its Turkish lowercase output
(**19.8791% of those eligible slots**).
There are **427** case-only changes in total; the other **164** do not match
full-word lowercasing. This supports studying additional casing patterns; it
is not a reason to present a single natural case-error rate for all Turkish.
The old 2/5/10% all-word levels remain controlled stress doses.

Punctuation retained in a normalization resource is not complete intended
punctuation gold. The aggregate file also shows orthographic punctuation
changes (for example apostrophe-glyph totals differ). The shared-task guideline
to keep punctuation is not an invariant guaranteeing every character is unchanged
in the released Turkish correction pairs. We infer no punctuation labels.

## Reproducibility, preservation, and next step

Audit identity: `31b1594e88c80d0607436a008a7b1a6e9bd74c67452eb73dcf0424a69a56aa7a`.
Private receipts/trace/snapshots:
`artifacts/corpus-audit/multilexnorm-turkish-v1/audit-31b1594e88c80d06/`.
An independent literal-TSV count and ID/hash reconciliation verified all
6,443 trace records, every view partition and all 10 identity-bound audit files.
The 109 pre-existing scientific/config/report files bound before this task,
including the final report PDFs, remain byte-identical. Only data/metadata
inspection and new corpus-audit code ran; no test suite or model ran.

The next step is a **separate supplemental transfer-evaluation protocol** using
fixed existing checkpoints and audit-supported unseen mechanisms. Derive
eligibility/dose rationales from declared corpus counters, not new model scores;
keep every original recipe and result. New perturbations of already scored test
text remain a post hoc supplemental study. Independent confirmation still needs
genuinely unused text and trustworthy punctuation targets. No inference is
started or authorized solely by this document.
