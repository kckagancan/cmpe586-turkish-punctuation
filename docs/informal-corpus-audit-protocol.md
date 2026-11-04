# Informal Turkish corpus audit: post-feedback extension

Specified **2026-11-04**, `informal-corpus-audit-v1`, after the original held-out
study and instructor feedback. This is a **data-only audit**, not a new model
experiment, training recipe, or claim that the original doses were empirically
calibrated. Existing results, models, word targets and frozen recipes are retained.
The machine-readable recipe is [the audit config](../configs/informal-corpus-audit.json).

## Source and format

Select MultiLexNorm monolingual Turkish after the documented
[candidate comparison](informal-corpus-selection.md). Pin author repository
commit `e58e946375299a04355fa2d7fa9e1c769234e68a`, using only
`data/tur/train.norm` (99,174 bytes). Verify the config's SHA-256. Do not read
the source's test file. Raw data stays ignored; per-language reuse terms are not
established here and no raw text is redistributed.

Read literal `original<TAB>normalized` token records, with empty lines as
sentence boundaries. Preserve trailing tabs and empty normalized cells; never
use generic whitespace splitting to parse columns. An empty target is a merge
continuation: flag both its preceding nonempty anchor and every continuation.
A target containing several whitespace-delimited words is a split. Exclude all
involved slots from single-word correction analysis, rather than treating a
merge anchor as an ordinary insertion. Report complex split/merge cases.
Reject malformed column counts, empty source tokens, orphan merge continuations
and non-NFC text; never repair or silently drop them.

## Denominators

Reconcile the pinned train file with 570 sentences and 6,443 source slots.
These are normalization-token slots, including punctuation, not directly the
original project's punctuation-free word denominator. Report all slots,
letter-bearing source/target slots, one-word corrected slots and pure-alphabetic
eligible pairs separately. Count unchanged/changed slots, structural boundary
exclusions, unique source/target forms, unique ordered pairs and duplicate rows.
The resource supplies normalization targets, not fine-grained error tags or our
four punctuation labels; do not invent those annotations.

## Operator representability, not model performance

The direction is normalized/corrected word to observed original word. Separate
unchanged pairs, changed pairs, boundary-changing exclusions and eligible
single-word pairs. Import the existing frozen `WordNoise` functions unchanged.
For each changed eligible pair, check exact reachability by case loss, full
Turkish folding, and every legal adjacent interior swap. Also enumerate all
eligible nonempty subsets in the frozen swap -> case -> fold order.

This is an upper bound on **word-level representability**: it ignores global
word budgets, row masks and hash-selected swap positions. It neither means the
training data contained that pair nor that a sampled historical noise plan would
produce it. Report overlapping single-family counts and a disjoint partition
into single-operator reachable, combination-only reachable, and unreachable.
No identity transformation counts as corruption coverage.

Repeat lexical diagnostics after Turkish-aware lowercasing **both** sides.
Keep exact and case-neutral views separate. This diagnostic does not mutate the
source or turn case-neutral comparison into natural case-loss frequencies.
Use unique pairs as a second weighting view, not as independent observations.

Describe possible new error shapes on case-neutral, changed, alphabetic words
in the frozen alphabet, with corrected length at least four: partial Turkish
folding, one internal character omission, character-run elongation, one character
substitution, and one character insertion. Shape counts are nonexclusive and do
not identify human intent. Report edit-distance and length-difference histograms.
Boundary-changing and punctuation-changing pairs are not silently included in
word-preserving transfer candidates. This audit generates no corrupted dataset.

## Casing and punctuation observations

Describe alphabetic source/target casing and punctuation glyph counts. Case-only
corrections must have identical Turkish lowercase form. Separate exact full-word
lowercasing from other casing corrections. Report their own finite-sample
denominators, including corrected words eligible for historical case loss.
Glyph counts can include non-sentence punctuation and are not restoration gold.

## Evidence and interpretation gates

Write a private per-slot trace of source row/word IDs, structural exclusions and
classification flags; no tweet text is needed in the trace. Retain source/config/
code/protocol hashes, runtime versions, raw-data receipt, aggregate JSON and
conservation checks under a new identity-bound ignored output directory. Refuse
overwriting an existing completed audit. Public files contain aggregates only.

Reconcile every row and slot, changed/unchanged partitions,
boundary exclusions, operator partitions and private trace counts. No model,
tokenizer, optimizer, training data, or old predictions are required.

The corpus's selection/preprocessing may support a **conditional error profile**,
not unbiased frequency estimates for all informal Turkish. Do not directly use
OOV-tag shares as 2/5/10% all-word budgets. Casing observability and trustworthy
punctuation-gold availability remain distinct questions. A later evaluation
requires its own frozen protocol; applying new noise to old test text yields a
supplemental stress study, not new independent held-out data.
