# MultiLexNorm Turkish: depth-analysis specification

Specified **2026-11-04** after the first corpus audit, before executing this
additional analysis. This is a descriptive, exploratory **data-only** extension,
not a new model experiment or a claim of blind preregistration. Reuse the
pinned local Turkish **train** file and saved per-slot classification trace
from [audit-v1](informal-corpus-audit-protocol.md). Do not rerun reachability,
read the source test split, load a tokenizer/model, infer punctuation labels,
or modify the original study. [Machine-readable settings](../configs/informal-corpus-depth.json).

## Scope, identity and conservation

Verify the original source/config/code/trace/summary hashes and audit manifest.
Read the literal TSV through the unchanged parser; reconcile all 570 sentences,
6,443 slots, exact pair hashes, row/word IDs, boundaries and stored shape flags.
Reuse the prior flags rather than performing a second operator-reachability
audit. New measurements describe **elongation, one internal omission and partial
Turkish folding**, with corrected length >=4 and Turkish-aware case-neutral
comparison. These are deterministic shapes, not human intent/error tags.

Store a new immutable, source-bound run with aggregate summary, private slot and
sentence feature traces, code/config/protocol snapshots, runtime versions and
file manifest. Public output contains aggregates and scientific figures only.
Raw examples, usernames and individual tweet text are not published.

## Frequencies and denominators

Report each count against all source slots, all one-to-one slots, and all 570
sentences. These views are not interchangeable with punctuation-free words in
the original project. For mechanism opportunity rates, use boundary-free pairs
whose corrected and observed strings are both entirely within the frozen
Turkish/ASCII alphabet, with corrected length >=4. Unchanged eligible pairs
stay in the denominator. Partial folding additionally requires at least two
foldable corrected **character occurrences**, not two distinct letter types.
The observed-side restriction makes all rates explicitly conditional on a
word-preserving, analyzable pair; it is not a prevalence estimate for all Turkish.

Show occurrence weighting and exact/neutral unique-pair sensitivity separately.
The latter reweights repeated pairs, not independent observations. Preserve
short-word, nonalphabetic and structural exclusions as disjoint counted bins.
Report length strata [4,5], [6,8], [9,12], [13,infinity] without changing the
overall family denominator. Describe earlier reachable patterns on the same
eligible pool using the saved flags, plus a disjoint unchanged/full-fold/partial-
fold/other partition among multi-foldable pairs.

## Density, concentration and joint occurrence

For each family and sentence, retain the number of eligible words and hits.
Report the pooled opportunity rate, the equal-sentence mean among sentences
with opportunities, all such sentence-rate quantiles **including zeros**, and
the separately conditioned distribution among affected sentences. Sentences
with no opportunities are undefined, not silently zero-filled. Report zero
mass, affected-sentence fraction, hit-count histogram, maximum/top-five sentence
shares, leave-one-sentence-out rate range and repeated-pair concentration.
Show same-word family intersections, sentence membership combinations and
co-occurrence with earlier reachable patterns. A natural tweet can contain
multiple mechanisms even when the three word-level shapes are disjoint.

Descriptive quantiles use the empirical inverse CDF (`inverted_cdf`, nearest
rank) at the configured probabilities. They select observed values and may tie.
Resample 570 whole sentences with replacement **5,000** times using NumPy
PCG64, seed586, batches100; recompute pooled hit/opportunity and sentence-presence
rates using the same row draws for all families. Report the 2.5/97.5 percentiles
with linear interpolation. Count any undefined replicates explicitly.
These are conditional **resampling stability intervals**, not confidence that
this selected corpus represents the Turkish population; authors, conversations
and corpus selection are not independently randomized. Do not treat interval
endpoints as mild/heavy doses. No p-values or significance claims are planned.
[Quantile definitions](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html)
and [percentile-bootstrap procedure](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)
document the numerical conventions; the implementation resamples whole rows.

## Word-level intensity and placement

Keep **exposure** (how many words/messages are affected) distinct from
**amplitude** (how much one affected word changes).

- Elongation: extra characters, extra/corrected-length ratio, changed-run count,
  added characters per changed run, exact ordered extra-count patterns,
  corrected length, repeated-letter counts and run location. Retain multi-run
  and extreme cases; do not cap the tail to match a preferred experiment.
- Omission: all legal matching interior deletion indices. Report ambiguous
  versus unique alignments; characterize relative deletion position only on
  uniquely aligned cases, and report the removed letter (which is invariant
  within an ambiguous repeated-character run). Record one/length intensity.
  Do not invent several deletions per word when the detector specifies one.
- Partial folding: affected/available foldable character counts, their joint
  histogram and fraction, remaining foldable occurrences, corrected length,
  character mappings and changed-position counts. Compare opportunities with
  fully folded and unchanged pairs; neither actual corpus event counts nor
  different eligibility denominators establish equal dose between mechanisms.

Also record exact versus case-neutral edit distance for each family; these are
surface differences, not unique linguistic edit histories. Intensity bins are
descriptive lower (<=P25), middle (P25,P75], upper (>P75), applied to extra chars,
omission 1/length and folded fraction respectively. Publish thresholds and
actual counts, including tied/empty bins. They are not human-annotated severity
classes and do not guarantee three useful experimental levels.

## From findings to a later experiment

This step produces calibration evidence and candidate anchors, **not a frozen
test recipe or generated noise**. The later protocol must declare selection
eligibility, original-word denominator, dose, per-word amplitude/placement and
how clustering is retained or simplified. Pooled finite-corpus rates, positive-
sentence burst rates and character-intensity quantiles answer different questions.
Do not replace one by another or rename an arbitrary multiplier a measured
natural severity. Sparse-event quantiles can be zero or identical.

The new primary question is fixed-model transfer to the three new mechanisms.
Retain a fresh clean anchor and all completed historical results. Consider a
small explicitly matched **seen-mechanism control**, especially full folding
on the same selected words as partial folding, before scoring. Such a pair holds
word exposure fixed while character dose differs and must be reported. It does
not require rerunning the complete old grid or rewriting its recipe. No new
training is needed. New corruptions on old test text remain a supplemental study
with shared sentences/targets, not fresh independent natural-text evaluation.
