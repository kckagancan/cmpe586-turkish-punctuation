# Noise protocol: Turkish punctuation restoration

Specified 2026-10-18. Version **noise-v1**. Operators are implemented and all 38
validation variants passed the [data-only audit](noise-audit.md). The medium,
seed-586 model [pilot](noise-pilot.md) is complete. Executable values come
from [configs/noise.json](../configs/noise.json).

## 1. Research question and literature adaptation

Measure how orthographic corruption changes punctuation predictions while retaining
the **same intended words' punctuation targets**. Compare clean/augmented models on
identical corrupted inputs. Label preservation is an experimental assumption about
intended text; ambiguous spellings can still make restoration intrinsically harder.

| Source | Useful idea | Our adaptation |
| --- | --- | --- |
| [Belinkov & Bisk, ICLR 2018](https://arxiv.org/abs/1711.02173), §4.2 | Their swap condition preserves word ends and requires length ≥4. | One adjacent interior swap; preserve word boundaries and target positions. |
| [Karpukhin et al., W-NUT 2019](https://aclanthology.org/D19-5506/), §2 and Table 2 | They distinguish attempted noise from actual changed tokens. Their training mixture is 60% clean and 10% each deletion/insertion/substitution/swap, applied dynamically. | Report realized corruption explicitly. Our initial evaluation isolates transposition; their training probabilities are not our severity settings. |
| [Náplava et al., W-NUT 2021](https://aclanthology.org/2021.wnut-1.38/), §3 | Separate casing, diacritics and spelling aspects; estimate natural error profiles from correction corpora. | Isolate our three aspects and combine controlled plans. We have no calibrated Turkish user-error profile; our levels are synthetic experimental doses. |
| [Wang et al., TextFlint, ACL-IJCNLP 2021](https://aclanthology.org/2021.acl-demo.41/) | Task-specific transformations and compositions support robustness analysis. | Keep token-classification alignment as a task constraint. No toolkit installation is required. |

The following rates and rules are project decisions. Results from translation,
other languages or other tokenizers do not establish suitable Turkish restoration
rates or expected gains. We study these defined corruptions, not all informal writing.

## 2. Severity: match the changed-word budget

The denominator is **all original word positions in a clean-v1 split**, including
words that a particular operator cannot change. For N words, select exactly
`K = N * coverage_percent // 100` eligible positions for each single family.

| Level | Changed words / all words | K on current validation |
| --- | ---: | ---: |
| Mild | 2% | 3,583 |
| Medium | 5% | 8,958 |
| Heavy | 10% | 17,916 |

Integer rounding is downward by less than one word. Each selected single-family
operation must change its clean word; unsupported words stay unchanged. Sample over
the whole split, without forcing an edit into every record. If fewer than K eligible
words exist, fail and report infeasibility; never silently lower the requested dose.

**Why these levels?** Train-only text statistics provide a feasible range without
looking at noisy model scores. The case-loss pool is the narrowest at 13.35% of
words; a 10% maximum remains below that ceiling. The 2/5/10 sweep gives a small,
moderate and larger matched dose with a manageable evaluation grid. These names
are experimental labels, not measured frequencies of real Turkish mistakes.

Measured design statistics using the operators below, without model inference:

| Split | All word positions | Case-loss eligible | Turkish-fold eligible | Swap eligible |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,463,897 | 195,453 (13.35%) | 612,612 (41.85%) | 1,071,932 (73.22%) |
| Validation | 179,164 | 23,910 (13.35%) | 75,070 (41.90%) | 130,793 (73.00%) |

Train statistics guided the design; validation statistics only confirm feasibility.
No test text or noisy F1 was used to choose levels. Matching changed-word counts
controls exposure, not word identity, character-edit count or information loss:
each family has a different eligible pool. Report those differences when ranking
sensitivity; avoid claiming a universal intrinsic ordering of noise types.

## 3. Operators

### Case loss

Lowercase the entire selected word with Turkish `I → ı`, `İ → i` overrides, followed
by ordinary lowercase for remaining characters. Eligible means the result differs
from the clean word. Examples authored for this protocol: `IŞIK → ışık`,
`İzmir → izmir`. Preserve punctuation and whitespace. This applies one uniform
Turkish locale policy, including foreign names; no language identification is used.
The special I pairs follow [Unicode 17.0 SpecialCasing](https://www.unicode.org/Public/17.0.0/ucd/SpecialCasing.txt).
Require NFC input; unexpected decomposed forms stop the audit before scoring.

### Turkish character folding (diacritic loss)

Replace every listed letter in the selected word, preserving case:

```text
ç→c  ğ→g  ı→i  ö→o  ş→s  ü→u
Ç→C  Ğ→G  İ→I  Ö→O  Ş→S  Ü→U
```

Examples: `ışık → isik`, `Çocuk → Cocuk`. This explicit Turkish ASCII-style map
includes dotless ı; it is not a generic removal of Unicode combining marks.
Unlisted letters, including â/î/û and non-Turkish accents, remain unchanged.

### Typo: adjacent interior transposition

Require length ≥4 and letters only from the config's ASCII-plus-Turkish alphabet.
For zero-based left indices `1 <= i < len(word)-2`, retain pairs with unequal
characters. Choose one pair deterministically and swap it once. Preserve first and
last characters. For example, `gittim → gititm` is a valid outcome.

Digits, apostrophes, hyphens, short words and words without a nontrivial interior
pair are ineligible. Thus short question particles receive no direct typo edit;
their surrounding context can still change. This is a specific transposition study,
not a complete model of keyboard mistakes, deletions or insertions.

## 4. Reproducibility, composition and full-loss endpoints

Build family-specific eligibility and edit plans from the **clean source**.
Rank positions by the full SHA-256 digest of the compact UTF-8 JSON array:

```text
[version, seed, split, row_id, word_index, family, "select"]
```

Take the first K eligible ranks; ties use `(row_id, word_index)`. For swap location,
hash the same base fields with purpose `"swap_position"` plus each candidate left
index; choose the smallest digest, breaking ties by index. Exclude severity,
condition/model names, labels and predictions from these keys. The same word then
keeps its edit location and mild selections are prefixes of medium/heavy selections.
Store plans and config hashes; record Unicode/Python versions with generated data.

**Combined:** reuse the three single-family plans at the same level/seed. Apply
`typo_swap → case_loss → diacritic_loss`. Do not resample when an earlier operation
makes a later edit redundant or a later mapping masks a swap. Log per-stage changes,
overlaps, cancellations and final changes. A component's mask is comparable across
single/combined runs; final edits need not add arithmetically. Selection is nested
across severity, but final character distance need not be monotonic.

Combined has up to three times the single-family word budget. Interpret it as a
co-occurrence stress condition, with measured union coverage, not an equal-budget
test of interaction or synergy. It is not guaranteed to be wholly lowercase/ASCII.

Two additional deterministic endpoints capture common writing styles:

- `case_loss_full`: lowercase every eligible word.
- `diacritic_loss_full`: apply the Turkish map to every eligible word.

These have different realized coverage and sit outside the matched 2/5/10 grid.
They need one realization each. They are not presented as extra matched heavy doses.

## 5. Audit gate, evaluation and compute

Before any noisy inference, audit all intended variants: unchanged split/row IDs,
word order/count and tags; no empty words, whitespace edits or punctuation edits;
NFC assumptions; selected/eligible/actually changed words; changed records and
per-class exposure. Report stage/final coverage and character-position/length
changes, plus subwords per word, length percentiles/max, UNK and missing word IDs.
Inspect seeded examples from each family/level, including Turkish I pairs.

Retokenize and realign every variant. Preserve all clean-v1 evaluation rows and
targets. Any missing word or >512 input stops the condition before model scoring.
A required shared chunking policy would receive a new version and a new clean
reference measurement; never truncate or drop different rows by condition.

Use existing macro/micro/per-class P/R/F1 and confusion matrices. Report positive
degradation as `100 * (clean_F1 - noisy_F1)` percentage points, exact supports and
the realized exposure. Use paired row IDs/inputs across the published, clean
fine-tuned and augmented models. Error examples complement numerical differences.

Plan three corruption seeds **586/587/588** and report their mean/std; these quantify
noise-realization variability, not training-seed variability. Per model: 12 stochastic
conditions ×3 seeds + clean + two deterministic endpoints = **39 evaluations**.
At the measured ~10-minute CPU rate, that is roughly 6.4 CPU hours per model.

Stage execution: implement/audit first; inspect four medium conditions with seed
586; plan the full grid on Colab GPU after timing/memory checks. A compute-driven
reduction must be documented before the relevant noisy scores are inspected.
No model calls are requested by this document or config alone.

Optional input ablation: combined-medium minus one family, reusing the remaining
masks. It measures removal within the mixture, with changed coverage reported.
The separate [training protocol](training-protocol.md) specifies the clean/noisy
row-view mixture, matched initialization/update budgets and training ablations.
Its clean training runner and Colab notebook are implemented but unexecuted;
the train augmentation preparer is still planned. This evaluation protocol's
saved conditions and word-level rates are unchanged.

## 6. Versioning and next step

Raw data and existing clean outputs stay intact. Future generated text/plans stay
under Git-ignored `data/processed/noise-v1/`; only code/config/aggregate results are
published. Structural audit findings may justify a recorded protocol revision
before noisy scoring. Revisions after seeing noisy scores are labeled exploratory.
Test remains held out during development. Dataset provenance/license remain open.

Implementation: `src/turkish_punctuation/noise.py`, `noise_audit.py` and
`scripts/prepare_noise.py`. The command archives settings/sources and refuses to
overwrite an existing run. `configs/noise.json` also sets tokenizer batch size
and the number of local examples. The completed data-only audit required no
transformation, rate or alignment-policy revision.

The published-model medium/seed-586 pilot is complete. Its clean-length batching
decision was based on padding counts before noisy aggregate scores; a matched
clean control reproduced all original word predictions. See [noise-pilot.md](noise-pilot.md).
The matched clean-versus-augmented training plan is now specified in
[training-protocol.md](training-protocol.md) and `configs/training.json`.
Next: review the implemented clean training path and calibrate GPU time/memory,
then prepare augmented training views and broader severity/seed evaluation.
