# Published BERT: medium-noise validation pilot

Completed **2026-10-18**. The clean control and all four prespecified medium/seed-586
noise conditions finished. No fine-tuning or held-out test evaluation was performed.

## Protocol

- Published checkpoint/tokenizer: `uygarkurt/bert-restore-punctuation-turkish`,
  revision `dd8b9481ed3b3c05110d7e041ecbeb6a1c23b8b0`.
- Fixed clean-v1 validation: 14,453 records and 179,164 word targets. All targets,
  record IDs and per-class supports remain fixed across conditions.
- Case loss, Turkish folding, adjacent interior swap and combined: **medium,
  corruption seed 586**. Each single family changes exactly 8,958 words; combined
  changes 25,673 distinct words (14.3293%) using the same component masks.
- Primary metric: fixed PERIOD/COMMA/QUESTION_MARK macro-F1. Also report micro
  P/R/F1, per-class P/R/F1, supports, full confusion matrices and paired changes.
  O positions remain in the counts; zero denominators yield 0.
- Local CPU, batch size 16, four PyTorch threads, float32 checkpoint, eval/inference
  mode. Reference class values are never sent to the model.
- Before inference, every saved noisy input was reproduced by re-tokenizing the
  hashed word records. Clean predictions/metrics and raw validation provenance
  were also verified. No records are truncated or excluded after noise.

## Batching decision before noisy scores

The first source-order attempt was interrupted after 6,500 records of its first
noise condition, before any noisy aggregate score was computed or inspected.
Data-only padding analysis showed that one shared ordering based on **clean input
length**, with source row ID as tie-breaker, reduces padded-token work substantially:

| Input | Source-order padded tokens | Shared-length-order padded tokens |
| --- | ---: | ---: |
| Clean | 701,362 | 278,388 |
| Case | 702,151 | 289,351 |
| Turkish fold | 722,924 | 316,078 |
| Swap | 744,418 | 341,980 |
| Combined | 765,025 | 355,374 |

These are workload counts, not measured speedups. Every condition shares the same
record order and batch membership, derived without labels or noisy scores. Inputs
within each record remain unchanged. A new clean control is evaluated with the
same order, and noisy degradation is measured against that matched control.
Its predictions are also compared word by word with the original clean reference.
The interrupted v1 provenance is retained; completed measurements use run **v2**.
The clean control has completed: **all 179,164 word predictions are identical** to
the original reference, with macro-F1 0.8363658278 and micro-F1 0.8266510513.
Its prediction loop including output I/O took about 405 seconds. This is a timing
observation under the current CPU load, not a hardware speed benchmark.

## Results

All P/R/F1 values are percentages. Drop is positive clean-minus-noisy **percentage
points**. Coverage counts actually changed words, not transformation attempts.

| Condition | Changed words (%) | Macro P | Macro R | Macro F1 | F1 drop (pp) | Micro F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 0.00 | 86.05 | 81.77 | 83.64 | 0.00 | 82.67 |
| Case loss | 5.00 | 84.77 | 78.19 | 81.23 | 2.41 | 80.12 |
| Turkish fold | 5.00 | 86.25 | 81.33 | 83.43 | 0.20 | 82.56 |
| Swap | 5.00 | 85.85 | 80.70 | 82.90 | 0.73 | 82.08 |
| Combined | 14.33 | 84.76 | 76.78 | 80.34 | 3.29 | 79.36 |

Per-class cells are **precision / recall / F1**, in that order:

| Condition | Period P/R/F1 | Comma P/R/F1 | Question P/R/F1 |
| --- | ---: | ---: | ---: |
| Clean | 94.50 / 97.17 / 95.82 | 70.49 / 56.23 / 62.56 | 93.17 / 91.90 / 92.53 |
| Case loss | 94.84 / 92.96 / 93.89 | 66.86 / 53.98 / 59.73 | 92.61 / 87.64 / 90.06 |
| Turkish fold | 94.49 / 97.17 / 95.81 | 71.16 / 54.93 / 62.00 | 93.09 / 91.90 / 92.49 |
| Swap | 94.38 / 97.12 / 95.73 | 70.14 / 53.75 / 60.86 | 93.04 / 91.22 / 92.12 |
| Combined | 94.63 / 92.85 / 93.74 | 66.80 / 50.26 / 57.36 | 92.83 / 87.21 / 89.93 |

Supports stayed O 150,845; PERIOD 15,252; COMMA 11,894; QUESTION_MARK 1,173.
Noisy prediction loops took approximately 414/446/482/497 seconds in table order.
Including the clean recheck, successful inference loops totaled **37.4 minutes**;
this excludes preparation, analysis and the interrupted first attempt.

## Paired error analysis

### Class response

At medium/seed 586, case loss has completed with macro-F1 **81.2256%**, a
**2.4110 pp** decrease from clean. Period recall fell from 97.1741% to 92.9583%,
and question-mark recall from 91.9011% to 87.6385%. Comma F1 fell from 62.5573%
to 59.7320%. There were 1,153 newly wrong word predictions and 437 recovered
errors. These counts include O; their difference is a word-error change, not F1.

Turkish folding at the same word budget has completed with macro-F1 **83.4347%**,
a **0.2019 pp** decrease. Period and question true-positive counts were unchanged;
comma recall fell from 56.2300% to 54.9269% while its precision increased from
70.4890% to 71.1578%. There were 311 new errors and 301 recovered errors.
This illustrates why per-class P/R and paired changes complement the macro score.
These observations apply to this specific dose, seed and checkpoint.

Adjacent interior swaps have completed with macro-F1 **82.9044%**, a **0.7321 pp**
decrease. The per-class F1 drops are 0.0889 pp (period), 1.6977 pp (comma) and
0.4099 pp (question). There were 671 new errors and 425 recovered errors.
Despite greater fragmentation than case loss, the observed F1 decrease is smaller.
Wordpiece count alone therefore does not explain the ordering in this pilot.

Combined macro-F1 is **80.3439%**, a **3.2927 pp** decrease. Comma F1 falls to
57.3622%, a 5.1951 pp decrease. Its realized 14.33% coverage is larger than the
single-family 5% budget, so this does not isolate an interaction or synergy effect.

For case loss, period-to-comma confusions increase from **220 to 765**, while
period-to-O confusions increase from **159 to 254**. Among the 654 formerly correct
period predictions that become wrong, **551 become comma**, 97 become O and six
become question marks. Thus much of the period recall loss is a boundary-type
confusion rather than simply omitting all punctuation.

### Outcomes and context

Each word has one of five clean-to-noisy outcomes:

| Outcome | Clean prediction | Noisy prediction |
| --- | --- | --- |
| Both correct | Correct | Correct |
| New error | Correct | Wrong |
| Recovered error | Wrong | Correct |
| Changed error | Wrong | Different wrong class |
| Unchanged error | Wrong | Same wrong class |

These counts explain changes in all-class word errors; punctuation F1 additionally
depends on which classes receive false positives/negatives. Error examples keep
clean/noisy words, both predictions, references and changed input positions locally.
At most two source-order examples per outcome/reference class are retained from
the new/recovered/changed-error categories, with distinct rows in each group.
They are inspection examples rather than a representative sample. Sorting batches
does not change this source-order sampling policy.

| Condition | New errors | Recovered errors | Different wrong class | Net added word errors |
| --- | ---: | ---: | ---: | ---: |
| Case loss | 1,153 | 437 | 73 | 716 |
| Turkish fold | 311 | 301 | 15 | 10 |
| Swap | 671 | 425 | 29 | 246 |
| Combined | 1,923 | 907 | 93 | 1,016 |

The following describes the full new-error population, not only sampled examples:

| Condition | Error word edited | Error word unchanged, another word edited | Entire record unchanged |
| --- | ---: | ---: | ---: |
| Case loss | 148 | 1,005 | 0 |
| Turkish fold | 81 | 230 | 0 |
| Swap | 270 | 401 | 0 |
| Combined | 616 | 1,307 | 0 |

**87.16% of case-loss new errors occur at an unchanged word in an edited record.**
Of its 97 new period omissions, 76 occur at an unchanged word whose following word
was lowercased. Local inspection examples also show period-to-comma and
question-to-comma changes after a following word loses capitalization. These
observations are consistent with surrounding casing acting as a punctuation cue;
they do not establish the model's internal causal mechanism. Tokenization/position
changes can also propagate through contextual representations.

No predictions changed in completely unedited records, for any condition.
The very small aggregate folding drop also hides both newly wrong and recovered
predictions, illustrating why the headline F1 alone is insufficient error analysis.

## Artifacts and reproduction

```bash
python scripts/evaluate_noise.py --config configs/noise-evaluation.json
python scripts/analyze_noise.py
```

Local output: `artifacts/evaluations/published-bert-noise-pilot-v2/`.
`run.json` stores code/config/input hashes and original clean reference information.
`source_snapshot/` preserves executed project sources and configuration. Each
`conditions/<name>/` contains predictions, metrics/degradation/paired counts and
local error examples. `clean-anchor` is the recomputed clean control.
`summary.json` is written only after the whole run completes.

Saved predictions use evaluation order, with original `row_id` retained for joins.
A completed condition is published atomically. `--resume` checks unchanged inputs,
code and settings and validates completed output hashes before continuing.
The progress file is updated during execution; completed outputs are not overwritten.

`scripts/analyze_noise.py` verifies prediction/input hashes, reconstructs original
selected-word text from the first pre-edit log entry, and compares it with final
text. Its `analysis/analysis.json` contains reproducible context counts;
`comparison.md` contains generated comparison tables, and `analyze_noise.py` is the
executed analysis source copy. `--output-directory` permits a new analysis version
without rerunning inference. The analysis never changes model predictions.

After completion, all five metric dictionaries were independently recomputed from
saved word predictions and matched exactly. Output/source hashes and all row/target
pairings passed. Executed Python snapshots match the current evaluation code.

## Interpretation limits

This pilot uses one synthetic corruption realization per condition and one
published checkpoint on a filtered validation subset. The upstream checkpoint
may have used validation for selection. It is not an independent final-test result
or evidence that training augmentation helps. Combined has a larger changed-word
budget than a single family. Equal global word budgets also do not equalize edited
characters, class exposure or information loss. Dataset provenance/license remain
unresolved; raw texts and inspection contexts stay in Git-ignored local artifacts.
"Clean" means the unmodified released subset, not manually certified error-free
writing. At the 5% global dose, case loss affects about 37.47% of case-eligible
words, whereas Turkish folding affects about 11.93% of fold-eligible words.
The matched budget controls word exposure rather than equalizing feature removal.

After reviewing the pilot, plan the complete severity/seed evaluation and matched
clean-versus-augmented fine-tuning. Preserve the final test partition for the
planned held-out comparison.
