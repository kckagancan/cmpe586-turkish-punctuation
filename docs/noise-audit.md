# Validation noise data audit

Completed **2026-10-18**, protocol `noise-v1`. This is a data/tokenizer audit:
**no model weights were loaded, no predictions or noisy F1 scores were produced,
and no training or test-text processing was performed.**

## Scope and provenance

- Fixed `clean-v1` validation subset: **14,453 records, 179,164 word targets**.
  The same 61 raw validation records remain quarantined under the existing policy.
- Prepared 3 single families + combined, at 3 levels and seeds 586/587/588:
  **36 stochastic variants**, plus two deterministic full-loss endpoints.
- Audited **6,808,232 noisy word-position instances** across 38 variants.
- Pinned release/tokenizer, source hashes and every executed source/config snapshot
  are recorded locally in `data/processed/noise-v1/manifest.json` and
  `source_snapshot/`. Preparation took about **244 seconds** after tokenizer loading;
  generated output occupies about **347 MiB**. This is not model inference timing.
- Re-encoding clean validation reproduced every stored clean-v1 model-input column,
  including row IDs and aligned labels, exactly.

## Integrity results

All variants passed:

- Same row IDs/order, word counts, label sequences and class supports.
- No empty output words, whitespace/punctuation changes or non-NFC words.
- All v1 operators preserved word codepoint lengths.
- Single-family selected and actually changed word counts matched the exact floor
  budgets: **3,583 / 8,958 / 17,916** for mild/medium/heavy at every seed.
- Observed selection hashes matched the clean-source plans. Severity masks were
  nested; combined component masks matched the corresponding single-family masks.
- No missing word IDs, over-limit records, truncation or condition-specific exclusions.
- **Zero unknown subword tokens** in clean and all noisy variants.

Class supports remained O **150,845**, PERIOD **15,252**, COMMA **11,894** and
QUESTION_MARK **1,173** in every variant. The maximum input length over the full
grid was **218**, including special tokens, below the model limit of 512.

## Corruption exposure and tokenization

Stochastic rows below average the three corruption seeds. Maximum length is the
largest observed at any of those seeds. Percentages refer to actually changed
words divided by **all 179,164 original word positions**; rounding is for display.

| Condition | Changed words (%) | Mean subwords / word | Max input length |
| --- | ---: | ---: | ---: |
| Clean | 0.00 | 1.3858 | 196 |
| Case mild | 2.00 | 1.3857 | 197 |
| Case medium | 5.00 | 1.3855 | 196 |
| Case heavy | 10.00 | 1.3849 | 187 |
| Turkish-fold mild | 2.00 | 1.4034 | 197 |
| Turkish-fold medium | 5.00 | 1.4300 | 199 |
| Turkish-fold heavy | 10.00 | 1.4744 | 204 |
| Swap mild | 2.00 | 1.4219 | 199 |
| Swap medium | 5.00 | 1.4765 | 204 |
| Swap heavy | 10.00 | 1.5671 | 211 |
| Combined mild | 5.90 | 1.4390 | 197 |
| Combined medium | 14.34 | 1.5174 | 200 |
| Combined heavy | 27.35 | 1.6427 | 203 |
| Full case loss | 13.35 | 1.3848 | 181 |
| Full Turkish fold | 41.90 | 1.7569 | 218 |

Combined final changed-word counts by seed were 10,560/10,553/10,588 at mild,
25,673/25,698/25,705 at medium and 49,002/48,973/49,049 at heavy.
The component pools overlap, so final coverage is below the sum of their budgets.
Stage redundancies and final cancellations are separately recorded in each local
audit; selected edits are never replaced to compensate for these effects.

Case loss slightly reduced average fragmentation; swaps and Turkish folding
increased it. These are tokenizer observations. They do not establish which noise
will reduce punctuation F1 most; casing cues can matter even when word pieces shrink.

## Direct exposure differs across punctuation classes

For **medium, seed 586**, percent of words changed *within each reference class*:

| Condition | O | PERIOD | COMMA | QUESTION_MARK |
| --- | ---: | ---: | ---: | ---: |
| Case loss | 5.33 | 1.33 | 5.85 | 1.11 |
| Turkish fold | 4.80 | 6.50 | 5.71 | 3.84 |
| Swap | 4.83 | 6.26 | 5.60 | 4.94 |
| Combined | 14.29 | 13.53 | 16.30 | 9.72 |

Labels were used only to describe exposure after selection. They did not determine
which words were selected. A label belongs to the word immediately before its
punctuation position. The table measures direct changes to those words; surrounding
corruptions can also affect a contextual model's prediction at an unchanged word.
Equal global word budgets therefore do not equalize class exposure or information loss.

## Local files and reproduction

Run from the project root after clean-v1 preparation, using the cached tokenizer:

```bash
python scripts/prepare_noise.py --config configs/noise.json
```

The completed directory already exists; this command refuses to overwrite it.
For another run, copy the JSON, set a new output directory and pass it via `--config`.
Unchanged version/seed/source preserve the ranking; changed rates choose different
prefix lengths. Every run archives its exact config bytes and relevant source files.
`noise_config_sha256` refers to those original bytes in
`source_snapshot/noise-config.json`; the root `noise-config.json` is a formatted
copy with identical parsed settings, whose whitespace can differ.

Under each `conditions/<name>/`: `records.jsonl`, `edits.jsonl`, `examples.json`,
`audit.json` and a Hugging Face `dataset/`. The clean encoded reference is stored
under `clean/dataset/`. All 282 generated files remain Git-ignored; dataset text
and examples are not redistributed. This report contains aggregate observations.

The source snapshot captures the protocol's status at run start. This completion
summary and the protocol's completion note were written afterward; transformation
rules, rates, seeds and executed Python sources were not changed after this run.
Ruff reported no issues for the three new Python files.

## Next step

Review the operators, selection and audit code, then connect the saved variants
to a published-model pilot: case/fold/swap/combined, medium, seed 586. There is no
length/coverage finding requiring a noise-v1 rule change. Optional drop-one input
ablations, training mixtures and controlled fine-tuning remain future stages.
