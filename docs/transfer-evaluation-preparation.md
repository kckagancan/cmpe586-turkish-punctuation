# Supplemental transfer inputs: preparation outcome

Prepared **2026-11-04** under the [fixed protocol](transfer-evaluation-protocol.md).
All **13 input views** are audited: one existing clean encoding reused through
its exact read-only hash reference, plus12 new noisy views. **No model weights
were loaded, no F1 was computed, and no GPU readiness was established.**
The later fixed-model evaluation has52 forwards /104 cohort score sets.
The original experiments, corpus analyses and public final PDF are preserved.

## Scientific scope and corpus anchors

The three separate mechanisms are elongation, one internal character omission,
and partial Turkish folding. A fourth condition applies the original full-fold
operator to exactly the same words as partial folding. Three corruption seeds
(586/587/588) retain one empirical anchor per family; there is no new severity
search, combined condition, checkpoint selection or training.

The completed MultiLexNorm Turkish **train-only** findings supply conditional
rates89/4512,50/4174,43/621. The new surface-preserving partial-fold implementation
excludes four casing-incompatible source opportunities and no hits, so its
separate operational anchor is43/617. This does not replace the earlier43/621
analysis, nor denote the initial min4 pool which also happened to have617 items.
The compiled profile contains anonymous aggregate morphology templates rather
than source words. Donor conditioning, position mapping and unsupported target
foldable-count extrapolation are explicit in the protocol and aggregate audit.

Uniform whole-split word sampling transports these finite-corpus rates; it
does not reproduce natural message clustering. Partial/full folding has matched
word exposure but unequal character dose. New corruptions reuse the previously
scored test text: this is supplemental controlled transfer, not an independent
natural-text test or a blind preregistration. The natural source test remains
unread; no natural punctuation gold is claimed.

## Realized primary input coverage

Each view retains14,600 rows /181,485 original word targets. The sensitivity
view filters the same primary records/edits to14,445 rows /181,116 targets;
it does not rerank positions or calculate a new corruption budget.

| Input view | Changed words | Changed rows | Character edits | Max sequence length |
| --- | ---: | ---: | ---: | ---: |
| clean-anchor | 0 | 0 | 0 | 153 |
| elongation-corpus-seed586 | 3,355 | 2,899 | 9,766 | 154 |
| internal_omission-corpus-seed586 | 1,826 | 1,675 | 1,826 | 153 |
| partial_folding-corpus-seed586 | 1,803 | 1,681 | 2,171 | 153 |
| full_folding_matched-corpus-seed586 | 1,803 | 1,681 | 4,762 | 153 |
| elongation-corpus-seed587 | 3,355 | 2,875 | 10,104 | 153 |
| internal_omission-corpus-seed587 | 1,826 | 1,682 | 1,826 | 153 |
| partial_folding-corpus-seed587 | 1,803 | 1,642 | 2,188 | 154 |
| full_folding_matched-corpus-seed587 | 1,803 | 1,642 | 4,772 | 153 |
| elongation-corpus-seed588 | 3,355 | 2,890 | 9,609 | 153 |
| internal_omission-corpus-seed588 | 1,826 | 1,686 | 1,826 | 153 |
| partial_folding-corpus-seed588 | 1,803 | 1,656 | 2,140 | 153 |
| full_folding_matched-corpus-seed588 | 1,803 | 1,656 | 4,777 | 153 |

Character edits count declared insertions, deletions or folded occurrences;
the full [aggregate JSON](transfer-evaluation-input-audit.json) records amplitude,
donor, placement, conditioning/extrapolation and label-exposure histograms.
The [coverage CSV](transfer-evaluation-input-coverage.csv) includes both cohorts.
Sequence lengths/unknown counts in that CSV are for primary encodings only.

## Audits actually performed

1. Data-only validation probe: source prefix512 rows,510 eligible under the
   unchanged whole-row quarantine;13 views, no model metrics. All first-subword
   targets were preserved; maximum93 tokens. No recipe was chosen from F1.
2. Exact code/config/profile/model-metadata/cohort freeze **before** generating
   new test views. Freeze identity: `b16cd070f37073b76503267668e27761922708eaa866ac3a80aab3fb790905d1`.
3. Full-source audit: unchanged row IDs/order, word counts, tags, nonempty NFC
   output, unselected-word protection, exact floor budgets and SHA-ranked masks.
   Separate insertion/deletion/folding shape checks preserve the old equal-length
   noise auditor. Partial and full-fold controls share identical selected masks.
4. Every new view was retokenized with the pinned cached fast tokenizer and
   unchanged alignment. No missing word, truncation, condition-specific drop or
   sequence over512 was accepted. Maximum realized length: **154**.
5. Completed files were read back: hashes/file membership, raw targets, edit
   semantics, deterministic donor parameters and independently reranked masks.
   Fresh tokenization of all saved noisy views and the reused clean reference
   matched every saved column/label and audit. No test suite was added or run.

Input identity: `235c0c01cf8b5a34b24e36f1bbdcfe8603c34e401171dd47ec9856c87a30b7a3`. Private artifacts are under
`data/processed/transfer-noise-v1/` and
`artifacts/evaluation-preparation/transfer-noise-v1/`. Source snapshots and
complete-file manifests bind the inputs. The freeze's historical status string
records its pre-generation stage; this report records completed preparation.

## Remaining action

Implement the separate orchestration/runtime gate, verify retained model files
and tensors, and perform validation-only GPU calibration on the pinned T4
runtime before the52-forward evaluation. Scoring remains disabled by default.
Report per-class support/precision/recall/F1 and paired contrasts for all outcomes;
seed SD measures corruption variation only, with one training seed unchanged.
No job is running after this preparation. No new result has been folded into
the original final paper or PDF.
