# Published BERT: clean validation reference

First reference measurement, completed 2026-10-18 (Europe/Istanbul).
No fine-tuning, noise generation or test-set evaluation was performed.

## Protocol

- Checkpoint: `uygarkurt/bert-restore-punctuation-turkish`, revision
  `dd8b9481ed3b3c05110d7e041ecbeb6a1c23b8b0`.
- Data: pinned Kurt & Çayır release, original validation partition only.
- `clean-v1` keeps **14,453 / 14,514 records** and **179,164 word targets**.
  The 61 records containing empty words remain in local quarantine. Their 1,162
  targets include 26 question marks; the eligible subset has 1,173 question marks.
- Use saved, untruncated inputs, first-subword supervision and dynamic padding.
  Ignore `-100` positions. Preserve source row order and check every reference
  sequence against the raw validation record before scoring.
- Primary score: **macro-F1 over PERIOD, COMMA and QUESTION_MARK**, equal weight
  per class. Macro precision and recall are also separate class averages.
- Secondary: punctuation micro-P/R/F1, per-class scores/supports, and the complete
  four-class confusion matrix. Keep O positions when counting errors; exclude O
  only from the punctuation averages. A mark inserted on an O word is an FP.
- Zero-denominator convention: 0. All three punctuation classes have support here.
- Local CPU, batch size 16, four PyTorch threads, eval/inference mode. The prediction
  loop including output I/O took about 588 seconds. This is not a speed benchmark.

## Results

All P/R/F1 values in this table are percentages. Supports count original words,
not subwords or padded positions.

| Class / average | Precision | Recall | F1 | Reference support |
| --- | ---: | ---: | ---: | ---: |
| Period | 94.50 | 97.17 | 95.82 | 15,252 |
| Comma | 70.49 | 56.23 | 62.56 | 11,894 |
| Question mark | 93.17 | 91.90 | 92.53 | 1,173 |
| **Punctuation macro (primary)** | **86.05** | **81.77** | **83.64** | 28,319 |
| Punctuation micro | 85.79 | 79.76 | 82.67 | 28,319 |

Diagnostic all-class word accuracy is 95.04%. For context, always predicting O
would obtain 84.19% accuracy and zero punctuation F1 on these same references.
This illustrates why accuracy is not the primary score.

### Confusion matrix

Rows are reference labels; columns are predictions. Counts include all 179,164
words, with 150,845 O references.

| Reference / prediction | O | Period | Comma | Question |
| --- | ---: | ---: | ---: | ---: |
| O | 147,688 | 568 | 2,572 | 17 |
| Period | 159 | 14,821 | 220 | 52 |
| Comma | 4,969 | 227 | 6,688 | 10 |
| Question | 20 | 67 | 8 | 1,078 |

Comma is the weakest class in this measurement. Its 5,206 false negatives include
4,969 omissions (predicted O); its 2,800 false positives include 2,572 insertions on
O references. This identifies an error pattern. It does not yet establish why
those contexts are difficult or predict which noise type will hurt most.

Twelve error examples were saved locally: the first mismatches in source order,
up to three distinct rows per reference class, including O insertion errors.
These are inspection examples, not a random or representative error sample.
Raw texts are not included in this public aggregate report.

## Artifacts and reproducibility

Run: `python scripts/evaluate.py`, using `configs/evaluation.json` and the local
prepared data/model cache. A pre-existing output directory is never overwritten.

Local Git-ignored directory:
`artifacts/evaluations/published-bert-clean-validation-v1/`

- `metrics.json`: exact scores, confusion matrix, settings, dataset/prediction
  hashes, package versions, Git commit and dirty flag.
- `predictions.jsonl`: source row ID and word-level reference/prediction class IDs.
- `error_examples.json`: the twelve selected contexts with aligned references
  and predictions.
- `source_snapshot/`: eight executed code/configuration files, retained because
  the run used an uncommitted working tree. Recorded source hashes refer to these
  archived versions. Later working-tree typing/provenance refinements do not
  change the saved predictions; recomputing metrics from them gives identical scores.

## Interpretation limits and next step

This is the published reference on a filtered validation subset. It is not a
reproduction of the paper's full-test scores. Upstream checkpoint selection may
have used the released validation partition; its independence for model selection
is not established. Dataset license/provenance questions remain open.

Next: define reproducible word-preserving noise rates and transformations; audit
their word coverage/subword lengths; then measure validation robustness on these
same eligible row IDs. Keep the held-out test partition for the planned final
comparisons. The controlled augmentation claim will come from matched clean versus
noise-augmented fine-tuning, not this published checkpoint alone.
