# Initial dataset audit — 2026-10-17

Status: the released CSV files are downloaded locally and structurally audited.
Dataset licensing and exact corpus/split provenance remain unresolved. The original
audit below records the pre-preprocessing findings; tokenizer diagnostics and the
subsequent clean-v1 modeling policy are documented in later sections. The later
published-model clean-validation measurement is recorded in
[clean-validation.md](clean-validation.md); training, noise augmentation and test
evaluation have not been performed.

## Release and reproducibility

Source: [Kurt & Çayır (2023), official repository](https://github.com/uygarkurt/Turkish-Punctuation-Restoration).
The inspected revision is `7ede7a4ba9e51dad085a7f0d9b6c0a9c32622397`.
Exact URLs, byte sizes and SHA-256 checksums are in [data-manifest.json](data-manifest.json).
Only the three CSV files were downloaded into `data/raw/kurt_cayir_2023/`.
Their combined size is 31,543,691 bytes, about 31.5 MB (30.1 MiB).

```bash
conda activate cmpe586
python scripts/download_data.py
python scripts/inspect_data.py
```

The downloader verifies existing files and refuses to silently replace a mismatch.
The inspector verifies file hashes, parses the complete release and writes aggregate
counts to [data-audit.json](data-audit.json). Raw files remain unchanged and Git-ignored.
No upstream Python code was executed or incorporated into this project.

## Format

Each CSV has `sentence` and `tags` columns. Each value is a serialized Python list;
use `ast.literal_eval`, never `eval`. The token and label lists have equal length in
every released row. Each label specifies the punctuation **after** that token:

| Label | Meaning |
| --- | --- |
| `O` | No punctuation |
| `PERIOD` | `.` |
| `COMMA` | `,` |
| `QUESTION_MARK` | `?` |

Illustrative example invented for this project (not copied from the dataset):

```text
tokens = ["Bugün", "geldin", "mi", "Yarın", "konuşuruz"]
tags   = ["O",     "O",      "QUESTION_MARK", "O", "PERIOD"]
```

This represents “Bugün geldin mi? Yarın konuşuruz.” A CSV row can contain multiple
sentences; row boundaries must not be treated as reliable sentence boundaries.
The upstream [loader and alignment code](https://github.com/uygarkurt/Turkish-Punctuation-Restoration/blob/7ede7a4ba9e51dad085a7f0d9b6c0a9c32622397/utils.py)
attaches each word's label to its first subword and ignores remaining subwords and
special tokens with `-100`. This convention needs an explicit implementation in our
future loader, including checks for words that produce no subwords.

## Observed counts

These are counts from the downloaded files, not model results. Token entries include
empty-string entries; they are not tokenizer subwords.

| Split | CSV rows | Token entries | `O` | Period | Comma | Question mark |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 117,431 | 1,471,806 | 1,238,979 | 124,817 | 98,194 | 9,816 |
| Validation | 14,514 | 180,326 | 151,841 | 15,306 | 11,980 | 1,199 |
| Test | 14,661 | 182,487 | 153,466 | 15,524 | 12,242 | 1,255 |

Totals: 146,606 rows and 1,834,619 token entries. Published token and punctuation
counts in the official README match these files exactly.

- Capitalization and Turkish diacritics are preserved, making case-loss and
  diacritic-removal transformations technically applicable.
- Question marks are rare: roughly 0.67–0.69% of token entries. Per-class metrics
  and class support will matter.
- Maximum row lengths are 97/92/87 token entries for train/validation/test.
  This does **not** guarantee that a row fits into a 128-subword model window,
  particularly after character noise. Subword lengths have not been measured.
- There are no nonempty tokens containing whitespace, and no unknown label names.

## Empty tokens require an explicit policy

| Split | Empty token entries | Affected rows | Period labels on empty tokens | Comma | Question mark |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 511 | 462 | 86 | 270 | 155 |
| Validation | 68 | 61 | 10 | 36 | 22 |
| Test | 72 | 61 | 12 | 35 | 25 |

There are 651 empty-string token entries in 584 rows. All of these entries carry
a punctuation label. Silently deleting them would also remove target punctuation;
merging their labels onto neighboring tokens could create conflicting marks.
The raw files are preserved. Before modeling, inspect their context and measure how
the selected tokenizer handles them. Choose and document one shared policy for all
model comparisons, and report the affected counts. No policy has been applied yet.

## Repeats and split provenance

Exact input-sequence overlaps, comparing the released tokens with casing preserved:

| Split pair | Shared unique inputs | Shared inputs of at least 10 tokens | Shared inputs with identical labels |
| --- | ---: | ---: | ---: |
| Train / validation | 66 | 1 | 64 |
| Train / test | 78 | 0 | 70 |
| Validation / test | 26 | 0 | 24 |

148 test rows (1.01%) have an identical input somewhere in train. The train/test
overlaps are all short, so these counts alone do not establish document leakage.
Within train there are 941 repeated input rows beyond the first occurrence and
32 unique inputs with multiple label sequences. Full counts are in the JSON audit.

The CSVs contain no talk IDs, document IDs or split-construction metadata. Thus
document-level independence is not verified. Keep the released splits intact for
now. Before final experiments, decide whether to add a clearly reported sensitivity
analysis excluding repeated test inputs; do not silently replace the benchmark split.

## Source and license findings

1. The release repository has no license file or explicit dataset license in the
   inspected tree/README; GitHub's repository metadata returns `license: null`.
2. The [BERT model card](https://huggingface.co/uygarkurt/bert-restore-punctuation-turkish)
   declares MIT. This is a model declaration, not evidence for the CSV dataset license.
3. The CSV folder is named `multitarget-ted`, suggesting a connection to Kevin Duh's
   [Multitarget TED Talks Task](https://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/).
   This is a provenance lead, not a verified derivation chain.
4. That original corpus page identifies TED/WIT3 as its source and states BY-NC-ND
   terms. Its [README](https://www.cs.jhu.edu/~kevinduh/a/multitarget-tedtalks/README.txt)
   describes development/test sets of 1,958/1,982 examples, unlike the released
   punctuation CSV row counts. We cannot assume the original corpus splits were retained.
5. The full Kurt & Çayır paper was not obtained during this audit; its publisher page
   was not readable through the available browser fetch. Corpus derivation, split
   construction, and the intended dataset license should be clarified from the paper
   or the authors. No author has been contacted.

The current repository publishes original project code, source links, file hashes
and aggregate audit statistics. It does not redistribute the CSVs or modified texts.
No confirmed permission conclusion is recorded for the released dataset.

## Reader implementation and next step

`src/turkish_punctuation/data.py` provides the shared pandas reader, record validation,
empty-token locations and summary counts. The audit script and
`notebooks/01_data_inspection.ipynb` use the same reader. Structural errors stop the
read with their source row ID; labeled empty strings are preserved and reported.

Word-to-subword alignment has now been inspected as described below. Choose the
empty-token modeling policy before preparing model batches or adding noise.
Implementation is handled by the assistant, with explanations of code, design
decisions and failure modes.

## Tokenizer and alignment audit

The initial BERT candidate uses the tokenizer released with
[`uygarkurt/bert-restore-punctuation-turkish`](https://huggingface.co/uygarkurt/bert-restore-punctuation-turkish),
pinned at revision `dd8b9481ed3b3c05110d7e041ecbeb6a1c23b8b0`.
Its source configuration identifies `dbmdz/bert-base-turkish-cased`, preserves
casing, maps `O/PERIOD/COMMA/QUESTION_MARK` to `0/1/2/3`, and specifies a 512-position
limit. The current candidate base-model revision is recorded separately; the exact
original fine-tuning initialization revision has not been established.

Only tokenizer assets were downloaded. No neural model weights were downloaded,
and no inference or training was run. The reproducible settings are in
`configs/tokenizer.json`; the measurement script is `scripts/inspect_tokenizer.py`.
Complete aggregate measurements and an invented illustration are in
`docs/tokenization-audit.json`. Run the script with `--download-tokenizer` on first
use, or without that flag for an offline run from the existing cache.

`src/turkish_punctuation/tokenization.py` assigns each word's label to its first
subword, matching the inspected upstream code and the
[Hugging Face token-classification recipe](https://huggingface.co/docs/transformers/tasks/token_classification).
Other subwords and special/padding positions get `-100`. The other subwords still
enter the encoder; they simply do not receive an additional supervised target.
`O` remains a real supervised class. Alignment rejects missing or reordered word
IDs instead of silently dropping target labels.

All 146,606 records were tokenized with no truncation or padding:

| Split | Fully covered rows | Rows with missing words | Missing nonempty words | Max input length | 95th percentile | Rows over 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 116,969 | 462 | 0 | 227 | 43 | 0 |
| Validation | 14,453 | 61 | 0 | 196 | 42 | 0 |
| Test | 14,600 | 61 | 0 | 153 | 44 | 0 |

Lengths include `[CLS]` and `[SEP]`. Percentiles use the nearest-rank definition.
The missing word positions are exactly the 651 empty strings from the CSV audit.
Four `[UNK]` pieces occur in train; none in validation or test. Unknown pieces still
represent their source words and do not trigger missing-word exclusion.
Every fully covered, in-limit row passes an exact check that its supervised label
IDs reproduce the original word-label sequence once each.

These measurements apply to clean inputs only. Character noise can increase the
number of subwords, so noisy variants must be checked again. No sequence length
has been selected for training; 512 here is the verified candidate model limit.

### Candidate policy simulation — not applied

One defensible option is to exclude an entire record when a word has no tokenizer
representation, keeping every retained word-label association unchanged. The
diagnostic also flags over-limit rows, though none occur in this clean release.

| Split | Rows excluded | Row percentage | Eligible rows | Question labels removed | Question-label percentage removed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 462 | 0.393% | 116,969 | 183 | 1.864% |
| Validation | 61 | 0.420% | 14,453 | 26 | 2.168% |
| Test | 61 | 0.416% | 14,600 | 27 | 2.151% |

This is a simulation, not a finalized cleaning policy. Entire-row exclusion removes
additional nonempty words and labels from affected records; the JSON report counts
all of those labels. The question-class impact is larger than the row percentage,
so the exclusion cannot be described as negligible for every class.

If this option is adopted, retain source row IDs, apply one fixed eligibility rule
before generating noise, and compare every model on the same eligible test rows.
Scores on that subset must not be presented as directly reproducing the paper's
full-test scores. Do not exclude different rows separately for each noise condition;
handle any later noise-induced coverage/length failures explicitly.

## Applied modeling preparation: clean-v1

After the diagnostics above, whole-row quarantine was adopted for the initial
modeling subset. `scripts/prepare_data.py` uses `configs/preprocessing.json` and
the pinned tokenizer. The raw reader still preserves every source record.

- Eligible train/validation/test counts: **116,969 / 14,453 / 14,600**.
- Quarantined row counts: **462 / 61 / 61**, matching the earlier simulation.
- No word/label merging, lowercasing, deduplication, split mixing or truncation.
- Hugging Face datasets preserve original split-local `row_id` alongside model
  inputs and aligned labels. The DataLoader removes row IDs from model arguments.
- Local quarantine JSONL files retain the affected records, their original IDs
  and the reason. The local preparation manifest records all eligible/excluded IDs,
  source hashes, settings, package versions, class support and preview batch shapes.
- Prepared data, quarantine texts and manifests stay under Git-ignored
  `data/processed/clean-v1/`. The versioned output cannot be overwritten in place.

Tokenization and label alignment completed for every eligible row. A first batch
from each split was collated with dynamic padding using the ready-made Hugging Face
token-classification collator and PyTorch DataLoader. Padding labels are `-100`.
This establishes prepared inputs, not model correctness or predictive performance.
No model weights, training or evaluation were used in this step.

All model comparisons must share these eligible test IDs. Scores describe the
filtered subset, not a direct reproduction of the published full test benchmark.
Future noise must be generated from this fixed subset; new missing words or
overlong sequences stop encoding instead of silently removing condition-specific
rows. The question-label losses in the earlier table remain a reporting limitation.
