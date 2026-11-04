# Selecting an informal Turkish corpus for the feedback extension

Reviewed **2026-11-04**, before any extension model scoring. The original
experiments and report remain intact. Selection uses source/annotation fitness,
not model gains. This is a targeted comparison of relevant accessible resources,
not a claim that every Turkish corpus has been exhaustively evaluated.

## Decision and selection criteria

Use **MultiLexNorm, monolingual Turkish, official training split** as the primary
data-only audit source. It combines real user text, aligned normalization targets,
capitalization annotation, small reproducible files and a documented shared-task
format. Keep its test file unread. Its normalization labels are not punctuation
restoration gold, and its sample is not an unbiased population survey.

Criteria, in order: (1) authentic informal input; (2) human/reference correction
pairs rather than synthetic degradation; (3) original casing and capitalization
corrections; (4) auditable word alignment; (5) accessible, pinned small inputs;
(6) clear task/provenance/terms. Fine-grained error labels are helpful but do not
outweigh losing original casing information. Bigger size alone is not a benefit
for estimating real error mechanisms.

## Candidate comparison

| Candidate | Verified useful evidence | Limit for this question | Role |
| --- | --- | --- | --- |
| [MultiLexNorm Turkish](https://noisy-text.github.io/2021/multi-lexnorm.html) | Natural tweets, word-level normalization and capitalization annotation; source paper [Table3](https://aclanthology.org/2021.wnut-1.55.pdf). Official train is 570 sentences / 6,443 source slots; observed file is 99,174 bytes. | Splits/merges need exclusion; punctuation is kept rather than repaired; source selection limits prevalence claims; per-language reuse terms not established here. | **Primary**, train only. |
| [#Turki$hTweets](https://github.com/atubakoksal/annotated_tweets) | Natural tweets, correction pairs, detailed error tags. Actual release also contains a raw column, beyond its processed/normalized fields. | Processing can confound casing; raw/processed alignment differs. Error-selected sample; explicit license not established. | Useful complementary source; initial format inspection retained, not chosen merely because it was suggested. |
| [ITU normalization resources / Web Treebank](https://nlp.itu.edu.tr/en/toolsandresources) | Official normalization annotations and web-text resources. | Dataset endpoint timed out during this review; immediate access/terms not verified. Not evidence that the resource is unavailable in general. | Potential future source, no agreement or account action performed. |
| [GECTurk real movie-review subset](https://aclanthology.org/2023.findings-ijcnlp.26.pdf) | Human-annotated natural text, separate from its synthetic training corpus. | 300 selected sentences, half correct and half erroneous; grammar-focused scope and selection make it unsuitable as a natural prevalence estimate. | Secondary option for broader transfer; do not dismiss it as all synthetic. |
| [Turkish-GEC evaluation](https://github.com/asimokby/Turkish-GEC) | Author repository distinguishes human-annotated evaluation from synthetic training. | General web GEC and sentence-level edits are less direct for a compact, word-preserving social-writing profile; tiktoken counts are not word counts. | Secondary option; not downloaded. |
| [TweetS / TS Corpus](https://tscorpus.com/corpora) | Large authentic social-media collection with spelling/slang tags. | Inspected catalogue does not establish paired human normalization gold or a suitable bulk correction file. | Could describe usage; insufficient verified evidence for this correction audit. |
| [Turkish chat normalization mini](https://huggingface.co/datasets/yagmurtuncer/turkish-chat-normalization-mini) and [Turkish normalization1M](https://huggingface.co/datasets/GoktugD/turkish-text-normalization-1m) | Cards disclose controlled/rule-based degradation. | Synthetic mistakes cannot independently calibrate natural mistake frequencies; introduces the template issue raised in feedback. | Exclude for natural calibration. |
| [Turkish text normalization](https://huggingface.co/datasets/yagmurtuncer/turkish-text-normalization) | Written/spoken representations of numbers, dates and currencies. | Different normalization task. | Exclude. |

The [MultiLexNorm paper](https://aclanthology.org/2021.wnut-1.55.pdf) and current
task webpage describe different versions/counts; the current repository has
also added languages since2021. Use the actual pinned Turkish file, not a mix
of paper/webpage totals. File-level metadata is preserved locally.

## Pinned input and refined earlier assessment

Author repository commit: `e58e946375299a04355fa2d7fa9e1c769234e68a`.
Only `data/tur/train.norm` was read; `test.norm`, `test-eval/`, and external
parsing data were not read. Source bytes are verified by SHA-256
`17793b8db8f41ae387cfcdf4b7a1c061f2ccdc677418b3c2cec912c4e570695d`.
Raw input stays in ignored storage. No explicit per-language license conclusion
is inferred from public downloadability; raw text is not redistributed.

The paper-only first review overstated the absence of observable original
casing in #Turki$hTweets: its actual file includes `RAW DATA`. That observation
was corrected before corpus selection. Having raw text still does not establish
gold punctuation or automatic raw-to-normalized alignment. The initially drafted
candidate-specific audit recipe is retained in ignored selection receipts; no
formal audit or model evaluation had been completed under it.

## Next step

Run the separately versioned [data-only audit](informal-corpus-audit-protocol.md)
on the selected train file. Determine coverage and remaining error shapes before
choosing a transfer-evaluation scope. Preserve the original synthetic-dose results;
this extension cannot retroactively make them corpus-calibrated.
