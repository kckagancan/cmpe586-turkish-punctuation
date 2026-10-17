# Project brief and decisions

## Confirmed scope

- Course: CMPE 586, Transformer/NLP.
- Title: **Robustness of Turkish Punctuation Restoration under Informal Writing Noise**.
- A one-page NeurIPS-format Project Idea has been prepared. It is an idea-stage
  document, not a comprehensive proposal. The document itself is not yet in this repo.
- Prioritize defensible research questions, controlled experiments, baseline
  comparisons, ablations, error analysis and explanations of observed effects.
- Keep compute low and support Google Colab. Use existing Turkish data and models
  where possible. Heavy generation projects such as sentence compression are out of scope.
- No experiments have been conducted. Do not invent results.

## Research questions

1. Which informal writing noise types and severity levels hurt restoration most?
2. When does noise augmentation help, and what happens to clean-text performance?
3. How do period, comma and question-mark classes differ in their response, and why?

## Planned comparisons

1. A published punctuation-restoration checkpoint, evaluated as released.
2. A clean-data fine-tuned baseline.
3. The same architecture fine-tuned with synthetic noise augmentation.

Choose a shared base initialization and matched training budgets for comparisons
2 and 3. Record the published model's training provenance separately. Its score is
a reference; augmentation effects must come from the controlled 2-versus-3 comparison.
Start with one architecture; additional architectures depend on compute and findings.

## Noise and evaluation

- Case loss, removal of Turkish diacritics, character-level typos.
- Generate mild/medium/heavy conditions automatically, with reproducible seeds.
- Define rates and denominators explicitly before experiments. Rates are not yet set.
- Preserve word-to-label alignment. Noise may change subword tokenization, so labels
  must be aligned again after tokenization; identical subword counts are not assumed.
- Handle Turkish `I/ı/İ/i` explicitly when designing transformations.
- Clean tests, each noise type separately, severity sweeps and combined noise.
- Overall and period/comma/question-mark precision, recall and F1.
- Fix the overall averaging convention and treatment of the no-punctuation class
  before comparisons. Avoid treating dominant no-punctuation accuracy as the main score.
- Keep original splits; derive test noise only from the held-out test partition.
  Use validation for choices and keep test data out of training/augmentation.
- Include ablations and qualitative error analysis.

## Starting sources and open questions

Checked on 2026-10-17:

- [Official Kurt & Çayır (2023) implementation](https://github.com/uygarkurt/Turkish-Punctuation-Restoration)
  lists BERT/ELECTRA/ConvBERT checkpoints and train/validation/test data.
- [BERT model card](https://huggingface.co/uygarkurt/bert-restore-punctuation-turkish)
  declares MIT for that model. This does not establish the dataset license.
- Three original CSV files downloaded locally from pinned revision
  `7ede7a4ba9e51dad085a7f0d9b6c0a9c32622397`; hashes and URLs recorded in
  `docs/data-manifest.json`. Format, counts and casing inspected across all rows.
- Dataset license and exact source/split provenance remain unresolved. The folder
  name suggests Multitarget TED, whose original source states BY-NC-ND terms;
  derivation of these specific CSVs is not yet verified.
- Audit found 651 empty tokens carrying punctuation labels and short repeated
  inputs across splits. Raw data is unchanged. See `docs/data-audit.md` for counts
  and decisions needed before modeling.

## Collaboration preferences

- Explain architecture and first steps before large implementations.
- Explain every stage, its purpose and decisions as we go; the student wants to
  build practical project and AI/LLM expertise through the work.
- Default to function-level overviews, important lines and research implications.
  Explain individual syntax or line-by-line mechanics only when the student asks.
- Updated preference: the assistant implements and debugs the project. Do not
  assign coding exercises or quizzes. Explain the code, unfamiliar APIs, design
  choices and failure modes so the student can read, critique and defend the work.
- Keep notebooks concise and oriented toward real project inspection and results.
  Put reusable reading, validation, noise, training and evaluation logic in Python
  modules; notebooks import that logic for inspection, plots and Colab execution.
- After every request that changes files, list every created, modified, renamed
  or deleted file, including local-only settings. Explain each file's purpose and
  change. Walk through the code with the student before advancing research stages;
  use code reading and review, not coding exercises, as the learning workflow.
- Lead report drafting while ensuring the student understands and can defend the work.
- AI use is allowed according to the student's account of course rules.
- Proceed incrementally and agree on research-stage transitions together.

## Development setup

- Repository: `kckagancan/cmpe586-turkish-punctuation` (public).
- Python 3.11, Conda environment `cmpe586`.
- VS Code + Python modules as the primary development workflow.
- Thin notebooks for inspection, plots and Colab execution.
- Local CPU development; plan Colab GPU training.
- Environment and public repository setup completed; local package imports and
  starter notebook execution verified. Candidate data downloaded and audited;
  no model weights downloaded or training performed.
- The one-off environment-check notebook was retired after setup; use
  `scripts/check_setup.py` for repeatable environment checks.
- Shared pandas reader and structural validation now live in
  `src/turkish_punctuation/data.py`; the inspection notebook imports those functions.
  Empty tokens are preserved and reported, with no modeling cleanup applied yet.
- BERT tokenizer candidate inspected at pinned Hugging Face revision
  `dd8b9481ed3b3c05110d7e041ecbeb6a1c23b8b0`. Only tokenizer assets were
  downloaded; no model weights or training. Config is in `configs/tokenizer.json`.
- First-subword label alignment is implemented with complete word-coverage checks.
  Special/additional subword positions get `-100`, distinct from the real `O` class.
- Actual tokenizer audit: only the 651 empty slots disappear; all nonempty words
  have coverage. Clean input max length is 227 including special tokens, below 512.
- Excluding whole affected rows was simulated only: 462/61/61 rows across splits.
  This is about 0.4% of rows but removes around 1.86–2.17% of question labels.
  No processed split was saved and no exclusion policy is finalized.
- Next milestone: agree on an explicit empty-token modeling policy and create
  reproducible eligible splits/model batches. Continue clarifying source and license.
