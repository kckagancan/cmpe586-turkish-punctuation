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
- Dataset origin, license, exact file paths, split provenance, casing and label
  format still need source-level verification. No dataset has been downloaded.

## Collaboration preferences

- Explain architecture and first steps before large implementations.
- Let the student implement some modules and review them together.
- Assist directly with implementation/debugging when the student is stuck.
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
  starter notebook execution verified. No model/data downloads or training yet.
- Next milestone: verify dataset source, license, schema and split provenance.
