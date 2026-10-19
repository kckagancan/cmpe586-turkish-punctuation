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
- The first published-reference clean-validation measurement is complete.
  A published-model medium-noise validation pilot is complete; fine-tuning has not started.
  Do not invent results.

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

- Protocol specified on 2026-10-18 in [noise-protocol.md](noise-protocol.md) and
  [configs/noise.json](../configs/noise.json). Operators and a validation-only
  preparation/audit command are implemented; the first model pilot is complete.
- Case loss with Turkish I rules, explicit Turkish character folding, and one
  adjacent interior character swap preserving word ends. This initial typo
  condition does not represent all spelling/keyboard errors.
- Single-family mild/medium/heavy budgets change exactly floor(2%/5%/10% of all
  original clean-v1 word positions). Select from that family's eligible pool;
  fail if the pool is too small. These are project-defined synthetic doses.
- Select by stable SHA-256 ranks with seeds 586/587/588. Exclude labels and model
  predictions from selection; use nested severity masks and fixed swap locations.
- Combined conditions reuse the component plans, applying swap, case loss, then
  Turkish folding. Report union/overlap/final coverage; this is a co-occurrence
  stress condition with a larger budget, not an equal-budget interaction estimate.
- Add deterministic full-case-loss and full-Turkish-fold endpoints outside that
  matched grid. Plan a medium, seed-586 validation pilot after the data-only audit.
- Preserve word-to-label alignment. Noise may change subword tokenization, so labels
  must be aligned again after tokenization; identical subword counts are not assumed.
- Handle Turkish `I/ı/İ/i` explicitly when designing transformations.
- Clean tests, each noise type separately, severity sweeps and combined noise.
- Overall and period/comma/question-mark precision, recall and F1.
- Primary score: fixed-class macro-F1 over PERIOD/COMMA/QUESTION_MARK, with equal
  class weights. Also report macro P/R, punctuation micro-P/R/F1, per-class
  P/R/F1/support and a four-class confusion matrix. Zero denominators yield 0.
  Retain O reference positions to count false-positive marks; omit O only from the
  punctuation averages. All-class accuracy is diagnostic, never the primary score.
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
- Keep established ML/software terms in English (for example precision, recall,
  accuracy, augmentation, batch size and checkpoint); explain them in natural
  Turkish instead of forcing unfamiliar literal translations.
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
  starter notebook execution verified. Candidate data downloaded and audited.
  Published BERT weights are now cached locally; training has not been performed.
- The one-off environment-check notebook was retired after setup; use
  `scripts/check_setup.py` for repeatable environment checks.
- Shared pandas reader and structural validation now live in
  `src/turkish_punctuation/data.py`; the inspection notebook imports those functions.
  Empty tokens are preserved and reported by the reader; modeling preparation
  applies an explicit separate policy without changing raw records.
- BERT tokenizer candidate inspected at pinned Hugging Face revision
  `dd8b9481ed3b3c05110d7e041ecbeb6a1c23b8b0`. The initial audit downloaded only
  tokenizer assets; the subsequent inference preview also downloaded weights at
  that same revision. Config is in `configs/tokenizer.json`.
- First-subword label alignment is implemented with complete word-coverage checks.
  Special/additional subword positions get `-100`, distinct from the real `O` class.
- Actual tokenizer audit: only the 651 empty slots disappear; all nonempty words
  have coverage. Clean input max length is 227 including special tokens, below 512.
- Initial modeling policy `clean-v1`: quarantine entire rows containing empty
  words, preserving their original tokens, labels and source row IDs locally.
  Do not merge punctuation targets or silently delete individual empty slots.
- Eligible train/validation/test rows: 116,969/14,453/14,600. Quarantine:
  462/61/61. This removes about 0.4% of rows and 1.86–2.17% of question labels.
  Report the subset and support explicitly; published full-test scores are not
  directly comparable. Evaluate all candidate models on the same eligible test IDs.
- `configs/preprocessing.json` and `scripts/prepare_data.py` create local versioned
  Hugging Face datasets, quarantine JSONL files and a provenance/count manifest.
  Existing output is never overwritten. No processed text is published to Git.
- `src/turkish_punctuation/preprocessing.py` separates records, encodes with strict
  first-subword alignment, and builds a PyTorch DataLoader using the ready-made
  token-classification collator. Padding is dynamic; source row IDs stay in the
  saved dataset but are removed before collation into model inputs.
- Future noise derives from the fixed eligible records. New missing words or
  length-limit violations stop encoding rather than causing condition-specific
  exclusions or silent truncation. A handling policy must be defined if encountered.
- Published BERT loaded without missing/unexpected/mismatched weight keys. Its
  four-label mapping and 512-position limit match our pipeline. Architecture:
  BertForTokenClassification, 110,029,828 parameters, 12 layers, hidden size 768,
  12 attention heads per layer. Only the classifier's first-subword outputs are
  decoded; other subwords still participate in the encoder's contextualization.
- `src/turkish_punctuation/inference.py` loads the checkpoint and predicts in
  eval/inference mode. Reference class values are not passed to model.forward().
  `scripts/inspect_model.py` runs three authored examples on CPU, preserving
  per-word logits and predictions in Git-ignored `artifacts/model-preview.json`.
  This illustration is not a validation/test performance estimate.
- Code/pipeline review completed: fixed strict CSV-width handling and a malformed
  token/label-count case hidden by an empty word. Added encoding/inference guards
  and 30 offline regression tests; all pass. Full raw/processed datasets and the
  cached model examples were rechecked without rewriting them. See `code-review.md`.
- First clean validation reference completed: 14,453 eligible records, 179,164
  word targets, macro-F1 0.8363658278 and micro-F1 0.8266510513. Period/comma/question
  F1: 0.9582026830 / 0.6255729118 / 0.9253218884. See `clean-validation.md`.
- Streaming inference now serves prepared datasets as well as the earlier small
  text preview. Metrics use word positions only. Settings are in
  `configs/evaluation.json`; the command is `scripts/evaluate.py`.
  Exact predictions/metrics and the executed sources are local Git-ignored artifacts.
- This filtered validation score is a reference, not the published full-test score.
  The upstream checkpoint may have used the same validation split for selection.
- Noise protocol and operators are implemented, with literature adaptations and limits recorded.
  Train-only eligibility guided the 2/5/10% budgets; validation confirmed feasibility.
  Case-loss eligibility is the limiting pool at about 13.35% of all words.
  No noisy model scores or test text were used to select these evaluation rates.
  The subsequent training protocol specifies augmentation mixtures/ablations and
  matched optimizer budgets after inspection of the published-model validation pilot.
- `src/turkish_punctuation/noise.py` implements word transformations, stable
  eligible-position rankings and composition. Plans are tied to clean word/ID
  hashes; labels never enter selection. `noise_audit.py` verifies invariants and
  measures realized exposure, overlap, class exposure and tokenizer fertility.
- `scripts/prepare_noise.py` prepares validation conditions and encoded inputs
  locally, checks exact clean-v1 reproduction, nesting and combined mask identity,
  and saves config/source snapshots and aggregate audits. Existing runs are not
  overwritten; output is published only after all conditions pass.
- Validation data audit completed on 2026-10-18: 38 noisy conditions retained all
  14,453 rows and 179,164 targets. Exact single-family budgets and nested/shared
  masks passed; no missing words, UNKs or >512 inputs. Maximum length was 218.
  Clean re-encoding matched clean-v1 exactly. See [noise-audit.md](noise-audit.md).
  That preparation stage performed no model evaluation, training or test-text processing.
- Medium/seed-586 published-model pilot completed on 2026-10-18. Macro-F1:
  case 0.8122560260, fold 0.8343473126, swap 0.8290443302, combined 0.8034385481.
  Drops: 2.41098 / 0.20185 / 0.73215 / 3.29273 pp. Combined edits 14.3293% of
  words versus 4.9999% in each single-family condition. See [noise-pilot.md](noise-pilot.md).
- To reduce padding, a source-order partial attempt was stopped before a noisy
  aggregate score existed. The completed v2 run shares clean-length sorting across
  conditions. Its recomputed clean control reproduced all 179,164 original word
  predictions. Successful inference loops totaled about 37.4 minutes on CPU.
- `robustness.py` uses existing inference/metrics for paired comparisons;
  `scripts/evaluate_noise.py` verifies inputs, snapshots code, saves per-condition
  results and supports identical-run resumption. `scripts/analyze_noise.py`
  derives context/direct-edit counts without model inference. Settings are in
  `configs/noise-evaluation.json`; local results are Git-ignored.
- Case loss produced 1,153 new errors, of which 1,005 were at unchanged words in
  edited records. Its period-to-comma confusions increased from 220 to 765.
  This is consistent with contextual casing cues, not proof of an internal causal
  mechanism. All completely unedited records retained their predictions.
- `training-v1` specified on 2026-10-18 in [training-protocol.md](training-protocol.md)
  and `configs/training.json`. The clean pipeline completed a short GPU calibration;
  full training is pending. Shared pinned
  BERTurk encoder plus one saved fresh four-class head per training seed;
  full fine-tuning, AdamW at 2e-5, weight decay 0.01 excluding bias/LayerNorm,
  dropout 0.1, 3 epochs, batch 8 × accumulation 4 = effective 32, word-normalized
  unweighted cross-entropy, 10% linear warmup/decay and clipping 1.0. Expected
  10,968 updates per arm; no early stopping. Select the best clean validation
  macro-F1 epoch, keep last checkpoint for equal-update sensitivity analysis.
- Augmented training presents half of row IDs through a combined-medium view,
  refreshed per epoch with train-specific seeds. Every original training row
  appears once per epoch; realized word corruption is audited after row mixing.
  Start with one clean/augmented pair at training seed 586, then at least a
  no-case-loss training ablation after GPU budgeting. Extra training seeds and
  the other leave-one-family-out arms are optional. Training and evaluation
  corruption seeds quantify separate variation.
- Upstream public training code has a declared LR/custom optimizer LR discrepancy;
  our recipe is literature-informed and independently specified, not an exact
  reproduction or claimed optimum. Test data has not informed its choices.
- Clean training implementation: `training_model.py`, `training_data.py`,
  `training.py`, `training_io.py` and `scripts/train.py`. Implements pinned base
  loading/fresh shared head, exact train/validation re-encoding, word-normalized
  accumulation, clean epoch selection, deterministic row order, resumable full
  state and source/data/environment snapshots. CLI requires CUDA/FP32; default
  mode is a separate 30-update calibration, not full training. Eager attention
  and strict deterministic operations are explicit runtime choices.
- `notebooks/02_clean_training_colab.ipynb` prepares train/validation only through
  `scripts/prepare_training.py`, with one Drive mount-if-needed cell and a
  separate inactive-by-default training cell. Old permission retries, calibration
  rerun and export cells were removed. Preparation preserves exact input files,
  shared initialization and calibration reports in the designated Drive project
  folder, recording hashes and actual dependencies with zero optimizer updates.
  `requirements/colab.constraints.txt` pins the main observed dependencies while
  retaining the Colab GPU PyTorch installation. Download/preparation
  scripts accept optional `--splits`; existing defaults and outputs are preserved.
  The notebook requires the reviewed training code to be synced to GitHub first.
- Candidate epoch exports are retained along with two rolling resumable states
  (roughly 4 GB per full run plus shared initialization). Resume requires matching
  source/config/input/package/GPU identity; the lock is scoped to one host.
- Clean GPU calibration completed on 2026-10-18, code commit `fc17d41`, seed 586,
  T4, runtime 2026.07, Python 3.12.13, PyTorch 2.11.0+cu128, FP32/eager. All
  train/validation re-encoding, tokenizer compatibility and base loading checks
  passed; 30 updates/960 records completed. Mean 0.475685 s/update after five
  startup updates, peak allocated 2.430 GiB, peak reserved 2.838 GiB. Projected
  10,968-update work is 86.96 min excluding validation/checkpoint I/O. Retain
  batch 8/accumulation 4 for clean training. Updated calibration weights discarded.
  See [training-calibration.md](training-calibration.md); raw reports/source snapshots
  are saved under ignored `artifacts/training/training-v1/calibration-clean-seed-586-v1/`.
- Drive mount returned `credential propagation was unsuccessful`, including after
  the user reported granting file/document access; the missing permission category
  is not known. Calibration used temporary Colab storage and reports were retrieved
  locally. A successful Drive mount was later observed after the user completed
  authorization. This does not identify a minimum sufficient permission subset.
  Persistent-asset preparation is a separate step from training. Full-epoch validation,
  checkpoint/resume, augmented training and noisy-input memory remain unobserved.
  No formal training or new model-quality/test score was produced. Continue
  clarifying source/license and prepare augmented views after the clean path.
- Persistent preparation completed on 2026-10-18 using code `a121194`, with a clean
  checkout, exit code 0 and 102.444 seconds elapsed. Train/validation inputs,
  the original shared initialization and the previous calibration reports were
  copied to the designated Drive project folder and verified by hash. Preparation
  performed zero optimizer updates and did not access test data. Drive writes
  were subsequently flushed successfully and the mount was closed.
  See [preparation-status.md](preparation-status.md) for the observed record.
- During the 2026-10-18 session, the user requested stopping at preparation and
  leaving full training for a later session.
- On 2026-10-19, the user authorized the next stage. Formal clean training was
  launched around 03:27 Europe/Istanbul on a new T4 runtime using code `7351f9f`.
  Input and shared-initialization checks passed; the first 525 updates ran without
  an error and execution continued past the synchronous step-500 checkpoint.
  Full-epoch validation/selection and checkpoint recovery remain unobserved.
  The user will watch the visible Colab notebook and return after it
  finishes. Training is in progress, not a completed result; see
  [clean-training-status.md](clean-training-status.md). The checked-in notebook
  keeps its training switch false; the active Colab copy enables this run.
- Clean training completion was verified on 2026-10-19: 3 epochs / 10,968 updates,
  best epoch 3, clean validation punctuation macro-F1 0.8403653366075169 and
  micro-F1 0.8298740077197582. The loop including validation/checkpoints took
  72.1855 minutes, excluding initial loading. Metadata identities, source/config
  hashes, epoch selection and supports agree. Model and validation exports were
  checked by the runner before its summary was published; checkpoint recovery
  remains unexercised. Metadata copies are saved locally under ignored artifacts;
  weights remain in the Drive project folder. Next: matched clean + medium-noise
  validation of the selected fine-tuned clean model, followed by augmentation
  implementation/audit/training. No new noisy-model score or test result yet.
- A post-completion Colab toast reported Drive storage quota exceeded, with
  approximately 702 MB free in a project-scoped mount check. No personal files
  or Trash were inspected/deleted. Preserve the selected model before resetting
  the runtime or starting another run. A 448,623,671-byte recovery archive was
  created and its six model files verified. In-app download did not produce a
  confirmed local file. The user subsequently downloaded it through a normal
  browser; the original ZIP hash, selected weights/tensors, all epoch metrics
  and training event counters were verified. A project-local archive/extracted
  copy is now confirmed; see [backup-verification.md](backup-verification.md).
  See the storage section of `clean-training-status.md`. Resolve durable storage
  before continuing the planned noisy validation or augmented training.
- The requested comprehensive post-download audit also reproduced 806/806 saved
  word predictions on CPU across 31 records and re-encoded all validation rows
  using the archived tokenizer. Local preparation has different package/split
  metadata, while validation content/alignment is identical. Thirty offline
  regression tests pass after isolating `sys.argv` in a preparation test helper;
  production training/model/metric implementations were not changed.
- The original T4 environment then replayed the entire clean validation set:
  all 179,164 word predictions and metrics matched, including prediction-file
  SHA-256. All epoch exports and both retained optimizer states were checked.
  No training update or held-out test access occurred. Plan roughly 4 GB per arm
  and archive completed runs locally before selective
  cleanup. The 449 MB selected-model ZIP excludes other epoch weights and
  optimizer states. A flush/unmount attempt failed. Reconnection succeeded, with
  all epoch/checkpoint hashes matching again, but fresh quota readings show only
  about 0.16 GB free (Google One: 98% used). The user's reported 12–13 GB free and
  an earlier mount estimate disagree with this. Do not start another large run
  until capacity is confirmed; the local selected-model archive is verified.

### Latest storage decision — 2026-10-19

- The user subsequently reported about 9.6 GB free after manually clearing Trash.
  This latest capacity report has not been independently rechecked. Existing
  clean-run models/checkpoints should stay in place for now; no cleanup is
  requested in this step.
- After every completed training run, transfer required models/results/provenance
  to ignored local storage, verify hashes and model loading, then clean up the
  corresponding unnecessary Drive copies. Preserve shared initialization/input
  assets needed by subsequent arms. Retain resumable state locally when needed;
  do not keep obsolete optimizer states just because they exist. Full checklist:
  [training-protocol.md](training-protocol.md#completed-run-storage-policy-agreed-2026-10-19).
- The existing 449 MB clean-model ZIP is not yet a complete run archive. Future
  cleanup must use project/run identity, never blanket Trash deletion. Automated
  permanent cleanup has been discussed but **has not been implemented or run**.
  Address active-run checkpoint Trash accumulation before the next training.
- Do not promise 13.5 GB free after every cleanup: measure actual usage, including
  retained shared assets and other account storage. A hypothetical 25-epoch run
  also needs bounded model retention during training; archiving only at the end
  would not prevent filling Drive. Current training remains three epochs.
