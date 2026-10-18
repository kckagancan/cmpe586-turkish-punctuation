# Code and pipeline review

Scope: current working tree, including the uncommitted preprocessing and model
preview code. Reviewed all project Python files, the inspection notebook, settings,
dependency declarations, configuration files and research notes. This was a local
CPU correctness review, not a benchmark or training experiment.

## Reproduced and fixed

1. **Extra CSV fields could disappear silently.** With two named columns but three
   fields in a row, pandas could infer an index from the extra leading field. The
   remaining lists then passed our structural checks. The reader now disables
   inferred indexes and raises on parser warnings/errors instead of accepting
   discarded fields. Regression cases cover extra leading/trailing fields and an
   extra field in a later row.
2. **An empty word could hide mismatched token/label lengths.** A direct call to
   encoding with `tokens=["gittim", ""]`, `tags=["PERIOD"]` previously passed:
   the tokenizer dropped the empty word and the remaining word IDs matched the
   shorter label list. Encoding now checks original token/label counts before
   tokenization. Such malformed records stop with the source row ID.

Neither issue was present in the pinned raw release or saved clean-v1 output.

## Additional guards and coverage

- Reject duplicate source IDs within an encoded split and requested lengths beyond
  the tokenizer/model position limits.
- Check the supplied prediction model's label mapping, including models passed
  directly without using our checkpoint loader.
- Reject NaN/infinite logits rather than silently converting them to a class.
- Exercise dynamic padding, ignored target positions, seeded shuffle, sequential
  evaluation order, unknown pieces and noise-induced subword splitting.
- Verify that reference label values do not affect predictions, model inputs omit
  labels, evaluation disables dropout and inference disables gradient tracking.
- Check that checkpoint loading rejects incomplete weights and that failed data
  preparation leaves no published partial output. Existing versioned data is kept.
- Refresh notebook/config documentation that still described preprocessing as a
  future step. Notebook code and saved raw/model outputs were not changed.

## Verification results

- **30 offline unit/regression tests passed.** Tiny tokenizer/random BERT fixtures
  require no downloads, pretrained checkpoint, dataset release or GPU.
- **14 Python files:** Pyright basic with the installed Pylance stubs reported
  **0 errors and 0 warnings**. Ruff checks passed; dependency consistency passed.
- **146,606 raw records:** file checksums, complete reader statistics and tokenizer
  statistics matched their existing manifests/audit reports.
- **146,022 eligible records:** re-encoding matched every saved token ID, attention
  mask, segment ID, aligned label and row ID across train/validation/test.
- Quarantine contents, source IDs, split counts and class supports matched the
  saved preparation manifest. Raw and prepared files were not rewritten.
- All **7 notebook code cells** executed sequentially in a fresh Python namespace;
  notebook format validated and saved outputs remain empty. This check did not
  launch a new Jupyter kernel or verify the VS Code UI.
- The cached published BERT loaded offline without missing/mismatched/unexpected
  weights. All three original example predictions and word logits were reproduced
  within floating-point tolerance; the preview file was not rewritten.

## Scope of the conclusion

No remaining blocker was found for the next local CPU validation-evaluation step.
This does not establish GPU/Colab behavior, training convergence, noise severity
choices or model quality. Dataset provenance/license questions and exact published
training provenance remain open as recorded in `project-brief.md`.

## Training type-check follow-up — 2026-10-18

The three reported training files reproduced **7 errors** in Pyright 1.1.414,
basic mode, using the local `cmpe586` environment:

- `training_data.py`: the initial metadata dictionary inferred string-only values.
  Explicit `dict[str, object]` metadata and `dict[str, Dataset]` split types now
  describe the heterogeneous contents correctly.
- `training_io.py`: NumPy's `get_state` annotation permits a dictionary or a legacy
  tuple. Capture explicitly requests legacy state and checks for a tuple before
  indexing its five fields.
- `training_model.py`: Transformers annotates `from_pretrained` as a model even
  though `output_loading_info=True` returns `(model, loading_info)`. The installed
  implementation confirms that branch; a cast describes the pair without changing
  the call or disabling diagnostics.

After the fixes, the same **3 files report 0 errors / 0 warnings** and Ruff passes.
A focused CPU check saved Python/NumPy/PyTorch RNG state, loaded it with
`weights_only=True`, restored it and reproduced the next draws exactly. No full
test suite, GPU calibration or training was rerun for this follow-up. The existing
calibration remains tied to its archived source snapshot; these edits do not
alter its recorded measurements or the training hyperparameters.
