# Augmented BERT: matched medium-noise validation pilot

## Scope fixed before inference — 2026-10-22

Evaluate the completed augmented run's selected **epoch-03** model on a fresh
clean control and the same four medium/seed-586 validation conditions already
used by the [clean fine-tuned model](clean-noise-pilot.md) and the
[published reference](noise-pilot.md). Reuse the existing noise words, masks and
encodings; do not regenerate noise or select a checkpoint using noisy scores.

- Run identity:
  `3cdf62cc3be4bbd21e48dc17aee7b450f90e6b1e2b6041dea185e3d8f0ceb01e`.
- Selected tensor digest:
  `f7e03bcf899af866cd7711a2747974c84fe6448b2deab002c3834811fe9c648b`.
- Same validation population: 14,453 rows / 179,164 word targets.
- Case loss, diacritic loss, internal adjacent swap and combined noise; medium
  dose, corruption seed 586. This pilot does not cover the severity/seed grid.
- Same CPU FP32 inference, batch size 16, four PyTorch threads, seed and
  clean-length ordering/batch membership as both completed reference pilots.
- Same PERIOD/COMMA/QUESTION_MARK macro-F1 primary score, micro and per-class
  P/R/F1, confusion matrices, paired outcomes and source-order error examples.
- Before inference, verify shared initialization tensors, training budget,
  optimization, selection policy, training runtime and the paired clean identity.

The principal comparison is augmented versus clean fine-tuning: both trained
from identical initialization with the same record/word/update budget. Report
absolute noisy F1 **and** each model's drop from its own clean score. The
published checkpoint is a separate reference with uncontrolled training history.
Also retain clean-score trade-offs and class-specific changes, including losses.

This is a descriptive validation experiment with one training seed and one
corruption seed. There is no significance or held-out generalization claim.
These validation data also served clean checkpoint selection. No training
updates, Drive writes/deletion or held-out test access are part of this step.

## Reproduction

The existing evaluator verifies the completed export, hashes, raw validation
targets, prepared inputs, tokenization, package versions and batch membership.
Its inference/metrics implementation is reused unchanged.

```bash
python scripts/evaluate_noise.py --config configs/augmented-noise-evaluation.json
python scripts/analyze_noise.py \
  --run-directory artifacts/evaluations/finetuned-augmented-seed586-noise-pilot-v1 \
  --comparison-run-directory artifacts/evaluations/finetuned-clean-seed586-noise-pilot-v1
python scripts/analyze_noise.py \
  --run-directory artifacts/evaluations/finetuned-augmented-seed586-noise-pilot-v1 \
  --output-directory artifacts/evaluations/finetuned-augmented-seed586-noise-pilot-v1/analysis-published \
  --comparison-run-directory artifacts/evaluations/published-bert-noise-pilot-v2
```

For an interrupted evaluation, repeat its identical command with `--resume`.
Completed condition outputs are retained and verified; the interrupted condition
is recomputed. Existing completed output directories are never overwritten.

## Execution status

Configuration and comparison scope are recorded; inference has not started yet.
Results, logs, source snapshots, progress and comparison reports stay under the
ignored `artifacts/evaluations/` directory. Aggregate results will be recorded
here after all conditions and comparison checks complete.
