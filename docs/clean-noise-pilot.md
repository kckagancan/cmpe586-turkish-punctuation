# Clean fine-tuned BERT: medium-noise validation pilot

## Prespecified scope

Evaluate the completed clean training run's selected epoch-03 model on a fresh
clean control and the same four medium/seed-586 validation conditions as the
[published-reference pilot](noise-pilot.md). This is validation analysis, with
one training seed and one corruption seed. It is not held-out test evidence or
an augmentation experiment. No optimizer or training update is involved.

- Training run identity:
  `5ddecc9b8bfaf0b9442a7942caa45a64173e037b0ea7309692c12e5767603f37`.
- Pinned selected model tensor digest:
  `487a8c7c38d05787f8527eca642b4d2a80bb94a814b960479b9668a87492cd96`.
- Validation: 14,453 records / 179,164 word targets; same quarantine policy.
- Noise: case loss, diacritic loss, internal adjacent swap and combined, medium,
  corruption seed 586. Existing records/masks are reused without regeneration.
- CPU, batch 16, four PyTorch threads, FP32; fresh clean control and all variants
  share the published pilot's clean-length row ordering and batch membership.
- Main score: PERIOD/COMMA/QUESTION_MARK macro-F1. Also retain micro and per-class
  P/R/F1, confusion matrices, clean-to-noisy paired outcomes and error examples.

## Implementation and reproduction

`configs/clean-noise-evaluation.json` pins the archived training run and selected
tensor digest. `evaluation_model.py` verifies the completed run/epoch selection,
export files, original validation outputs and model loading. The existing
`scripts/evaluate_noise.py` also handles this verified local model, preserving
the same tokenization, alignment, prediction and metric functions used by the
published pilot. It re-tokenizes all five inputs before scoring and rejects
condition-specific row drops, changed targets or mismatched batching/packages.

The original training-time GPU predictions/metrics are retained as historical
references. Fresh CPU clean predictions are used for this pilot's paired noisy
comparisons; any differences from the training-time predictions are recorded in
the clean-anchor paired analysis. Training-time prepared-file hashes differ from
the local Arrow cache because the preparation environments differ; raw validation
hashes, every source target and every local encoding are checked explicitly.

```bash
python scripts/evaluate_noise.py --config configs/clean-noise-evaluation.json
python scripts/analyze_noise.py \
  --run-directory artifacts/evaluations/finetuned-clean-seed586-noise-pilot-v1
```

After interruption, use the identical evaluation command with `--resume`.
Completed condition files are checked and retained. Existing completed runs are
never overwritten; code/input/config changes require a new output directory.

The model ZIP/extracted archive and prepared noise data must already exist locally.
Results, progress and source snapshots are saved under the ignored evaluation
directory named above. This step writes no Drive files and deletes no models.

## Status

Runner/configuration prepared; full clean/noisy inference has not yet completed.
