# Held-out test: completion, verification and archive

Completed **2026-10-23**. This record supersedes the running-process handoff in
[the launch record](held-out-test-launch.md). Experimental decisions and frozen
source snapshots were preserved; no training or completed inference was repeated.

## Observed completion

- Formal inference started at **12:25:56 Istanbul** and persisted
  `inference_complete`, **122/122 conditions**, at **13:43:43 Istanbul**.
  Elapsed wall time from the saved timestamps: approximately **77.8 minutes**.
- Run identity:
  `60e31867535de346accb64be45df04984434f6ff328b31c8819115cb689538a0`.
- Published, clean-trained and augmented arms each committed 39 conditions;
  the without-case-loss arm committed its frozen five conditions.
- Primary scores use **14,600 rows / 181,485 original word targets**; sensitivity
  scores filter the same predictions to **14,445 rows / 181,116 targets**.
  No additional inference or reduced-cohort re-noising occurred.
- The unmodified frozen analyzer completed **122 raw-target audits / 244 cohort
  score sets** on Colab. It checked original targets/order, condition commit and
  prediction hashes, supports/confusion matrices, padded batch work, own-clean
  drops, paired outcome counts and deterministic source-ordered error samples.
- Analysis identity:
  `e39f8803a6e6614363a6db97c1edce870757a2e909dec9dba5e93d0d9edc7f91`.

The first analysis-launch wrapper initially checked the generic status string
`complete`; the evaluator actually writes `inference_complete`. That wrapper
stopped before the analyzer started. Its check was corrected to the evaluator's
actual status plus exact 122/122 counts. No frozen code, hash gate, inference
result or experimental choice changed.

## Results and interpretation

The [complete results report](held-out-test-results.md) includes both cohorts,
all frozen condition groups, macro/micro and per-class metrics, paired deltas,
own-clean drops, fixed diagnostics and exportable figures.

Primary clean F1 is **84.3437%** for clean training and **84.4824%** for
augmentation (**+0.1387 pp**). Combined-heavy three-corruption-seed means are
**76.7502%** and **80.5289%**; the paired difference is **+3.7787 ± 0.2191 pp**.
The sensitivity paired difference is **+3.7951 ± 0.2178 pp**.

For the medium/seed586 ablation, augmented minus without-case-loss F1 is
**+0.0905 pp clean**, **+0.9585 case loss**, **−0.0205 diacritic loss**,
**−0.1078 typo swap**, **+0.8799 combined**. These five differences use the
matching seed586 augmented scores, not its three-seed means. Unlike the earlier
validation result, augmentation has a small positive clean-test difference.
The validation and held-out tables must remain separately identified.

One training seed remains a limitation. Corruption-seed SD is not training
uncertainty or statistical significance. Omitting case loss lowers total noise;
the result does not estimate equal-dose component importance. Exact-input
filtering does not establish document separation; licensing, split provenance
and published-model prior exposure remain unresolved.

## Transfer and local verification workflow

A proposed upload of the new archive helper source was rejected by automatic
approval review because that additional source export was not specifically
authorized. No helper was uploaded and the rejection was not retried. Colab's
standard-library ZIP operation instead archived the **existing** completed Drive
run; the new verification/reporting helpers stayed local.

The original Drive transfer ZIP contains **526 files**, **15,826,627 bytes**,
SHA-256:
`a60914a12c1fbb9eb5a0274ead9c9a0178bc1a3e65460ed65b44090276f6e0ae`.
Its CRC check and post-copy Drive SHA readback passed. The connector downloaded
the ZIP, the exact archive receipt and the complete Colab analysis log. Local
byte count/SHA matched; safe member paths and every restored condition/provenance/
analysis hash were checked. Different existing local files are never overwritten.

The local workflow uses the same frozen analyzer on the existing audited local
inputs, without a model load. `archive_heldout_results.py verify-local` requires
complete equality of the two analyses except their creation timestamps, exact
CSV/Markdown equality, and the original readiness/freeze bindings before writing
the final manifest-bearing results ZIP. The final local verification receipt,
archive size/hash and inventory are retained at the locations below.

Local CPU analysis **completed all 122 conditions / 244 scores**. The analysis
objects matched exactly except their creation timestamp; aggregate CSV and
Markdown were byte-identical. Original readiness/freeze bindings also passed.
The final local ZIP has **570 source files plus manifest**, **16,011,054 bytes**,
SHA-256 `2463a23ed22dc9543341393bedfb902f693448330da860374c708748d4830925`.
All manifest members passed complete byte-count/SHA readback. The final local
verification receipt was written at **2026-10-23T11:11:03 UTC**.

The final local ZIP adds the original protocol/row-mask freeze, existing local
readiness/launch receipts, the local analysis log and the verification helper.
It includes no model weights or optimizer state; the previously verified model
archives remain separate. Each ZIP member has a byte count and SHA-256 in
`archive-manifest.json`, and every member is read back for verification.

## Locations and commands

- Original Drive run: project root + `artifacts/evaluations/held-out-test-v1/`.
- Local run: `artifacts/evaluations/held-out-test-v1/`.
- Local completion evidence:
  `artifacts/evaluation-preparation/held-out-test-v1/completion-2026-10-23/`.
- Final local results ZIP: that directory + `held-out-test-v1-results.zip`.
- `remote-archive.json`, `restoration.json`, `local-verification.json`:
  exact transfer/archive/scientific verification receipts.
- `colab-analysis.log`, `local-analysis.log`, `local-analysis/`: independent
  analysis records. `inventory.json` and `files-changed.md`: full local inventory.
- `report-generation.json`: report inputs, output hashes and diagnostic counts.
- Colab stays open at its completion summary:
  <https://colab.research.google.com/drive/1SxfoAohLpM2UoLsz3yBe5M3nwgqDlO_P>.

```bash
PYTHONPATH="$PWD/src" python scripts/analyze_heldout.py \
  --input-root "$PWD" \
  --run-directory artifacts/evaluations/held-out-test-v1 \
  --output artifacts/evaluation-preparation/held-out-test-v1/completion-2026-10-23/local-analysis

PYTHONPATH="$PWD/src" python scripts/archive_heldout_results.py verify-local \
  --local-analysis artifacts/evaluation-preparation/held-out-test-v1/completion-2026-10-23/local-analysis

PYTHONPATH="$PWD/src" python scripts/report_heldout_results.py
```

These commands audit saved predictions and produce aggregate reports; they do
not perform inference. Completed audits need not be repeated without a reason.
No test suite was requested or run. No Drive cleanup, account-wide Trash action,
permanent deletion, Git reset or new training was performed in this outcome.
The protocol/config's historical preparation-phase wording remains frozen;
current completion is recorded here and in `PROJECT_STATUS.md`.
