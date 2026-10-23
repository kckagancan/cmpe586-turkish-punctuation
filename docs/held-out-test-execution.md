# Held-out execution preparation

Prepared locally **2026-10-22**. The protocol/cohort freeze is complete; all test
inputs are prepared and audited. **GPU readiness and test scoring remain unrun.**
No new training or completed validation experiment was repeated.

## Fixed plan and separate implementation

[The frozen protocol](held-out-test-protocol.md) and
[`held-out-test.json`](../configs/held-out-test.json) remain byte-identical to
freeze revision 2:
`ba6bf8decbd62a64f898a4eb47054776a97b3131fc8c6a20d8a50e22c6dc9653`.
The protocol config's `phase` and `current_step_generates_noise` describe that
historical freeze step. This subsequent implementation is wired separately by
[`held-out-execution.json`](../configs/held-out-execution.json); those historical
fields were not rewritten to imply execution or readiness.

The existing validation-only commands/guards, noise recipe and shared numerical
implementations were preserved. New entry points:

| File | Responsibility |
| --- | --- |
| `src/turkish_punctuation/heldout.py` | Frozen-plan, input-tree, original-target, tokenizer and selected-export gates |
| `src/turkish_punctuation/heldout_runtime.py` | Exact GPU binding, shared clean-length order, two-cohort scoring and fixed diagnostic samples |
| `scripts/prepare_heldout_inputs.py` | Generate the 39 test views once; no prediction model |
| `scripts/ready_heldout_evaluation.py` | Re-encode inputs; check four actual checkpoints and matching T4 runtime; validation-only technical probes |
| `scripts/evaluate_heldout.py` | Explicitly enabled 122-condition inference; per-condition hash commits and verified resume |
| `scripts/analyze_heldout.py` | Recompute 244 cohort score sets, paired outcomes, examples and fixed aggregates from saved predictions |
| `scripts/package_heldout_execution.py` | SHA-pinned source/input bundles, full member readback and safe restoration/reuse |
| `notebooks/07_held_out_test_colab.ipynb` | Orchestration; all execution switches off, outputs empty, counts null |

## Completed input audit

Prepared root: `data/processed/noise-held-out-test-v1/`.
Input identity: `fce0d2c6f8a47ffaf012e55af5dd34d84b2089b8bbac73ae8ea80208f1fe8e19`.
Its immutable tree contains **274 files plus `complete.json`**, including exact
preparation-source snapshots. Preparation took about 139 seconds on the local CPU.

- **39 views:** original clean anchor, 36 family/level/seed variants, two full-loss
  stresses. Noise plans explicitly use split key `test`.
- Every view retains **14,600 original rows / 181,485 word targets**; clean
  tokenization exactly reproduces the previously saved clean-v1 test dataset.
- Same word IDs, order, targets, word-slot counts and first-subword supervision.
  No missing word, unknown subword or overlength row in any view. No truncation
  or condition-specific exclusion. Maximum length including special tokens: **172**.
- Deterministic budget masks, nested severity prefixes and equality of combined
  versus single-family component masks were checked during preparation.
- Sensitivity uses the **same noisy words/edits**, filtered by frozen IDs:
  **14,445 rows / 181,116 words**. It has no new plans or inference.
- After publication, a separate pass verified the exact committed file set,
  original targets and all **39 full re-tokenizations**, comparing every saved
  encoded column. It exited successfully; no model weights were loaded.

Aggregate evidence without raw texts/row masks:
[input audit JSON](held-out-test-input-audit.json),
[coverage CSV](held-out-test-input-coverage.csv).
Detailed raw inputs, edits and examples remain ignored.

## Verified local transfer

Directory: `artifacts/evaluation-preparation/held-out-test-v1/transfer-v1/`.

| Archive | Bytes | Source files | SHA-256 |
| --- | ---: | ---: | --- |
| `held-out-inputs-v1.zip` | 72,979,242 | 312 | `18953b8eb131c18d261aee909ba7d1092f4178d2384b02d2139eedb4038159f1` |
| `held-out-source-v1.zip` | 74,440 | 33 | `115877201eb35cb81f3e24e870a9a0acabc44e42933b399518db56ed10ebb9b2` |

Each ZIP also has one `bundle-manifest.json`. Full archive/member SHA readback
passed. The restore command was run against the existing verified local files;
all 312 input and 33 source files matched and were reused without duplication.
Restoration rejects duplicate/unsafe members, hash differences and differing
existing destination files. Exact existing files are reused.

The inputs bundle contains prepared test views, original test CSV, active freeze
and the exact pinned local copy of the historical GPU calibration receipt.
It contains no weights or predictions. The source bundle contains the required
frozen core/config/docs and new entry points. Neither bundle was uploaded this
step. Reuse the existing validation input bundle/cache and retained Drive
epoch-03 models; do not upload models again.

Historical validation bundle:
`artifacts/evaluation-preparation/robustness-grid-gpu-v1/validation-inputs-noise-v1.zip`,
SHA `579a6a6ff5969ce0406ef7bbd057a5fe56c5058ebb17e6cb4918e6dc360c97eb`.
Its original validation CSV is included. The planned cache root is
`/content/cmpe586-evaluation-inputs-noise-v1`; actual presence must be checked.

## Runtime gates and reporting behavior

Readiness requires the exact recorded T4/Python/Torch/CUDA/package/numerical
policy, all original shared numerical source hashes, every prepared input hash,
and actual selected model files/tensors. Its technical probes use fixed clean
validation batches plus validation words padded to the maximum planned tensor
length. They produce no test predictions, quality metrics or optimizer updates.
An identical completed readiness can be reused. A local preparation receipt is
never substituted for actual GPU readiness.

Before any scoring, the runner checks all four actual checkpoints again. It
derives one clean-length/source-ID order shared across every model/condition,
records padded-token work and creates a fresh test clean anchor for each model.
Sensitivity scores filter those same saved predictions. Noisy scores cannot
select checkpoints. Completed condition files are copied/read back before their
commit marker is written; only matching verified commits can be reused.

The analyzer requires all **122** condition commits and rechecks original word
targets, both cohort metrics, clean-to-noisy drops, paired outcomes, prescribed
source-order diagnostic samples and padded-token work. It retains per-class
P/R/F1/support/confusion matrices and emits full JSON, aggregate CSV and Markdown.
Mean/sample SD uses the three corruption seeds; clean/full and the five-condition
seed586 ablation have no seed SD. Paired model deltas are formed per seed before
aggregation. Raw qualitative examples stay ignored. No global pooled robustness
score, p-value, equivalence or training-seed stability claim is introduced.

## Checks actually run and next step

Completed: source/config/freeze/hash audits, clean-v1 reproduction, all 39
preparation/target/noise audits, independent full re-tokenization, ZIP/member and
local restoration readback, Python AST/import/help checks, Ruff checks, notebook
format/switch/output checks and `git diff --check`. **No test suite was added or
run.** The GPU paths have not yet executed; syntax/import checks do not establish
their runtime readiness.

Operational logs and full file inventory:
`artifacts/evaluation-preparation/held-out-test-v1/input-execution-2026-10-22/`;
preparation, verification and transfer logs are in its parent directory.

Next: inspect the current Colab connection/GPU/cache/storage, transfer only the
two verified bundles if absent, restore without changing mismatched files,
then run the separate validation-only readiness. Do not claim readiness or start
the formal held-out matrix before that actual check succeeds. The notebook's
test switch and CLI `--allow-test-scoring` are separate explicit controls.
After formal inference, run the saved-prediction analyzer and archive/verify the
results before considering any separately scoped Drive cleanup.

No Colab/Drive operation ran here. No background job remains. Working-tree
changes remain uncommitted. Licensing, exact split provenance, document
independence, published-model prior exposure and the single training seed remain
the recorded limitations.

## Subsequent GPU execution — 2026-10-23

The preceding sections record the completed local preparation step on October
22. On October 23 the expired runtime was restored, actual T4 readiness passed
and the unchanged 122-condition held-out process was launched. The earlier
"unrun" statements describe the local preparation handoff. Current execution,
correction evidence and running-job information:
[held-out Colab launch](held-out-test-launch.md) and
[PROJECT_STATUS.md](../PROJECT_STATUS.md).
