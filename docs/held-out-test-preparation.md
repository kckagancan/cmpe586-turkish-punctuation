# Held-out protocol freeze: preparation record

Completed **2026-10-22** before project test scoring. This is a protocol/cohort
freeze, **not runtime readiness or a completed test experiment**.

- [Protocol and literature adaptation](held-out-test-protocol.md).
- [Machine-readable fixed plan](../configs/held-out-test.json).
- [Aggregate structural audit](held-out-test-cohort-summary.json).
- Helper: `scripts/freeze_heldout_protocol.py`.

## Structural audit

The pinned raw train/validation/test file hashes matched the original manifest.
The current shared reader and `clean-v1` whole-row quarantine policy reproduced
all test eligible/quarantined IDs exactly. Original words, IDs and tags were not
modified. Only input token lists determine the exact-overlap sensitivity mask.

| Cohort | Rows | Original word targets |
| --- | ---: | ---: |
| Primary clean-v1 eligible test | 14,600 | 181,485 |
| Sensitivity: no exact train/validation input match | 14,445 | 181,116 |

There are 61 quarantined raw test rows. Among the primary eligible rows, 148
match train inputs, 90 match validation inputs, and 83 match both. The union
excluded from sensitivity is **148 + 90 − 83 = 155 rows**. The earlier raw-split
[data audit](data-audit.md) reports 91 validation-overlap test rows; that count
includes a quarantined row, whereas the new 90-row count concerns eligible test
rows. The historical audit was not rewritten or treated as a failed check.

Full ID lists remain ignored. The sensitivity IDs are a strict subset of the
primary IDs, with the same source order; neither cohort is chosen using labels
or predictions. Future sensitivity scores must filter saved primary predictions
without rebuilding noise or running a second set of model predictions.

## Frozen model metadata and matrix

All four reference config/summary hashes matched. For the three fine-tuned
models, saved training specifications, run identities, selected export metadata,
shared initialization and clean-validation selection were checked. All select
epoch 3 / 10,968 updates. Existing selected export files are present.

Weights were not reopened, rehashed, loaded or downloaded; their expected file
and tensor hashes are preserved in the plan for the actual runtime gate. The
published model's earlier test exposure remains unknown.

The matrix contains 39 conditions each for published/clean/augmented and five
for the case-loss ablation, mirroring the completed validation scope. This is
**122 model-condition evaluations**, or 244 score sets when the fixed sensitivity
view is reported from the same predictions. No test noise variants were generated.

## Immutable preparation evidence

Freeze identity:
`ba6bf8decbd62a64f898a4eb47054776a97b3131fc8c6a20d8a50e22c6dc9653`.

Local directory:
`artifacts/evaluation-preparation/held-out-test-v1/protocol-freeze-r2/` contains
`freeze.json`, exact `cohorts.json`, aggregate counts, source/config/model-metadata
snapshots and a last-written `complete.json` with file hashes. The freeze covers
protocol/config bytes, implementation sources, raw source hashes, original
clean-manifest hash, model metadata and row masks. Do not edit frozen snapshots.

The config and protocol keep separate primary and sensitivity results, original
noise budgets/seed identities, first-subword alignment, ignored padding, word
coverage checks, matched GPU batches and descriptive corruption-seed statistics.
The literature informed these decisions; no new architecture, dataset, score
selection rule or tuning stage was copied into the completed training recipe.

Actual checks: Python syntax parsing, changed-script lint, config/matrix
consistency, original raw-source/clean-manifest hashes, model-selection metadata,
cohort structure and frozen artifact/source readback. No test suite, training,
model inference or test metric computation was performed. No Colab or Drive
operation was needed for this outcome.

## Next implementation

Existing preparation/evaluation commands intentionally enforce validation-only
scope. Implement separate held-out input preparation/orchestration using shared
functions, audit/re-tokenize all test variants against these fixed IDs, and bind
new source/input/runtime/model identities to this freeze. Reuse retained Drive
models and audited local assets. Do not replace the old split guards, regenerate
validation views, retrain models or report a protocol freeze as runtime ready.

Document-level independence, dataset permissions/provenance, published-model
prior exposure, one training seed and synthetic-noise generality remain open
limitations. The test split already had structural inspection and clean input
preparation before this freeze; it remains unused for project model scoring.

## Pre-scoring editorial revision

The first freeze had a citation-section typo: the relevant Nagy et al. (2021)
section is **4 Results**, not 4.2. Active freeze revision 2 corrects that pointer
and records its predecessor. The complete initial `protocol-freeze/` snapshot is
retained alongside `protocol-freeze-r2/`. The config's `supersedes_freeze` records
the original identity and reason. All model/noise/cohort/metric/inference/matrix
choices and the full row masks were verified identical across revisions. No
model scoring occurred under either revision.

## Subsequent execution preparation

The separate input/runner preparation subsequently completed on 2026-10-22:
39 test views passed independent re-tokenization, transfer bundles were verified,
and separate readiness/inference/analysis code plus the `07` notebook were added.
This does not change the historical checks above or the immutable freeze.
That October 22 step did not include actual GPU readiness or test scoring.
Input/execution file map: [held-out execution preparation](held-out-test-execution.md).
On October 23 actual GPU readiness passed and the held-out process was launched;
current handoff: [Colab launch record](held-out-test-launch.md).
