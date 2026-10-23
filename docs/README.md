# Documentation guide

Start with the [paper](../output/pdf/turkish-punctuation-report.pdf) for the
complete research narrative and [verified held-out results](held-out-test-results.md)
for every planned condition. The public repository contains aggregate evidence;
raw text, models, predictions and operational receipts remain in ignored storage.

## Current methods and results

| Read | Purpose |
| --- | --- |
| [Experiment reproduction](reproduction.md) | Environment, frozen recipes, commands and required external artifacts |
| [Held-out results](held-out-test-results.md) | All main/ablation comparisons, sensitivity, figures and limits |
| [Held-out protocol](held-out-test-protocol.md) | Validation-informed decisions frozen before test scoring; literature adaptation |
| [Noise protocol](noise-protocol.md) | Operators, eligibility, exact budgets, composition and invariant checks |
| [Training protocol](training-protocol.md) | Matched initialization/budgets, row mixing, optimizer and clean selection |
| [Data audit](data-audit.md) | Released counts, empty-word quarantine and unresolved provenance/license |
| [Validation robustness grid](robustness-grid-results.md) | 117 matched main-model validation results |
| [Validation ablation](case-loss-ablation-results.md) | Five new results plus 15 reused matched references |

## Public numeric evidence

- [All held-out aggregate metrics and paired contrasts](held-out-test-aggregates.csv).
- [Condition metrics/supports/confusions](held-out-test-condition-summary.json).
- [Fixed contextual/direct-edit outcome partitions](held-out-test-diagnostics.csv).
- [Realized coverage and tokenizer counts](held-out-test-input-coverage.csv).
- [Cohort counts and ID/source hashes](held-out-test-cohort-summary.json),
  [input audit](held-out-test-input-audit.json).
- [Validation ablation table](case-loss-ablation-comparison.csv),
  [three-arm clean-validation history](case-loss-ablation-clean-validation.csv).
- [Source file manifest](data-manifest.json), [data counts](data-audit.json),
  [tokenizer audit](tokenization-audit.json), [figures](figures/).
- [Report tables and source binding](../report/README.md).

## Dated experiment and operational records

These record what was known at each stage. Statements such as “training/test is
pending” and “Drive outputs remain” describe that stage's handoff. They do not
override the [current state](../PROJECT_STATUS.md). Original recipes/snapshots
remain intact; completed experiments should not be rerun to refresh these pages.

| Stage | Records |
| --- | --- |
| Early project design | [Project brief/history](project-brief.md), [initial code review](code-review.md) |
| Initial validation | [Clean reference](clean-validation.md), [noise data audit](noise-audit.md), [published pilot](noise-pilot.md) |
| Clean training | [Calibration](training-calibration.md), [persistent preparation](preparation-status.md), [completion](clean-training-status.md), [backup](backup-verification.md), [medium pilot](clean-noise-pilot.md) |
| Augmented training | [Preparation/calibration](augmentation-preparation.md), [completion/backup](augmented-training-status.md), [medium pilot](augmented-noise-pilot.md) |
| Matched GPU validation | [Grid design/calibration](robustness-grid.md), [completed grid](robustness-grid-results.md) |
| Case-loss removal | [Preparation](case-loss-ablation-preparation.md), [training/archive](case-loss-ablation-training-status.md), [evaluation/archive](case-loss-ablation-evaluation.md) |
| Held-out execution | [Protocol freeze](held-out-test-preparation.md), [inputs/execution](held-out-test-execution.md), [launch](held-out-test-launch.md), [analysis/archive](held-out-test-completion.md) |
| Drive retention | [Initial storage inventory](drive-storage-audit.md), [verified cleanup](drive-cleanup.md) |

Maintainer workflow is in [PROJECT_SETUP.md](../PROJECT_SETUP.md); the concise
handoff is [PROJECT_STATUS.md](../PROJECT_STATUS.md).
