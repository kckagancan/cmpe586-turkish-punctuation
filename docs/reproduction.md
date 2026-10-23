# Reproduction guide

The experiments are complete. Reading the paper, rebuilding its tables/figures
and checking repository contents require no new training or inference. Exact
scientific reruns are a separate, compute-intensive workflow requiring inputs
and model artifacts that are deliberately excluded from Git.

## 1. Inspect and reproduce the public report

Start with [README](../README.md), [the paper](../output/pdf/turkish-punctuation-report.pdf),
[full held-out results](held-out-test-results.md), and [documentation guide](README.md).
With Python 3.11+, the following use only the standard library:

```bash
python scripts/build_report_tables.py --check
python scripts/check_repository.py
```

To rebuild the tables, omit `--check`. To redraw the dedicated paper figures
from public aggregate CSVs, install the project's plotting dependencies and run:

```bash
python scripts/build_report_figures.py
```

See [report instructions](../report/README.md) for LaTeX/Tectonic compilation.
No raw data, tokenizer, GPU, model weights or access to the author's Drive is
needed for report reproduction. The published PDF omits private course metadata.

## 2. Local development environment

The project uses the existing `cmpe586` Conda environment, Python 3.11 and CPU
PyTorch locally. A fresh development checkout can be installed with:

```bash
conda env create -f environment.yml
conda activate cmpe586
python -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python scripts/check_setup.py
```

If the environment already exists, reuse it. VS Code/Jupyter must select that
interpreter explicitly. `PROJECT_SETUP.md` gives maintainer-specific paths and
workflow. Numerical comparison requires the run's recorded packages/hardware,
not simply a fresh installation satisfying these broad development ranges.

## 3. Data and published assets

The [data manifest](data-manifest.json) pins the three original CSV URLs at
revision `7ede7a4ba9e51dad085a7f0d9b6c0a9c32622397` and records exact SHA-256s.
Dataset licensing and precise derivation/split provenance remain unresolved;
the corpus is not redistributed here. The [data audit](data-audit.md) describes
the whole-row quarantine and overlap limitations. Existing suitable local
copies should be reused instead of downloaded or regenerated.

Relevant preparation entry points are:

```bash
python scripts/download_data.py --help
python scripts/prepare_data.py --help
python scripts/inspect_tokenizer.py --help
python scripts/prepare_noise.py --help
```

The tokenizer/published model and BERTurk initialization revisions are in
`configs/tokenizer.json`. Loading weights is separate from inspecting a config.
The authored examples in `scripts/inspect_model.py` are illustrative and not a
performance measurement. Historical validation commands intentionally remain
validation-only; changing a split setting does not create a valid held-out run.

## 4. Frozen recipes and execution order

| Stage | Recipe / protocol | Orchestration |
| --- | --- | --- |
| Clean fine-tuning | `configs/training.json`; [training method](training-protocol.md) | `scripts/prepare_training.py`, `scripts/train.py`, notebook 02 |
| Augmented fine-tuning | `configs/training-augmented.json`; [augmentation](augmentation-preparation.md) | `scripts/prepare_augmentation.py`, shared `scripts/train.py`, notebook 03 |
| Main validation grid | `configs/robustness-grid.json` plus three model configs | `scripts/run_robustness_grid.py`, `scripts/analyze_robustness_grid.py`, notebook 04 |
| Case-loss removal training | `configs/ablation-case-loss.json`, `configs/ablation-case-loss-inputs.json` | `scripts/prepare_ablation.py`, shared training functions, notebook 05 |
| Removal validation | `configs/ablation-case-loss-evaluation.json` | preparation/run/finalize/report ablation scripts, notebook 06 |
| Held-out freeze / inputs | `configs/held-out-test.json`; [frozen protocol](held-out-test-protocol.md) | `scripts/freeze_heldout_protocol.py`, `scripts/prepare_heldout_inputs.py` |
| Held-out readiness / execution | `configs/held-out-execution.json`; [launch](held-out-test-launch.md) | `scripts/ready_heldout_evaluation.py`, `scripts/evaluate_heldout.py`, notebook 07 |
| Saved-prediction analysis / archive | [completion](held-out-test-completion.md) | `scripts/analyze_heldout.py`, `scripts/archive_heldout_results.py`, `scripts/report_heldout_results.py` |

All fine-tuned arms use training seed586, shared saved initialization and three
epochs / 10,968 updates. Selection is clean-validation-only. All selected models
are epoch3; the equal-update final comparison therefore needs no extra inference.
The published reference has uncontrolled prior training.

Notebook execution switches are off in Git. Formal runs require prepared assets,
readiness and deliberate execution; opening a notebook should not start them.
Do not run these stages merely to inspect results. Existing outputs reject
overwriting. Historical phase/status strings in frozen configs describe their
creation and remain unchanged to preserve source/config identities.

## 5. Runtime and external artifact requirements

Training was FP32/eager on a Tesla T4. Matched validation/held-out inference used
FP32 SDPA/math, TF32 off, deterministic algorithms, batch16, four CPU threads,
and shared clean-length/source-ID batch membership. The observed Colab runtime
used Python3.12.13 and PyTorch2.11.0+cu128. Recorded constraints, runtime receipts
and model tensor hashes are binding; the local CPU environment is not an
interchangeable numerical substitute.

Full reproduction requires the exact separately retained raw/processed inputs,
initialization and selected model exports, protocol masks, readiness/run
receipts, and historical source snapshots. Config paths describe these external
assets; their absence from a fresh clone is expected. This repo alone does not
provide the private raw-prediction audit or an end-to-end model rerun.

The locally verified final result ZIP contains 570 files plus manifest, SHA
`2463a23ed22dc9543341393bedfb902f693448330da860374c708748d4830925`.
Clean/augmented/removal training archives preserve all three epoch models and
shared initialization separately. Completed Drive-copy cleanup does not remove
local archives. See [completion](held-out-test-completion.md),
[artifact map](../artifacts/README.md) and [cleanup](drive-cleanup.md).

## Checks and historical source preservation

Repository checks parse source/JSON/notebooks, validate local public links and
generated report bindings, and reject data/model/private-metadata files in the
publication set. They do not import training modules, download assets, execute
notebooks, or run the historical unit suite. The
[Repository checks workflow](../.github/workflows/repository-checks.yml) runs on
pushes and pull requests with read-only repository permissions, Python 3.11,
pinned action revisions and Ruff 0.17.1. It installs only the lightweight linter;
the project/training dependencies and model/data assets are not installed.
The same checks are available locally:

```bash
python scripts/check_repository.py
python -m ruff check --select E9,F scripts src tests
```

Full-style lint has seven existing
findings in two frozen historical grid/calibration scripts; those sources are
preserved rather than rewritten for formatting. New publication scripts pass
the project's complete Ruff rules. The initial workflow proposal was retained
locally when the connection lacked publication permission; the user subsequently
granted `workflow` scope and authorized its activation.

Claims are limited to the recorded checks. Fresh test-informed recipes must use
new versions and be identified as exploratory; do not edit frozen snapshots or
repeat completed runs to make a publication check pass.
