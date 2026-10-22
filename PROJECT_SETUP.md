# Project setup and working method

This is a regular project document. `AGENTS.md` routes agents here when setup or
workflow information is needed; this filename has no special auto-loading role.
Current progress and the next action belong in [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Environment and layout

- Primary checkout: `/home/kagan/cmpe586-turkish-punctuation`.
- Local environment: Conda `cmpe586`, Python 3.11, CPU PyTorch.
- Local interpreter: `/home/kagan/miniconda3/envs/cmpe586/bin/python`.
- Installation: [README instructions](README.md#yerel-kurulum), `environment.yml`,
  `pyproject.toml`. Use the existing environment rather than reinstalling it per chat.
- GPU runs: Colab T4, observed 2026.07 runtime (Python 3.12.13,
  Torch 2.11.0+cu128), `requirements/colab.constraints.txt` and recorded runtime pins.
  Local CPU and Colab packages intentionally differ; numerical comparisons check
  the recorded execution environment rather than silently substituting one.

| Location | Role |
| --- | --- |
| `src/turkish_punctuation/` | Shared data, noise, alignment, training, inference, metrics |
| `scripts/` | Reproducible CLI entry points and preparation/analysis tools |
| `configs/` | Versioned recipes, model identities and experiment plans |
| `notebooks/` | Input inspection and Colab orchestration; switches off in Git |
| `docs/` | Protocols, completed reports and aggregate figures |
| `data/`, `artifacts/` | Ignored data, weights, predictions and receipts |

Do not read large tensors/JSONL/Arrow files to learn the project structure.
Start with the status page, relevant configuration and a small manifest/receipt.

## Common commands

Activate the existing environment before invoking Python:

```bash
conda activate cmpe586
```

Commands relevant to the current ablation preparation:

```bash
python scripts/prepare_ablation.py audit
python scripts/prepare_ablation.py --help
```

GPU calibration/formal training use `05_case_loss_ablation_colab.ipynb` and the
commands assembled from `configs/ablation-case-loss.json`. Do not run an
unrequested training command to check whether preparation succeeded.

When formatting or import checks are useful, limit them to touched files:

```bash
python -m ruff check path/to/changed_file.py
git diff --check
```

Do not add or execute a test suite unless the user asks. Scientific input/target,
artifact/hash and saved-metric audits are described by their relevant protocols;
report their actual scope, rather than calling them a full implementation test.

## Start and finish a chat

1. Choose the existing project folder and a concrete outcome.
2. Read `AGENTS.md`, then `PROJECT_STATUS.md`; consult setup/protocol details only
   as needed. Check Git changes and existing/running outputs before doing work.
3. Make the requested change, perform the authorized checks, record evidence and
   update the current status. Save run details in their artifact/protocol location.
4. Report changed files, checks, remaining work and whether anything is running.

A useful new-chat prompt is:

> Read AGENTS.md and PROJECT_STATUS.md. My task is: [one concrete outcome].
> Continue from verified repository/artifact state and follow the recorded stop point.

Do not paste the entire old chat. Automatic compaction and `/compact` also support
long work within a chat; a new chat is useful when the outcome changes, not for
every small follow-up. Savings are not guaranteed: model, tools, reasoning,
caching and rediscovery affect usage.

## Parallel work

For concurrent code-writing chats, use separate Git worktrees and branches with
bounded tasks. Files provide guidance; worktrees provide checkout isolation.
Agree which chat owns integration and updates the shared current-status document.
The same Colab VM or Drive run directory must have one active writer.

Local ignored assets do not automatically follow a worktree. Inspect artifacts
before creating a worktree, reuse an existing suitable checkout when possible,
and archive managed worktrees when their work has been integrated.

The shared Conda environment's editable package can point at the primary
checkout. In a separate worktree, explicitly put that checkout's `src` first
when running its Python tools, rather than reinstalling the shared editable
package while another chat is using it:

```bash
PYTHONPATH="$PWD/src" python scripts/prepare_ablation.py --help
```

Run actual experiments from the designated experiment checkout with its verified
assets. A new worktree does not imply another training run or Colab session.

## Persistent assets and handoffs

Drive project root: `MyDrive/cmpe586-turkish-punctuation`.
Each run has a distinct versioned output; shared initialization is reused by hash.
Full epoch models and results from completed clean/augmented runs are locally
preserved under `artifacts/training/training-v1/.../completed-archive/`.
Follow [training protocol](docs/training-protocol.md) for retention and archiving.
Neither this document nor an agent chat substitutes for verified file presence.

## Official references

- [AGENTS.md discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
- [Focused chats and concise context](https://learn.chatgpt.com/docs/learn/best-practices)
- [Git worktrees](https://learn.chatgpt.com/docs/environments/git-worktrees)
- [Desktop slash commands](https://learn.chatgpt.com/docs/reference/slash-commands)
