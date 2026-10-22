# Project guidance

## Start with a small amount of context

- Read `PROJECT_STATUS.md` first for the current milestone, evidence and next action.
- Use `PROJECT_SETUP.md` for environment, commands and workflow. Read only the
  source/configuration or protocol relevant to the requested task.
- Prefer current files and verified receipts over assumptions from another chat.
  Do not load the complete documentation/history or large artifacts by default.
- At a meaningful handoff, update `PROJECT_STATUS.md` with completed work,
  remaining work, result locations and any running job. Keep this file concise;
  detailed experiment history belongs in `docs/` and immutable run artifacts.

## Working with the user

- Explain in friendly Turkish; keep established English technical terms.
- Implement the work and explain important blocks/decisions. Do not assign exercises.
- List created/modified files after each task and explain their purpose. For many
  generated artifacts, provide a complete local inventory and summarize the key files.
- Do not add or run tests unless requested. State which checks actually ran;
  never imply an unperformed test, experiment or verification passed.

## Research invariants

- This project studies Turkish punctuation restoration under controlled writing noise.
  Use the existing four labels, first-subword alignment and ignored-label padding.
- Preserve original word targets/IDs under noise and re-tokenize changed words.
  Keep train, validation and held-out test roles separate.
- Configurations, source hashes, initialization, row masks and run identities make
  comparisons reproducible. Do not weaken a hash/runtime check to make a run pass.
- Preserve completed experiments and their frozen recipes. Version changes that
  would invalidate comparison/resume; do not rewrite historical source snapshots.
- Noisy validation does not select checkpoints in training-v1. Follow the recorded
  clean-validation selection rule and report the single-training-seed limitation.
- Never fabricate results. Dataset licensing and exact split provenance remain
  unresolved; do not claim they have been verified without source evidence.

## Compute, storage and Git

- Use the existing `cmpe586` environment and shared training/inference functions.
  Put reusable logic in `.py`; notebooks orchestrate runs and display results.
- Long formal training requires the user's instruction to start it. Preparation
  and a separately authorized technical calibration are not a completed experiment.
  Commit notebooks with execution switches off, outputs cleared and counts null.
- Data, weights, predictions and run receipts stay in ignored artifact directories.
  Commit aggregate reports/figures and code/configuration; never commit secrets.
- Reuse audited inputs. Do not regenerate/upload/download large data or weights
  when a verified suitable copy exists. State what a long operation is doing.
- After completed training: prepare an archive, let the user download it, verify
  local models/results, then perform only the agreed scoped Drive cleanup.
  Do not add account-wide Trash cleanup or permanent deletion automation.
- Inspect Git state before edits; preserve unrelated changes. Do not reset,
  discard, overwrite completed runs or change frozen files just to clean the tree.

## Multiple chats and agents

- Use one chat per coherent outcome; a follow-up on that outcome can stay there.
- When independent writing tasks run concurrently, give each chat a separate
  worktree/branch (`codex/` prefix) and clear file ownership. Read-only review can
  share the checkout. Do not assume Markdown files provide locking or isolation.
- Do not let separate chats control the same Colab runtime, mutate the same Drive
  outputs or rewrite `PROJECT_STATUS.md` concurrently. Integrate status updates
  with the code changes, resolving them against actual artifacts and Git state.
- Ignored local data/models are not copied by Git into a new worktree. Reuse a
  suitable read-only input location explicitly; do not silently duplicate weights.
- A shared editable Python install can still resolve to the primary checkout.
  Use the intended checkout's `src` on `PYTHONPATH`; do not rebind the shared
  environment while another chat uses it. Keep experiments in their designated checkout.
