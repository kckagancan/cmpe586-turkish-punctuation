# Held-out Colab restoration and launch

Date: **2026-10-23** (Europe/Istanbul). This continues the locally completed
[input/execution preparation](held-out-test-execution.md). The fixed scientific
protocol, 39 input views and 122-condition matrix are unchanged.

## Runtime restoration evidence

The old Colab runtime had disconnected. A separate project notebook was uploaded
and opened: [07 held-out notebook](https://colab.research.google.com/drive/1SxfoAohLpM2UoLsz3yBe5M3nwgqDlO_P).
T4 and runtime version `2026.07` were explicitly selected. Actual base runtime:
Python 3.12.13, Torch 2.11.0+cu128, CUDA 12.8, cuDNN 91900, Tesla T4 / capability
7.5. These match the historical GPU comparison.

The fresh image supplied Transformers **5.13.1**, although the experiment pins
**4.57.6**. The separate restoration helper restored the recorded constraints,
including Transformers 4.57.6 / tokenizers 0.22.2; Torch/CUDA were retained.
Datasets 4.0.0, NumPy 2.0.2 and scikit-learn 1.6.1 already matched. No runtime/hash
gate was loosened. Pip reported a conflict with the fresh image's unrelated
Gradio 6.20.0, which expects a newer Hub; the project does not import/use Gradio.
The recorded Transformers release requires Hub <1, so the Hub was restored to
0.36.2. Actual project execution remains subject to the full readiness gate.

The source and test-input ZIPs were uploaded once to the project-specific
`preparation/held-out-test-v1/transfer-v1/`. Their SHA-256 values are unchanged:

- Source: `115877201eb35cb81f3e24e870a9a0acabc44e42933b399518db56ed10ebb9b2`.
- Test inputs: `18953b8eb131c18d261aee909ba7d1092f4178d2384b02d2139eedb4038159f1`.
- Existing validation ZIP reused:
  `579a6a6ff5969ce0406ef7bbd057a5fe56c5058ebb17e6cb4918e6dc360c97eb`.

Full member/restoration checks passed for 33 source files, 312 input files and
285 historical validation files. Data/noise were not regenerated. Fine-tuned
weights were not uploaded/downloaded again; selected models remain on Drive.
The expired runtime's published-checkpoint cache was absent, so only that public
checkpoint/tokenizer was restored from the previously pinned revision. Its
actual tensor digest matched
`391cf196c87a2a1a03bce3516f1ba933acb4cf7d10919f91ea141512f7232bc5`.

## Preparation correction preserved before scoring

The first cold-restoration helper restored the files/packages, then stopped with
`ModuleNotFoundError`: the process had cached a negative importer for the initial
nonexistent `PYTHONPATH` directory. Revision 2 explicitly invalidates import
caches after source restoration. The original helper/log remain preserved; no
frozen source or completed experiment was edited. A temporary mounted-Drive
visibility delay also stopped the first revision-2 invocation before execution;
the file was subsequently observed and its exact hash verified. The refresh
cell found the file already visible, so it did not unmount Drive.

- Initial helper SHA:
  `0492397c69ec94cbcdaee71fd5937bdcf102693819bb4060b5876252f675d486`.
- Active `scripts/restore_heldout_colab.py` / Drive `restore_heldout_colab-r2.py`:
  `063f3939f2b91124d3b565c531978dbe82e54e58fc84f36097d4f9c05af75554`.
- Reused historical validation restorer SHA:
  `59897a4a17ae9a3e29aceb7d07ce0590ec949dfd50b42b087b5ca20eeab2c4bd`.
- Successful `restoration-r2.json` SHA:
  `ee01ad1b73271e9a5e9b54052634e8239e056b269fd56ff00372147c3dc8cdd0`.

Restoration completed at **12:12:34 Istanbul time**, with zero forward passes,
training updates or test predictions/scores. It is not a readiness result.
`scripts/restore_heldout_colab.py` contains reusable restoration logic; the local
`07` notebook now includes its separate disabled orchestration step. Local
execution switches remain off and outputs/counts are cleared.

## Readiness and formal launch

Actual validation-only GPU readiness **completed successfully**. Its output
directory is `preparation/held-out-test-v1/runtime-v1/`. All 39 full test
re-tokenizations matched; the four actual selected models passed file/tensor
checks and eight validation-only probes (245 fixed-batch rows and 16 padded rows
per model). The maximum padded probe length was 196, covering the test maximum
172 and validation maximum 196. No optimizer update or test prediction/score
was produced by readiness.

- Readiness identity:
  `f52980b75c4fafaac6138613f131eb1ca5d6b66a5436689fa4f424588793f2c7`.
- Readiness JSON SHA:
  `3a102bb3f03b2d0c91acd65648d5876d540cc7026685ab89a6980fdb80588b46`.
- Readiness commit SHA:
  `a9fd6efbfaa77c463617e6d7ab6aa3112edc98361f4bd8c7afb284ea904bafcc`.
- The original Drive bytes were downloaded locally. Independent receipt checks
  compare the identity, complete marker's exact expected membership/hashes,
  current local source/config/input hashes, the recorded reference runtime and
  packages, and all four model identities / eight probe records.

The formal 122-condition process was started at **12:25:56 Istanbul time**,
PID **7592**, with the explicit `--allow-test-scoring` flag. The one-shot launch
switch resets to false before the process starts. The runner reverified all 39
input views and all four actual checkpoints before its first test clean anchor.
It writes streaming logs and hash-committed condition outputs to Drive.

The notebook's launch wrapper independently audited the first five published
conditions (clean plus all four mild/seed586 families) at **12:31:22 Istanbul**.
Every condition retained 14,600 original rows, 181,485 primary targets and 181,116
sensitivity targets; both saved cohort metrics and prediction/commit hashes
matched recomputation. No F1 values were printed or used for selection.

- Run identity: `60e31867535de346accb64be45df04984434f6ff328b31c8819115cb689538a0`.
- Initial observation SHA:
  `8f132dcdadd93d469c04480dd7cbf798adefeacc961c681f8d36a643d4a4b5b9`.
- At **12:38:32 Istanbul**, **16/122 conditions** were visibly committed and the
  next condition was processing 14,000/14,600 rows. No traceback appeared in the
  active output. This is a timestamped startup observation, not full completion.
- Colab was left visible, the tab marked to remain open, and a screenshot saved.
  No monitoring automation or wait through the full run was created.

Completion of the full matrix and final saved-prediction analysis/archive remain
pending. When the user reports completion, inspect persisted progress and all
122 commits, run the analyzer, then archive and verify locally. Do not start a
second writer or silently change the frozen run recipe to resume.

Operational local evidence:
`artifacts/evaluation-preparation/held-out-test-v1/launch-2026-10-23/`.
The same dated Drive preparation directory holds streaming logs and receipts.
Previous local preparation receipts and immutable source snapshots remain intact.

## Future experiments and test feedback

Additional ablations remain possible. Develop/select them on train/validation,
record an independent rationale and freeze the recipe before evaluating. Record
post hoc additions as additions; do not present them as part of the original
pre-test matrix. Aggregate test F1 can influence selection even without opening
raw test examples. Experiments motivated or selected by this test's scores/errors
must be identified as test-informed exploratory work; a fresh unseen evaluation
set is needed for a strong independent confirmation of those subsequent choices.
The current matrix is not expanded or tuned based on interim test performance.

## Subsequent completion

The formal run completed **122/122 conditions at 13:43:43 Istanbul on October 23**.
The unchanged Colab analyzer subsequently verified 122 original-target condition
outputs / 244 cohort scores without repeating inference. This dated launch
record remains historical; current analysis/archive evidence is in the
[completion record](held-out-test-completion.md) and [results report](held-out-test-results.md).
