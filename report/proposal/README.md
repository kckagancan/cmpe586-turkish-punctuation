# Project proposal: retrospective design reconstruction

**Author:** Kagan Can · **Course:** CMPE 586 · **Prepared:** 23 October 2026.

[Public proposal PDF](../../output/pdf/turkish-punctuation-proposal.pdf) ·
[Editable source](main.tex) · [Completed final report](../README.md).

This five-page document uses future tense to explain the project's research
questions, hypotheses, data handling, controlled noise, matched training,
case-loss removal ablation, evaluation, risks, and staged work plan.
It excludes experimental performance results.

**Chronology:** it was written after the experiments and final report were
complete, at the author's request. The first page explicitly identifies it as
a retrospective reconstruction. It is not an original pre-experiment proposal,
a preregistration, or evidence that every later evaluation decision existed at
project inception. The work plan is relative, not a backdated schedule or an
instruction to rerun the completed study. The current state and actual outcomes
remain in [PROJECT_STATUS.md](../../PROJECT_STATUS.md) and the final report.

The initial course idea used NeurIPS formatting. No separate proposal template,
page limit, or instructor approval of this reconstruction has been supplied.
This document uses the official, unchanged NeurIPS 2026 **preprint** style for
visual consistency; it makes no conference submission or acceptance claim.
Its wording and layout can be adapted if the instructor supplies requirements.

## Build without data, models, or experiments

Use a LaTeX installation with the packages listed in `main.tex`, or
[Tectonic](https://github.com/tectonic-typesetting/tectonic). This edition was
compiled using Tectonic 0.17.0. A first build may download TeX packages/fonts;
a populated cache permits offline builds. From the repository root:

```bash
mkdir -p artifacts/proposal-build output/pdf
tectonic --untrusted --outdir artifacts/proposal-build report/proposal/main.tex
cp artifacts/proposal-build/main.pdf output/pdf/turkish-punctuation-proposal.pdf
```

The proposal is self-contained: no generated final-report tables, scientific
figures, predictions, or private inputs are needed. Its local copy of
`neurips_2026.sty` is byte-identical to the final report's official style file;
source, attribution, and SHA-256 are in
[third-party notices](../../THIRD_PARTY_NOTICES.md).

## Private course copy

The public source and PDF omit the student number. A local-only wrapper named
`course-submission.tex` can explicitly opt in with
`\def\CourseSubmission{1}\input{main.tex}` and obtain the author suffix from
`private-author.tex`. The public build never includes the private author file
unless that macro is defined. Both private source files are ignored by Git.

The prepared local course PDF is `output/pdf/proposal-course-submission.pdf`.
It contains the same text and chronology disclosure, plus the supplied student
number. It is ignored by Git and is not part of the public repository. No
document has been submitted to the instructor by this preparation step.

## Design records used

- [Training protocol](../../docs/training-protocol.md): shared initialization,
  three-epoch budget, word-normalized loss, row replacement, clean selection.
- [Noise protocol](../../docs/noise-protocol.md): operators, eligibility,
  deterministic plans, severity budgets, and target-preserving alignment.
- [Case-loss ablation preparation](../../docs/case-loss-ablation-preparation.md):
  same remaining plans and masks, with no redistribution of the omitted dose.
- [Held-out protocol](../../docs/held-out-test-protocol.md): final matrix,
  validation/test roles, paired metrics, cohorts, and interpretation limits.

This document summarizes those records; it does not replace or modify them.
