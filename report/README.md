# Project report

**Author:** Kagan Can · **Course:** CMPE 586.

[Read the public PDF](../output/pdf/turkish-punctuation-report.pdf).
The editable paper is [`main.tex`](main.tex). It uses the official, unchanged
NeurIPS 2026 style with the **non-anonymous `preprint` option**. This is a course
report, with no claim of conference submission, acceptance, or peer review.
The initial course idea used NeurIPS formatting; the final course requirements
have not been supplied. The source can be adapted when those are known.

## What is included

- Abstract, research questions and related work with primary-source references.
- Data eligibility/quarantine, exact overlap sensitivity and licensing limits.
- Noise operators/budgets, matched training, ablation and checkpoint selection.
- Complete main held-out matrix, figures, per-class and contextual diagnostics.
- Unfavorable ablation results, statistical limits and responsible-use discussion.
- Complete sensitivity matrix, realized coverage and reproducibility appendix.

## Rebuild tables without data or models

From the repository root, Python 3.11+ with the standard library is sufficient:

```bash
python scripts/build_report_tables.py
python scripts/build_report_tables.py --check
```

This uses only public aggregate CSV/JSON files. It writes `tables.tex` and
`results-binding.json`, which binds the inputs to their SHA-256 values and the
completed run/analysis identities. The ablation table uses matched seed586
values; it does not subtract an ablation score from a three-seed mean.
No metric is obtained by rerunning training or inference.

## Compile the public PDF

Use a LaTeX installation with the packages in `main.tex`, or
[Tectonic](https://github.com/tectonic-typesetting/tectonic).
This edition was compiled using **Tectonic 0.17.0**. Its first build may download
TeX packages/fonts; later builds can reuse the cache. From the repository root:

```bash
mkdir -p output/pdf
tectonic --untrusted --outdir output/pdf report/main.tex
mv output/pdf/main.pdf output/pdf/turkish-punctuation-report.pdf
```

The dedicated `report/figures/*.pdf` files are vector figures from public
aggregate results, with labels sized for the paper. Redraw them with
`python scripts/build_report_figures.py` in the project's plotting environment.
The original full-cohort figures remain unchanged under `docs/figures/`.
`neurips_2026.sty` comes from the
[official template](https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip)
linked by the [NeurIPS 2026 call](https://neurips.cc/Conferences/2026/CallForPapers).
Its source/hash and attribution are in [third-party notices](../THIRD_PARTY_NOTICES.md).

## Private course copy

The public paper omits the student number. A local submission wrapper may define
`\CourseSubmission` before inputting `main.tex` and supply
`report/private-author.tex`. The public build ignores that private file unless
the wrapper explicitly opts in. The private author file, private wrapper and
course-submission PDF are Git-ignored. The student number must not be added to
public sources or commit messages.

To adapt the layout later, retain the text, tables and figures, replacing the
document class/style and author block. The original experimental configs and
source snapshots do not need to change.
