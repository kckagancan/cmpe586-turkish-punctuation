# Current project state

Updated **2026-11-04**. Both the original study and instructor-feedback extension
are complete. The final public/private reports have **15 pages** and include
both stages. The original **10-page** report editions, sources, metrics and
result archive are separately preserved. **No training, inference or cleanup
job is running.** Colab is left visible with a completion summary.

## Current outcome and next action

[Supplemental results](docs/transfer-evaluation-results.md),
[completion/archive](docs/transfer-evaluation-completion.md),
[report editions](docs/report-versions.md), [report build](report/README.md).

- All **52/52 forwards /104 primary+sensitivity score sets** finished on the
  pinned T4. Four existing selected models; no new training/model selection.
  Fresh clean anchors + elongation, one internal omission, partial folding and
  word-mask-matched full folding at seeds586/587/588.
- Local independent saved-prediction/raw-target analysis matches every score,
  support/confusion, own-anchor drop and exposure record. All four fresh clean
  anchors equal the original held-out anchors in both cohorts. Frozen inputs,
  runtime/readiness/source/model bindings and shared padded batch work match.
  No training or completed inference was repeated. No unit suite was added/run.
- Primary14,600 rows /181,485 targets; sensitivity14,445 /181,116 filters the
  same predictions without resampling. Original61 quarantined rows stay outside.
- Augmented−Clean primary macro-F1 gains: elongation **0.2491±0.0843pp**,
  omission **0.2290±0.0635**, partial folding **0.1882±0.0425**. Matched full:
  **0.1279±0.0446**. Means/SD describe three corruption seeds for fixed models;
  one training seed remains. The clean-anchor advantage is already0.1387pp.
  Retaining case loss does not uniformly reduce own-anchor degradation.
  Mixed-sign partial−full contrasts cannot isolate novelty: character dose differs.
- Corpus anchors:89/4512 elongation,50/4174 omission,43/617 operational partial
  folding; original case-neutral43/621 remains intact. Exact edited words per
  seed3,355/1,826/1,803; control uses the same1,803 words. Morphology tails,
  feasibility conditioning, ambiguous omission mapping and extrapolation remain
  explicit. Maximum154 tokens; no truncation or condition-specific target drop.
- The new distortions reuse already scored project test texts: **supplemental
  controlled transfer**, not independent natural-data evaluation or blind
  preregistration. MultiLexNorm source test remains unread. No natural punctuation
  gold is created. Original2/5/10% stress doses remain unchanged.

**Next:** user can retain the final ZIP and use the appropriate report edition
for submission. No further experiment, proposal work or Drive cleanup is requested.
Any later cleanup needs its own scoped backup check; nothing was deleted here.
Course template/submission requirements remain a user-side confirmation.

## Result identities and locations

- Freeze `b16cd070f37073b76503267668e27761922708eaa866ac3a80aab3fb790905d1`.
- Inputs `235c0c01cf8b5a34b24e36f1bbdcfe8603c34e401171dd47ec9856c87a30b7a3`.
- Readiness `4c2ffe08e55cf5df2acbfe6f16cbc6feaede680e982f05b0af6b1723e19084f9`.
- Run `8cf43b9d6a166baf005fb0320a52177f859c4b6215d06baaa0f93be57156d200`.
- Analysis `7847a6b85dcf7d4f1296cff6495974bcbf4103c635b628fce2d8820a45ef06f6`.
- Raw saved run and immutable CPU analysis:
  `artifacts/evaluations/transfer-noise-v1/` and `analysis-v1/`.
- Completion/download/PDF/preservation/check receipts and complete inventory:
  `artifacts/evaluation-preparation/transfer-noise-v1/completion-2026-11-04/`.
- Final local ZIP there: `transfer-noise-v1-results-and-report-v2.zip`,
  **411 files+manifest /27,337,920bytes**, SHA
  `c956e2ee2eb505982df9a0adfb9a09d0263b1f9598be7f54cba1e76798d6d6e1`.
  Includes old/new report/result evidence and private course metadata: keep local.
  Model weights remain separately retained and were not copied/downloaded again.
- Drive run and original304-file raw transfer archive remain in the existing
  project namespace. [Colab](https://colab.research.google.com/drive/1QWib5-KYb7FkSC49KB5NuD2iA5N4ktZm)
  displays completion; no monitor or automatic deletion is scheduled.

## Original study and versions: do not rerun

Matched seed586, shared initialization/rows/optimizer and3epochs/10,968updates.
All selected epoch3 using highest clean-validation punctuation macro-F1; noisy
validation selects no checkpoint. Original validation117 conditions; ablation5
new +15 reused references. Original held-out122 forwards /244 cohort score sets.
[Original results](docs/held-out-test-results.md): Clean84.3437%, Augmented84.4824%;
combined-heavy76.7502% vs80.5289%, paired gain3.7787±0.2191pp. Medium/seed586
case-loss retention gives0.9585pp on casing and0.8799pp on combined; unfavorable
fold/swap contrasts remain. Single training seed and fixed-budget limits remain.

Original result ZIP SHA
`2463a23ed22dc9543341393bedfb902f693448330da860374c708748d4830925` remains separate.
All570 archive members and its source run bytes were checked; original configs,
shared scientific sources, aggregate metrics, original tables/bindings/figures
are unchanged. Exact public/student original PDFs are in `output/pdf/` with
`original-study-v1-` names. The216-file prior source/report/evidence snapshot is
under `artifacts/publication/original-study-v1/`, SHA
`6caa26fd646589d41610fd07202434c1d1e4970e7c70fb825b97ccf666f7b0ec`.
Full epoch models/shared initialization remain in their existing local archives;
selected Drive exports are retained. Earlier scoped cleanup is historical and
complete; no new cleanup is pending. Proposal remains local-only/outside scope.

## Publication and remaining limits

[Current public PDF](output/pdf/turkish-punctuation-report.pdf);
private current PDF `output/pdf/course-submission-report.pdf`. All15 public and
private pages rendered/reviewed; numeric content, private metadata separation,
references and overflow checked. Separate original/supplement table bindings
and repository/Ruff publication checks pass; they are not experiment/unit tests.
Publication receipts/inventory live in the completion directory. Git contains
code/config/aggregate results/cleared notebooks/public PDFs, never raw predictions,
weights, private metadata or local archives. Existing MIT/third-party terms apply.

Dataset licensing, exact split provenance/document independence and published
model exposure remain unresolved. Corpus train-only audit/depth/calibration
runs are already complete and reused, not rerun; see
[corpus findings](docs/informal-corpus-depth-results.md). Do not rewrite frozen
recipes or historical source snapshots to refresh operational status.
