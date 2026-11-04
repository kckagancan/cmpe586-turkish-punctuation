# Current project state

Updated **2026-11-04**. The original study, final report and GitHub publication
are complete. The instructor-feedback extension has a **frozen supplemental
transfer protocol and 13 prepared, saved-input-reconciled views**, plus separate
execution wiring and a new uploaded notebook. The actual T4/2026.07 base runtime
and pinned restore are verified; validation-only GPU readiness completed and
the supplemental evaluator is running in the new notebook.
Initial three **published-arm** conditions passed independent local saved-file/
raw-target recomputation for both cohorts; at least9/52 conditions were committed
at the last UI observation. User will report completion; do not wait/monitor
through the run or start another writer on this Colab/Drive output.
No training or cleanup job is running.
Read this first, then the relevant linked record. This file authorizes no run.

## Current milestone and next action

[Transfer protocol](docs/transfer-evaluation-protocol.md),
[completed input preparation](docs/transfer-evaluation-preparation.md),
[aggregate audit](docs/transfer-evaluation-input-audit.json),
[coverage CSV](docs/transfer-evaluation-input-coverage.csv).

- Four existing selected models: published, clean, augmented, without-case-loss.
  No new training/checkpoint selection. Clean anchor + elongation, one internal
  omission, partial folding and word-mask-matched full folding at seeds586/587/588:
  **13 views /52 later forwards /104 primary+sensitivity score sets**.
- One corpus anchor per family, not an unsupported mild/medium/heavy grid.
  Rates apply to eligible opportunities:89/4512 elongation,50/4174 omission,
  **43/617 operational partial folding**. The original case-neutral43/621
  analysis remains intact: four surface-casing-incompatible opportunities and
  zero hits are excluded for this implementation. This is distinct from the
  initial min4 pool that also had617 members.
- Anonymous joint morphology templates retain measured elongation tails,
  ambiguous omission-position marginalization and partial-fold subset patterns.
  Conditioning/adaptation/extrapolation and uniform word sampling are explicit;
  this does not simulate the full natural tweet process.
- Data-only validation-prefix probe:512 source rows,510 eligible,13 views,
  maximum93 tokens. Exact protocol/config/profile/code/model-metadata/cohort
  freeze preceded new test-view generation. No model metrics chose this recipe.
- Primary14,600 rows /181,485 targets; sensitivity14,445 /181,116 filters the same
  generated inputs/edits, without reranking. Per seed:3,355 elongations,
  1,826 omissions,1,803 partial folds; full-fold control edits the same1,803 words
  but more characters. Maximum154 tokens, no truncation or condition-specific
  drops. All original targets and first-subword alignment preserved.
- Completed files were reconciled with raw targets, edit semantics, independent
  SHA mask ranking and deterministic donor parameters. Fresh re-tokenization
  matched every saved encoding column and audit. No unit suite was added/run.
  Old equal-length noise audit remains unchanged; insertion/deletion has its
  own auditor. The old clean encoding is reused by explicit read-only hashes;
  old39 noisy views and existing weights were not regenerated/copied/downloaded.
- Freeze `b16cd070f37073b76503267668e27761922708eaa866ac3a80aab3fb790905d1`.
  Inputs `235c0c01cf8b5a34b24e36f1bbdcfe8603c34e401171dd47ec9856c87a30b7a3`.
  Local inputs: `data/processed/transfer-noise-v1/`; snapshots/probe/freeze:
  `artifacts/evaluation-preparation/transfer-noise-v1/`.
  Checks, preservation/publication receipts and complete changed-file inventory:
  `artifacts/evaluation-preparation/transfer-noise-v1/preparation-2026-11-04/`.

**Current action:** [execution wiring](docs/transfer-evaluation-execution.md) and
the [08 Colab notebook](https://colab.research.google.com/drive/1QWib5-KYb7FkSC49KB5NuD2iA5N4ktZm)
are prepared. Source/input bundle205 files/22,133,412 bytes, SHA
`d13f88d06fd97f5a74a891fefa0f33248bb7e9be4f7363063570d57f4905ed80`,
uploaded with the pinned restorer; no weights/predictions included. Existing
Drive models and validation archive are reused. Source checks and local
repository publication checks passed before upload; no unit suite ran.
Receipts/current handoff: `artifacts/evaluation-preparation/transfer-noise-v1/launch-2026-11-04/`.
Actual T4/FP32/SDPA-math/batch16 readiness passed:13 fresh input/target audits,
four checkpoints and24 validation-only GPU probes. Downloaded readiness/complete
bytes independently match identity/runtime/packages/models/probe/source-hash map.
Readiness identity `4c2ffe08e55cf5df2acbfe6f16cbc6feaede680e982f05b0af6b1723e19084f9`.
The explicit52-condition evaluator is running. Run identity
`8cf43b9d6a166baf005fb0320a52177f859c4b6215d06baaa0f93be57156d200`.
Initial published clean-anchor, elongation/seed586 and omission/seed586 were
downloaded: every saved file hash, original IDs/order/targets, both-cohort
metrics and own-anchor drops match independent local recomputation. Fresh
published clean anchor exactly matches the original held-out anchor. No extra
inference ran for that audit. Initial3 verified; observed at least9 committed;
these are handoff snapshots, not a claim that all52 are done.
Colab cell9 stays executing and the tab is left visible. Streaming logs:
`preparation/transfer-noise-v1/launch-2026-11-04/evaluation.log` on Drive;
outputs/progress: `artifacts/evaluations/transfer-noise-v1/` under project Drive.
Source scoring switches stay off in Git. **Next after the user's completion
message:** audit all saved predictions/104 score sets and specified contrasts,
archive/verify results, then update the final report. No cleanup or proposal
work is requested. Report every outcome, per-class P/R/F1/support and paired contrasts;
three-seed SD measures corruption variation only, not training uncertainty.
No new scores have been added to the original report/PDF.

New corruptions reuse already scored test texts: this is **supplemental
controlled transfer**, not independent natural-text evaluation or blind
preregistration. MultiLexNorm source test remains unread. Optional natural-data
annotation and LLM/silver-vs-human requirements remain in the
[feasibility record](docs/informal-transfer-feasibility.md); no natural punctuation
gold has been created.

## Corpus evidence already complete: do not rerun

[Candidate review](docs/informal-corpus-selection.md) selected official
**MultiLexNorm monolingual Turkish train only** over #Turki$hTweets and other
candidates. [Audit](docs/informal-corpus-audit-results.md):570 sentences /6,443
slots,354 boundary-changing slots separate. Old-operator reachability1,109/2,031
(54.6036%) is a word-level upper bound, not sampled training coverage or F1.
Audit `31b1594e88c80d0607436a008a7b1a6e9bd74c67452eb73dcf0424a69a56aa7a`.

[Depth findings](docs/informal-corpus-depth-results.md) preserve min4 shape counts
55/49/42 on3,598 eligible pairs. Separate length sensitivity and broader minima
2/3/2 calibration include short words:89/4,512,50/4,174,43/621; affected tweets
63/40/35, union182 words /131 tweets. Elongation extra-char median2/P75=3/max25;
omission always1; partial fraction P25=median=P75=0.5. Per-tweet quartiles all
zero; do not force three natural severity levels or add unlike denominators.
5,000 sentence-resampling intervals describe conditional corpus stability,
not population prevalence or model performance. Old2/5/10% levels remain
synthetic stress doses.
Depth `cde287765953c12dc78f4f56d0eca6ec81c017d21dd10626f3f9fffc7d161cca`;
length `1bd0abe5b1d6542d4a5b138ae5c1d8c3f8b2fc35cf3b835e832a1f9d7229703c`;
calibration `f8b903a57451c895712c7a7e71bb7061a02dbeeff123cf3efe4b58605f15039f`.
Immutable runs and earlier preservation evidence are under
`artifacts/corpus-audit/multilexnorm-turkish-v1/` and
`artifacts/corpus-audit/multilexnorm-depth-v1/`. These completed analyses were
reused, not rerun to prepare the new views.

## Original study and delivered report

**Robustness of Turkish Punctuation Restoration under Informal Writing Noise**
(CMPE586), **Kagan Can**. [Public final PDF](output/pdf/turkish-punctuation-report.pdf):
10 pages, official NeurIPS2026 preprint style; no conference-acceptance claim.
[Editable source/build](report/README.md), [documentation guide](docs/README.md),
[reproduction](docs/reproduction.md). User-authorized [MIT](LICENSE) applies to
original code/docs, with separate [third-party terms](THIRD_PARTY_NOTICES.md).

- Shared initialization, training seed586,3epochs/10,968updates, identical rows
  and optimizer budget. Clean-validation selection chooses epoch3 for all arms;
  noisy validation selects no checkpoint. No convergence/optimal-budget claim.
- Four labels O/PERIOD/COMMA/QUESTION_MARK, first-subword supervision and-100
  continuation/padding. Original case-loss/full-fold/interior-swap operators
  preserve targets; removal does not redistribute omitted dose.
- Main validation117 conditions; ablation5 new +15 reused references.
  [Grid](docs/robustness-grid-results.md), [ablation](docs/case-loss-ablation-results.md).
- Original held-out122/122 forwards,244 cohort scores completed October23;
  independent raw-target recomputation matches Colab. No extra inference.
  Primary14,600/181,485 after61 empty-word-row quarantines; sensitivity excludes
  155 exact input overlaps. [All results](docs/held-out-test-results.md).
- Clean F1: clean-trained84.3437%, augmented84.4824%; combined-heavy76.7502%
  vs80.5289%, paired gain3.7787±0.2191pp over corruption seeds. Medium/seed586
  retaining case loss adds0.9585pp on case-loss and0.8799pp on combined;
  small folding/swap differences favor removal. One training seed remains.

## Retention, publication and course handoff

Original run `60e31867535de346accb64be45df04984434f6ff328b31c8819115cb689538a0`;
analysis `e39f8803a6e6614363a6db97c1edce870757a2e909dec9dba5e93d0d9edc7f91`.
Raw results: `artifacts/evaluations/held-out-test-v1/`; final ZIP570 files+manifest,
16,011,054bytes, SHA `2463a23ed22dc9543341393bedfb902f693448330da860374c708748d4830925`.
[Completion/archive](docs/held-out-test-completion.md), [backup map](artifacts/README.md).
Full epoch models/shared initialization remain local; selected Drive exports
are retained. Scoped Drive cleanup completed1,148 backed copies/925,753,181bytes,
all protected IDs retained; no further cleanup pending. [Receipt](docs/drive-cleanup.md).

User priority November4: **final report**, no further proposal work. Proposal
remains local-only/outside this task; historical retention/path-discrepancy
evidence is preserved in the earlier preparation receipts. Student final PDF:
`output/pdf/course-submission-report.pdf`, byte-identical to the historical
`output/pdf/course-submission.pdf`. Public final PDF also matches the receipt.
No PDF was rebuilt/renamed here. Student metadata stays in ignored files.
Final course submission/template confirmation remains a user action.

Read-only repository checks validate source/JSON/cleared notebooks, links and
report bindings, not experiments or the historical unit suite. The live GitHub
[workflow](.github/workflows/repository-checks.yml) uses read-only permissions,
pinned actions/Ruff and Python3.11; it requires no workflow permission change.
Publication/proposal/CI historical receipts: `artifacts/publication/`.
Seven historical style findings in frozen scripts remain; do not rewrite them
for cosmetics. Commit only code/config/aggregate reports, never raw data,
weights, predictions or private metadata.

Dataset licensing, exact split provenance/document independence and published
model prior exposure remain unresolved. Validation informed original design;
future test-informed changes are exploratory and require genuinely unseen data
for independent confirmation. Preserve completed recipes/snapshots/results.
