"""Publish aggregate ablation tables and a figure from locally verified saved scores."""

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from turkish_punctuation.training_io import sha256, utc_now

ROOT = Path(__file__).resolve().parents[1]
COMPLETION = ROOT / 'artifacts/evaluation-preparation/ablation-case-loss-medium-gpu-v1/completion-2026-10-22'


def main():
    verification = json.loads((COMPLETION / 'local-verification.json').read_text())
    if (verification['status'] != 'verified_complete'
        or verification['raw_target_metrics_recomputed'] != 20
        or sha256(ROOT / verification['archive']) != verification['archive_sha256']):
        raise ValueError('A verified local archive/raw-target audit is required.')
    analysis = ROOT / verification['local_run'] / 'analysis'
    for name, expected in json.loads((analysis / 'complete.json').read_text())['files'].items():
        if sha256(analysis / name) != expected:
            raise ValueError(f'Committed analysis changed: {name}')
    comparison = json.loads((analysis / 'comparison.json').read_text())
    if comparison['summary_sha256'] != verification['summary_sha256']:
        raise ValueError('Report inputs differ from the local verification.')
    scores = {(r['model'], r['condition']): r for r in comparison['results']}
    conditions = ['clean-anchor', 'case_loss-medium-seed586', 'diacritic_loss-medium-seed586',
                  'typo_swap-medium-seed586', 'combined-medium-seed586']
    labels = ['Clean', 'Case loss', 'Diacritic loss', 'Typo swap', 'Combined']
    models = ['published', 'clean', 'augmented', 'without_case_loss']
    display = ['Published', 'Clean-trained', 'Augmented', 'Without case loss']
    def delta(name, key='macro_f1_percent'):
        return scores['without_case_loss', name][key] - scores['augmented', name][key]

    table = ['| Condition | Published | Clean-trained | Augmented | Without case loss | Without − augmented (pp) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    classes = ['| Condition | Period ΔF1 | Comma ΔF1 | Question ΔF1 |',
               '| --- | ---: | ---: | ---: |']
    drops = ['| Noise | Augmented clean-to-noisy drop | Without-case-loss drop | Extra drop without case loss |',
             '| --- | ---: | ---: | ---: |']
    for name, label in zip(conditions, labels, strict=True):
        table.append(f'| {label} | ' + ' | '.join(f"{scores[model, name]['macro_f1_percent']:.4f}" for model in models)
                     + f' | {delta(name):+.4f} |')
        classes.append(f'| {label} | ' + ' | '.join(
            f"{delta(name, tag + '_f1_percent'):+.4f}" for tag in ('PERIOD', 'COMMA', 'QUESTION_MARK')) + ' |')
        if name != 'clean-anchor':
            full = scores['augmented', name]['drop_from_own_clean_pp']
            omitted = scores['without_case_loss', name]['drop_from_own_clean_pp']
            drops.append(f'| {label} | {full:.4f} | {omitted:.4f} | {omitted - full:+.4f} |')

    figure, (absolute, effect) = plt.subplots(1, 2, figsize=(11.8, 5.2), sharey=True,
                                            gridspec_kw={'width_ratios': [1.5, 1]})
    colors = ['#777777', '#0072B2', '#D55E00', '#009E73']
    for offset, model, caption, color in zip((-.21, -.07, .07, .21), models, display, colors, strict=True):
        absolute.scatter([scores[model, name]['macro_f1_percent'] for name in conditions],
                         [i + offset for i in range(5)], label=caption, color=color, s=38)
    absolute.set_yticks(range(5), labels)
    absolute.invert_yaxis()
    absolute.set_xlabel('Punctuation macro-F1 (%)')
    absolute.set_title('Matched four-model scores')
    absolute.set_xlim(79.5, 84.7)
    absolute.grid(axis='x', alpha=.2)
    benefits = [-delta(name) for name in conditions]
    effect.barh(range(5), benefits, height=.55,
                color=['#D55E00' if value >= 0 else '#0072B2' for value in benefits])
    effect.axvline(0, color='#555555', linewidth=.8)
    effect.set_xlim(-.55, 1.12)
    effect.set_xlabel('Augmented − without case loss (pp)')
    effect.set_title('Effect of retaining case-loss augmentation')
    effect.grid(axis='x', alpha=.2)
    for i, value in enumerate(benefits):
        effect.text(value + (.025 if value >= 0 else -.025), i, f'{value:+.4f}',
                    va='center', ha='left' if value >= 0 else 'right', fontsize=9)
    figure.suptitle('Case-loss training ablation · medium noise / corruption seed 586', fontsize=13)
    figure.text(.5, .91, 'Validation · one training seed · lower total noise exposure in the ablation',
                ha='center', fontsize=9)
    figure.legend(*absolute.get_legend_handles_labels(), loc='lower center', ncol=4, frameon=False)
    figure.subplots_adjust(left=.12, right=.98, top=.82, bottom=.19, wspace=.17)
    figures = ROOT / 'docs/figures'
    figures.mkdir(exist_ok=True)
    for extension in ('png', 'pdf'):
        figure.savefig(figures / f'case-loss-ablation-comparison.{extension}', dpi=240)
    plt.close(figure)
    shutil.copyfile(analysis / 'comparison.csv', ROOT / 'docs/case-loss-ablation-comparison.csv')

    combined = 'combined-medium-seed586'
    case = 'case_loss-medium-seed586'
    clean_gain = scores['without_case_loss', combined]['macro_f1_percent'] - scores['clean', combined]['macro_f1_percent']
    comma_full = scores['augmented', combined]
    comma_without = scores['without_case_loss', combined]
    report = f'''# Case-loss training ablation: verified validation results

Completed **2026-10-22**, with five new GPU evaluations and 15 matching saved
reference results. Local verification independently recomputed all 20 metrics
against **14,453 validation records / 179,164 word targets per condition**.

## Main finding

Removing case loss from augmented training lowers macro-F1 by
**{-delta(case):.4f} pp under case loss** and **{-delta(combined):.4f} pp under
combined noise**, while clean F1 rises **{delta('clean-anchor'):.4f} pp**.
The observed effect is specific to this recipe and one training seed; the
removed word budget was not redistributed, so total training noise is lower.

## Matched four-model comparison

All noisy rows are **medium / corruption seed 586**, not the three-seed means
from the earlier [full grid](robustness-grid-results.md). F1 is a percentage;
differences are percentage points. The primary metric averages PERIOD, COMMA
and QUESTION_MARK, excluding O from that average while retaining O in counts.

{chr(10).join(table)}

![Matched scores and full-augmentation advantage](figures/case-loss-ablation-comparison.png)

The ablation is slightly better on clean text, diacritic loss and typo swaps in
this realization. Full augmentation performs better on case loss and combined
noise. The ablation still exceeds clean training by **{clean_gain:.4f} pp** on
combined noise; removing the casing component does not erase every observed
augmentation gain. These differences are not statistical significance or
equivalence claims.

## Robustness relative to each model's own clean anchor

A positive drop is worse. The ablation starts from a slightly higher clean score;
reporting its own clean-to-noisy drop preserves that distinction.

{chr(10).join(drops)}

## Punctuation classes

Values below are **without-case-loss minus full-augmentation per-class F1 (pp)**.
All original precision/recall/F1, supports and confusion counts remain in the
ignored saved reports; the accompanying aggregate CSV retains class P/R/F1.

{chr(10).join(classes)}

For combined noise, comma recall is **{comma_full['COMMA_recall_percent']:.4f}%**
with full augmentation and **{comma_without['COMMA_recall_percent']:.4f}%** without
case loss; comma precision is **{comma_full['COMMA_precision_percent']:.4f}%** and
**{comma_without['COMMA_precision_percent']:.4f}%**, respectively. Report both
when interpreting the comma F1 difference.

## Fixed method and limitations

The three fine-tuned models use the same initial tensors, training rows, row
order, update budget and numerical recipe. The ablation preserves row masks
and retained typo/diacritic word operations, omitting case loss only. Each model
selected **epoch 3 / 10,968 updates by clean validation**. No noisy ablation
score selected a checkpoint or changed a hyperparameter. The published model is
an external reference with a different training history.

Inference uses the same Tesla T4, FP32 SDPA/math, deterministic settings,
batch 16, clean-length order and unchanged shared numerical/data source hashes
as the completed reference grid. A fresh GPU clean control is used for each
model. [Evaluation gates and execution](case-loss-ablation-evaluation.md).

There is **one training seed (586)** and **one corruption seed (586)** in this
ablation comparison. No error bars or training-seed stability claims are made.
Removing a component also lowers total noise exposure; this is not an equal-dose
importance ranking or an isolated measurement of a linguistic cue. Synthetic
noise does not represent all informal Turkish. Validation also selected the
checkpoints; **held-out test remains unrun**. Dataset licensing and exact
source/split provenance remain unresolved.

## Evidence and local archive

The original GPU summaries, source/config snapshots, condition commit markers,
predictions, error examples, comparison files and readiness records were copied
byte for byte through the Drive connector. All source/config/model/input and
batch identities passed; comparison JSON/CSV values matched audited metrics.
No training or inference was repeated and no Drive file was deleted.

The results-only ZIP has **{verification['archive_source_files']} source files plus
manifest**, **{verification['archive_bytes']:,} bytes**, SHA-256
`{verification['archive_sha256']}`. It contains no weights or optimizer state;
the previously verified model archive remains separate.

- New local run: `{verification['local_run']}/`.
- Archive/receipts: `artifacts/evaluation-preparation/ablation-case-loss-medium-gpu-v1/completion-2026-10-22/`.
- Aggregate table: [comparison CSV](case-loss-ablation-comparison.csv).
- Exportable figure: [PDF](figures/case-loss-ablation-comparison.pdf).

Next: freeze a separate held-out evaluation protocol/matrix before scoring test
data, including the existing split-overlap caveats. This completion adds no new
training or test run.
'''
    report_path = ROOT / 'docs/case-loss-ablation-results.md'
    report_path.write_text(report)
    outputs = [report_path, ROOT / 'docs/case-loss-ablation-comparison.csv',
               figures / 'case-loss-ablation-comparison.png', figures / 'case-loss-ablation-comparison.pdf']
    receipt = {'generated_at_utc': utc_now(), 'report_source_sha256': sha256(Path(__file__)),
               'verified_analysis_sha256': sha256(analysis / 'comparison.json'),
               'files': {str(p.relative_to(ROOT)): sha256(p) for p in outputs},
               'inference_performed': False, 'test_evaluated': False}
    (COMPLETION / 'report-generation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('Published aggregate report, exact CSV copy and PNG/PDF figures from verified results.')


if __name__ == '__main__':
    main()
