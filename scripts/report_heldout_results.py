"""Publish aggregate held-out reports and fixed diagnostic counts, with no inference."""

import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from turkish_punctuation.heldout import load_plan, load_test, verify_commit
from turkish_punctuation.heldout_runtime import order_and_records, read, variant
from turkish_punctuation.training_io import sha256, utc_now

ROOT = Path(__file__).resolve().parents[1]
COMPLETION = ROOT / 'artifacts/evaluation-preparation/held-out-test-v1/completion-2026-10-23'
RUN = ROOT / 'artifacts/evaluations/held-out-test-v1'
MODELS = ('published', 'clean', 'augmented', 'without_case_loss')
CAPTIONS = ('Published', 'Clean-trained', 'Augmented', 'Without case loss')
FAMILIES = ('case_loss', 'diacritic_loss', 'typo_swap', 'combined')
FAMILY_NAMES = ('Case loss', 'Diacritic loss', 'Typo swap', 'Combined')
COLORS = ('#777777', '#0072B2', '#D55E00', '#009E73')


def fixed_diagnostics(plan, execution, cohorts, analysis):
    prepared = ROOT / execution['prepared_directory']
    clean, _ = load_test(ROOT, ROOT, plan, cohorts)
    # Use the same original targets and shared evaluation order as the frozen analyzer.
    order, originals = order_and_records(prepared, clean)
    sensitivity = set(cohorts['sensitivity_row_ids'])
    rows = []
    for arm in MODELS:
        available = analysis['conditions'][arm]
        anchor = {r['row_id']: r for r in map(json.loads,
                  (RUN / 'models' / arm / 'conditions/clean-anchor/predictions.jsonl').read_text().splitlines())}
        for name in plan['qualitative_analysis']['conditions']:
            if name not in available:
                continue
            _, words = variant(prepared, name, order, originals)
            counts, totals = Counter(), Counter()
            path = RUN / 'models' / arm / 'conditions' / name / 'predictions.jsonl'
            for line in path.read_text().splitlines():
                row = json.loads(line)
                row_id = row['row_id']
                changed = [a != b for a, b in zip(originals[row_id]['tokens'], words[row_id]['tokens'], strict=True)]
                edited_context = any(changed)
                for i, (true, old, new) in enumerate(zip(row['reference_ids'], anchor[row_id]['prediction_ids'],
                                                        row['prediction_ids'], strict=True)):
                    outcome = ('both_correct' if true == old == new else 'new_error' if true == old
                               else 'recovered_error' if true == new else 'changed_error' if old != new
                               else 'unchanged_error')
                    location = ('edited_word' if changed[i] else 'unchanged_word_edited_context'
                                if edited_context else 'unchanged_row')
                    label = originals[row_id]['tags'][i]
                    totals[outcome] += 1
                    for cohort in ('primary', 'sensitivity') if row_id in sensitivity else ('primary',):
                        counts[cohort, outcome, location, label] += 1
            if dict(totals) != available[name]['paired_outcomes_primary']['outcomes']:
                raise ValueError(f'Diagnostic partition differs from frozen outcome counts: {arm}/{name}')
            for cohort in ('primary', 'sensitivity'):
                scored = available[name]['cohorts'][cohort]
                if sum(v for (c, _, _, _), v in counts.items() if c == cohort) != scored['evaluated_words']:
                    raise ValueError(f'Diagnostic cohort word coverage differs: {arm}/{name}/{cohort}')
                for label, metrics in scored['per_class'].items():
                    if sum(v for (c, _, _, tag), v in counts.items()
                           if c == cohort and tag == label) != metrics['support']:
                        raise ValueError(f'Diagnostic class support differs: {arm}/{name}/{cohort}/{label}')
            for (cohort, outcome, location, label), count in sorted(counts.items()):
                rows.append({'cohort': cohort, 'model': arm, 'condition': name, 'outcome': outcome,
                             'input_location': location, 'reference_class': label, 'word_count': count})
    return rows


def main():
    verification = read(COMPLETION / 'local-verification.json')
    if (verification['status'] != 'verified_complete'
        or verification['cohort_scores_recomputed_locally'] != 244
        or sha256(Path(verification['archive'])) != verification['archive_sha256']):
        raise ValueError('A complete local raw-target/archive verification is required.')
    marker = verify_commit(RUN / 'analysis-v1')
    analysis = read(RUN / 'analysis-v1/analysis.json')
    if analysis['identity'] != verification['analysis_identity'] or marker['identity'] != analysis['identity']:
        raise ValueError('Verified analysis differs.')
    plan, execution, freeze, cohorts = load_plan(ROOT)
    if freeze['identity'] != read(RUN / 'provenance/run.json')['specification']['runtime_binding']['freeze_identity']:
        raise ValueError('Report protocol differs from completed inference.')
    docs = ROOT / 'docs'
    figures = docs / 'figures'
    figures.mkdir(exist_ok=True)
    index = {(r['cohort'], r['arm_or_contrast'], r['condition_group'], r['metric']): r
             for r in analysis['aggregates']}

    def metric(arm, group, cohort='primary', key='punctuation_macro.f1'):
        return index.get((cohort, arm, group, key))

    def formatted(row):
        if row is None:
            return 'Outside frozen scope'
        factor = 100 if row['unit'] == 'fraction' else 1
        return f"{factor * row['mean']:.4f}" + (
            f" ± {factor * row['sample_sd']:.4f}" if row['sample_sd'] is not None else '')

    groups = ['clean-anchor', *[f'{family}-{level}' for family in FAMILIES
                               for level in plan['noise']['levels']], *plan['noise']['full_stress_conditions']]
    lines = ['# Held-out test: verified results', '',
        'Completed **2026-10-23**. All **122 forward conditions / 244 cohort score sets**',
        'were recomputed from original word targets on Colab and independently on the local CPU.',
        'No training or model inference was repeated during analysis.', '',
        '## Main comparison', '',
        f"On clean primary test text, clean-trained macro-F1 is **{formatted(metric('clean', 'clean-anchor'))}%**",
        f"and augmented macro-F1 is **{formatted(metric('augmented', 'clean-anchor'))}%**",
        f"(augmented − clean **{formatted(metric('augmented_minus_clean', 'clean-anchor'))} pp**).",
        "Under combined-heavy noise, the augmented − clean paired difference is",
        f"**{formatted(metric('augmented_minus_clean', 'combined-heavy'))} pp** across three corruption seeds.",
        '', 'These are descriptive results for fixed models trained with one training seed.',
        'Means and sample SD (ddof=1) describe corruption realizations only;',
        'clean/full-loss controls and the seed-586 ablation have no estimated seed SD.', '',
        '![Held-out robustness curves](figures/held-out-test-robustness.png)', '',
        '## Complete frozen matrix', '',
        'Punctuation macro-F1 (%) averages PERIOD, COMMA and QUESTION_MARK.',
        'O positions remain in false-positive/negative counts. All planned conditions,',
        'including unfavorable results and full-loss stresses, are reported.',
        'Stochastic main-arm columns are three-seed means ± SD; the ablation column',
        'contains seed586 alone. Use the matched-seed ablation contrasts below,',
        'rather than subtracting that single value from a three-seed mean.', '']
    for cohort, caption in (('primary', 'Primary: 14,600 rows / 181,485 words'),
                            ('sensitivity', 'Sensitivity: 14,445 rows / 181,116 words')):
        lines += [f'### {caption}', '',
                  '| Condition | Published | Clean-trained | Augmented | Without case loss | Augmented − clean (pp) |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for group in groups:
            cells = [formatted(metric(arm, group, cohort)) for arm in MODELS]
            lines.append('| ' + ' | '.join([group, *cells,
                formatted(metric('augmented_minus_clean', group, cohort))]) + ' |')
        lines += ['']
    lines += ['## Case-loss training ablation', '',
        'These five comparisons use **corruption seed 586 only**, matching the frozen validation scope.',
        'Full augmentation minus the without-case-loss arm is shown below in F1 percentage points.',
        'Removing case loss also lowers total training noise; its budget was not redistributed.',
        'This is not an equal-dose ranking of noise components.', '',
        '| Condition | Primary augmented − without | Sensitivity augmented − without |',
        '| --- | ---: | ---: |']
    for group in ('clean-anchor', *[f'{family}-medium' for family in FAMILIES]):
        lines.append('| ' + ' | '.join([group, *[formatted(metric('augmented_minus_without_case_loss', group, c))
                                                for c in ('primary', 'sensitivity')]]) + ' |')
    lines += ['',
        f"Retaining case loss improves primary F1 by **{formatted(metric('augmented_minus_without_case_loss', 'case_loss-medium'))} pp**",
        f"under case loss and **{formatted(metric('augmented_minus_without_case_loss', 'combined-medium'))} pp** under combined noise.",
        'The small diacritic/typo differences favor the ablation. The clean-validation',
        'advantage of removing case loss did not recur on held-out clean text;',
        'the full augmented model has a small positive clean-test difference here.', '',
        '![Case-loss ablation](figures/held-out-test-ablation.png)', '',
        '## Relative drops, classes and fixed diagnostics', '',
        'The [complete aggregate CSV](held-out-test-aggregates.csv) retains macro/micro and per-class',
        'precision, recall and F1; own-clean-to-noisy drops; and paired model contrasts.',
        'Positive own-clean drop means deterioration. Supports, confusion matrices and every',
        'condition’s realized noise coverage are preserved in the [aggregate condition JSON](held-out-test-condition-summary.json).',
        'The [fixed diagnostic CSV](held-out-test-diagnostics.csv) partitions outcomes into edited',
        'words, unchanged words with edited sentence context, and entirely unchanged rows,',
        'for both cohorts and the prelisted diagnostic conditions. Counts are checked against',
        'the frozen analyzer’s primary outcome totals. Source-ordered raw examples stay outside Git.', '',
        'The sensitivity cohort removes 155 exact original-input matches to train or validation',
        '(148 train, 90 validation, 83 both). It filters the **same** saved primary predictions;',
        'noise was not regenerated or rebudgeted. It is not a new independent test set or proof',
        'of document separation. Full-loss stresses are separate endpoints, not another 10% dose.', '',
        '## Method, interpretation and limits', '',
        'All fine-tuned arms use the same initialization, training rows/order and 3-epoch /',
        '10,968-update budget. Epoch 3 was selected by clean validation; noisy validation and',
        'test scores selected no checkpoints. Published BERT is an external reference with',
        'uncontrolled training history. Actual T4 model tensors, source/input hashes, shared',
        'batch membership and FP32 SDPA/math numerical settings passed the recorded gates.', '',
        'This protocol was frozen before project test scoring, after validation development;',
        'it is validation-informed, not a blind preregistration. No p-values, equivalence or',
        'training-seed stability claim is supported. Synthetic spelling edits do not cover all',
        'natural informal Turkish or establish semantic invariance. Dataset licensing, exact',
        'split provenance, document independence and published-model prior test exposure',
        'remain unresolved. Future test-informed changes must be labeled exploratory.', '',
        '## Evidence and retention', '',
        f"- Run identity: `{verification['run_identity']}`.",
        f"- Analysis identity: `{verification['analysis_identity']}`.",
        '- Original predictions/condition commits: `artifacts/evaluations/held-out-test-v1/`.',
        '- Local CPU analysis, ZIP and verification: `artifacts/evaluation-preparation/held-out-test-v1/completion-2026-10-23/`.',
        f"- Results ZIP: **{verification['source_files']} source files + manifest**, **{verification['archive_bytes']:,} bytes**.",
        f"- ZIP SHA-256: `{verification['archive_sha256']}`.",
        '- Existing training/model archives remain separate. No Drive file was deleted.',
        '- [Execution and archive record](held-out-test-completion.md); [frozen protocol](held-out-test-protocol.md).', '']

    fig, axes = plt.subplots(2, 4, figsize=(13.6, 7.4), sharey=True)
    for row_i, cohort in enumerate(('primary', 'sensitivity')):
        for col_i, (family, caption) in enumerate(zip(FAMILIES, FAMILY_NAMES, strict=True)):
            axis = axes[row_i, col_i]
            for arm, label, color in zip(MODELS[:3], CAPTIONS[:3], COLORS[:3], strict=True):
                data = [metric(arm, 'clean-anchor', cohort), *[metric(arm, f'{family}-{level}', cohort)
                                                             for level in plan['noise']['levels']]]
                axis.errorbar([0, 2, 5, 10], [100 * r['mean'] for r in data],
                              yerr=[100 * (r['sample_sd'] or 0) for r in data],
                              marker='o', color=color, label=label, capsize=3, linewidth=1.5)
            axis.set_title(caption)
            axis.set_xticks([0, 2, 5, 10])
            axis.grid(alpha=.2)
            if row_i == 1:
                axis.set_xlabel('Component word budget (%)')
            if col_i == 0:
                axis.set_ylabel(cohort.capitalize() + '\nPunctuation macro-F1 (%)')
    fig.suptitle('Held-out robustness · mean ± sample SD over 3 corruption seeds · 1 training seed', fontsize=12)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=3, frameon=False)
    fig.text(.5, .066, 'Combined uses each component budget; its realized union coverage is larger.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.07, right=.985, top=.90, bottom=.15, wspace=.13, hspace=.25)
    for extension in ('png', 'pdf'):
        fig.savefig(figures / f'held-out-test-robustness.{extension}', dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    ablation_groups = ['clean-anchor', *[f'{family}-medium' for family in FAMILIES]]
    for axis, cohort in zip(axes, ('primary', 'sensitivity'), strict=True):
        values = [metric('augmented_minus_without_case_loss', group, cohort)['mean'] for group in ablation_groups]
        axis.barh(range(5), values, color=['#D55E00' if v >= 0 else '#0072B2' for v in values])
        axis.set_yticks(range(5), ['Clean', *FAMILY_NAMES])
        axis.axvline(0, color='#555555', linewidth=.8)
        axis.set_title(cohort.capitalize())
        axis.set_xlabel('Augmented − without case loss (F1 pp)')
        axis.grid(axis='x', alpha=.2)
        padding = max(.05, .16 * max(abs(v) for v in values))
        axis.set_xlim(min(0, min(values)) - 2 * padding, max(0, max(values)) + 3 * padding)
        for i, value in enumerate(values):
            axis.text(value + (padding / 4 if value >= 0 else -padding / 4), i, f'{value:+.4f}',
                      va='center', ha='left' if value >= 0 else 'right', fontsize=9)
    axes[0].invert_yaxis()
    fig.suptitle('Case-loss ablation · medium noise / seed 586 · one training seed', fontsize=12)
    fig.text(.5, .035, 'The omitted budget was not redistributed; the ablation has lower total training noise.',
             ha='center', fontsize=9)
    fig.subplots_adjust(left=.12, right=.98, top=.85, bottom=.19, wspace=.14)
    for extension in ('png', 'pdf'):
        fig.savefig(figures / f'held-out-test-ablation.{extension}', dpi=220)
    plt.close(fig)

    diagnostics = fixed_diagnostics(plan, execution, cohorts, analysis)
    with (docs / 'held-out-test-diagnostics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)
    summary = {'run_identity': verification['run_identity'], 'analysis_identity': analysis['identity'],
               'conditions': analysis['conditions'], 'limitations': analysis['limitations'],
               'raw_text_included': False, 'model_inference_performed': False}
    (docs / 'held-out-test-condition-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    shutil.copyfile(RUN / 'analysis-v1/aggregates.csv', docs / 'held-out-test-aggregates.csv')
    (docs / 'held-out-test-results.md').write_text('\n'.join(lines))
    outputs = [docs / name for name in ('held-out-test-results.md', 'held-out-test-aggregates.csv',
               'held-out-test-condition-summary.json', 'held-out-test-diagnostics.csv')]
    outputs += [figures / f'held-out-test-{name}.{extension}' for name in ('robustness', 'ablation')
                for extension in ('png', 'pdf')]
    receipt = {'generated_at_utc': utc_now(), 'report_source_sha256': sha256(Path(__file__)),
               'verified_analysis_sha256': sha256(RUN / 'analysis-v1/analysis.json'),
               'files': {str(p.relative_to(ROOT)): sha256(p) for p in outputs},
               'fixed_diagnostic_count_rows': len(diagnostics), 'model_inference_performed': False}
    (COMPLETION / 'report-generation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('PUBLISHED: complete aggregate tables, fixed diagnostics and PNG/PDF figures; no inference.')


if __name__ == '__main__':
    main()
