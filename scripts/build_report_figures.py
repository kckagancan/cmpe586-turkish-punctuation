"""Draw legible paper figures from public CSV aggregates; no inference."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report/figures'


def main():
    rows = list(csv.DictReader((ROOT / 'docs/held-out-test-aggregates.csv').open()))
    index = {(r['cohort'], r['arm_or_contrast'], r['condition_group'], r['metric']): r
             for r in rows}

    def score(arm, group, cohort='primary'):
        row = index[cohort, arm, group, 'punctuation_macro.f1']
        factor = 100 if row['unit'] == 'fraction' else 1
        return factor * float(row['mean']), factor * float(row['sample_sd'] or 0)

    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
                         'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.7), sharex=True, sharey=True)
    for ax, family, label in zip(axes.flat,
                                 ('case_loss', 'diacritic_loss', 'typo_swap', 'combined'),
                                 ('Case loss', 'Turkish character folding', 'Interior swap', 'Combined'),
                                 strict=True):
        for arm, name, color, marker in [('published', 'Published', '#737373', '^'),
                                          ('clean', 'Clean-trained', '#0072B2', 'o'),
                                          ('augmented', 'Augmented', '#D55E00', 's')]:
            values = [score(arm, 'clean-anchor'), *[score(arm, f'{family}-{level}')
                                                   for level in ('mild', 'medium', 'heavy')]]
            ax.errorbar([0, 2, 5, 10], [x[0] for x in values], yerr=[x[1] for x in values],
                        color=color, marker=marker, markersize=3.5, linewidth=1.3,
                        capsize=2, label=name)
        ax.set(title=label, xticks=[0, 2, 5, 10], ylim=(75.5, 85.5))
        ax.grid(axis='y', alpha=.22)
        ax.spines[['top', 'right']].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel('Punctuation macro-F1 (%)')
    for ax in axes[1]:
        ax.set_xlabel('Component word budget (%)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.52, 1.01))
    fig.tight_layout(rect=(0, 0, 1, .96), h_pad=1.2)
    for extension in ('pdf', 'png'):
        fig.savefig(OUT / f'robustness.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)

    groups = ['clean-anchor', 'case_loss-medium', 'diacritic_loss-medium',
              'typo_swap-medium', 'combined-medium']
    labels = ['Clean', 'Case loss', 'Character folding', 'Interior swap', 'Combined']
    values = [score('augmented_minus_without_case_loss', group)[0] for group in groups]
    fig, ax = plt.subplots(figsize=(5.6, 2.65))
    y = np.arange(len(groups))
    ax.barh(y, values, color=['#0072B2' if x >= 0 else '#CC79A7' for x in values], height=.55)
    for i, value in enumerate(values):
        ax.text(value + (.022 if value >= 0 else -.022), i, f'{value:+.4f}',
                ha='left' if value >= 0 else 'right', va='center', fontsize=8)
    ax.axvline(0, color='#555555', linewidth=.8)
    ax.set(yticks=y, yticklabels=labels, xlim=(-.32, 1.15),
           xlabel='Augmented − without case loss (F1 percentage points)')
    ax.invert_yaxis()
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for extension in ('pdf', 'png'):
        fig.savefig(OUT / f'ablation.{extension}', dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('Built primary-cohort paper figures from public aggregate CSV; no model/data access.')


if __name__ == '__main__':
    main()
