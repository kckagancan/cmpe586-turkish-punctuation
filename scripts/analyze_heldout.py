"""Recompute all held-out scores and fixed aggregates from saved predictions only."""

import argparse
import csv
import shutil
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory

from turkish_punctuation.gpu_evaluation import condition_committed
from turkish_punctuation.heldout import execution_sources, verify_commit, verify_inputs
from turkish_punctuation.heldout_runtime import (
    condition_report,
    diagnostics,
    order_and_records,
    prepared_binding,
    read,
    score_cohorts,
    variant,
    write,
)
from turkish_punctuation.training_io import (
    directory_lock,
    file_hashes,
    sha256,
    utc_now,
    value_digest,
)

ROOT = Path(__file__).resolve().parents[1]


def flatten(metrics):
    result = {f'{average}.{key}': metrics[average][key]
              for average in ('punctuation_macro', 'punctuation_micro')
              for key in ('precision', 'recall', 'f1')}
    result.update({f'per_class.{label}.{key}': values[key]
                   for label, values in metrics['per_class'].items()
                   for key in ('precision', 'recall', 'f1')})
    return result


def summarize(values):
    return {'mean': statistics.mean(values),
            'sample_sd': statistics.stdev(values) if len(values) == 3 else None,
            'n_corruption_seeds': len(values) if len(values) == 3 else None,
            'values': values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan, execution, freeze, cohorts, manifest, clean = verify_inputs(ROOT, args.input_root)
    run_marker = verify_commit(args.run_directory / 'provenance')
    run = read(args.run_directory / 'provenance/run.json')
    identity, specification = run['identity'], run['specification']
    binding = specification['runtime_binding']
    if (run_marker['identity'] != identity or value_digest(specification) != identity
        or specification['readiness_identity'] != value_digest(binding)
        or binding['freeze_identity'] != freeze['identity'] or binding['models'] != plan['models']
        or binding['inference'] != plan['inference']
        or binding['execution_config_sha256'] != sha256(ROOT / 'configs/held-out-execution.json')
        or binding['inputs'] != prepared_binding(args.input_root, execution)
        or binding['source_sha256'] != execution_sources(ROOT, freeze)):
        raise ValueError('Inference provenance differs from the frozen/current execution version.')
    prepared = args.input_root / execution['prepared_directory']
    order, originals = order_and_records(prepared, clean)
    batching = read(args.run_directory / 'provenance/batching.json')
    if (value_digest(batching) != specification['batching_sha256']
        or batching['source_row_ids'] != list(originals) or batching['batch_size'] != plan['inference']['batch_size']):
        raise ValueError('Saved shared batch order changed.')
    for name, digest in binding['source_sha256'].items():
        if sha256(args.run_directory / 'provenance/source_snapshot' / name) != digest:
            raise ValueError(f'Inference source snapshot changed: {name}')
    reports, commit_hashes, total = {}, {}, 0
    for arm in execution['model_order']:
        names = (plan['run_matrix']['ablation_conditions'] if arm == plan['run_matrix']['ablation_model']
                 else plan['run_matrix']['main_conditions'])
        reports[arm] = {}
        baseline, clean_metrics = None, None
        for name in names:
            directory = args.run_directory / 'models' / arm / 'conditions' / name
            if not condition_committed(directory, identity):
                raise ValueError(f'No verified complete output: {arm}/{name}')
            current, metrics = score_cohorts(directory / 'predictions.jsonl', originals, cohorts)
            if name == 'clean-anchor':
                baseline, clean_metrics = current, metrics
            dataset, noisy = variant(prepared, name, order, originals)
            lengths = [len(row) for row in dataset['input_ids']]
            batch = plan['inference']['batch_size']
            padded = sum(max(lengths[i:i + batch]) * len(lengths[i:i + batch])
                         for i in range(0, len(lengths), batch))
            if padded != batching['padded_tokens'][name]:
                raise ValueError('Saved condition padded-token work differs.')
            paired, examples = diagnostics(plan, name, originals, noisy, baseline, current,
                                            cohorts['sensitivity_row_ids'])
            expected = condition_report(identity, arm, name, metrics, clean_metrics,
                                        manifest['conditions'][name], paired)
            saved = read(directory / 'metrics.json')
            if (any(saved[key] != value for key, value in expected.items())
                or saved['predictions_sha256'] != sha256(directory / 'predictions.jsonl')
                or read(directory / 'error_examples.json') != {'selection': plan['qualitative_analysis'],
                                                             'examples': examples}):
                raise ValueError(f'Saved raw-target/paired/qualitative audit differs: {arm}/{name}')
            reports[arm][name] = expected
            commit_hashes[f'{arm}/{name}'] = sha256(directory / 'complete.json')
            total += 1
            print(f'RAW_TARGETS_VERIFIED {total}/122 {arm} {name}: both cohorts', flush=True)
    if total != 122:
        raise ValueError('Frozen matrix coverage differs.')
    groups = {'clean-anchor': ['clean-anchor']}
    for family in plan['noise']['families']:
        for level in plan['noise']['levels']:
            groups[f'{family}-{level}'] = [f'{family}-{level}-seed{seed}' for seed in plan['noise']['seeds']]
    for name in plan['noise']['full_stress_conditions']:
        groups[name] = [name]
    aggregate, csv_rows = [], []
    for cohort in ('primary', 'sensitivity'):
        for arm, conditions in reports.items():
            for group, names in groups.items():
                available = [name for name in names if name in conditions]
                if not available:
                    continue
                for metric in flatten(conditions[available[0]]['cohorts'][cohort]):
                    values = [flatten(conditions[name]['cohorts'][cohort])[metric] for name in available]
                    row = {'cohort': cohort, 'arm_or_contrast': arm, 'condition_group': group,
                           'observed_conditions': ','.join(available),
                           'metric': metric, 'unit': 'fraction', **summarize(values)}
                    aggregate.append(row)
                    csv_rows.append({key: value for key, value in row.items() if key != 'values'})
                    drops = [flatten(conditions[name]['own_clean_to_noisy_drop_pp'][cohort])[metric]
                             for name in available]
                    drop_row = {**row, 'arm_or_contrast': f'{arm}_own_clean_drop',
                                'unit': 'percentage_points', **summarize(drops)}
                    aggregate.append(drop_row)
                    csv_rows.append({key: value for key, value in drop_row.items() if key != 'values'})
        for other in ('clean', 'without_case_loss'):
            for group, names in groups.items():
                available = [name for name in names if name in reports[other]]
                if not available:
                    continue
                for metric in flatten(reports[other][available[0]]['cohorts'][cohort]):
                    values = [100 * (flatten(reports['augmented'][name]['cohorts'][cohort])[metric]
                                      - flatten(reports[other][name]['cohorts'][cohort])[metric]) for name in available]
                    row = {'cohort': cohort, 'arm_or_contrast': f'augmented_minus_{other}',
                           'condition_group': group, 'observed_conditions': ','.join(available),
                           'metric': metric, 'unit': 'percentage_points',
                           **summarize(values)}
                    aggregate.append(row)
                    csv_rows.append({key: value for key, value in row.items() if key != 'values'})
    analysis_spec = {'version': 'held-out-analysis-v1', 'run_identity': identity,
                     'condition_commits_sha256': commit_hashes,
                     'source_sha256': execution_sources(ROOT, freeze)}
    analysis_identity = value_digest(analysis_spec)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with directory_lock(args.output.parent / 'held-out-analysis-lock'):
        if args.output.exists():
            marker = verify_commit(args.output)
            if marker['identity'] != analysis_identity:
                raise ValueError('Existing analysis differs; retain it and use a versioned output.')
            print('Matching complete analysis verified; saved files not rewritten.', flush=True)
            return
        with TemporaryDirectory(prefix='held-out-analysis-') as temporary:
            staging = Path(temporary)
            write(staging / 'analysis.json', {'identity': analysis_identity, 'specification': analysis_spec,
                'created_at_utc': utc_now(), 'model_inference_performed': False,
                'verified_forward_conditions': total, 'verified_cohort_score_sets': 2 * total,
                'limitations': plan['limitations'], 'conditions': reports, 'aggregates': aggregate})
            with (staging / 'aggregates.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
                writer.writeheader()
                writer.writerows(csv_rows)
            lines = ['# Held-out test: saved-prediction audit', '',
                'All 122 conditions / 244 cohort scores recomputed from original word targets.',
                'F1 below is percent. Mean ± sample SD uses three corruption seeds only.',
                'Single values have no seed SD. One training seed; no p-values.', '']
            for cohort in ('primary', 'sensitivity'):
                lines.extend([f'## {cohort}', '', '| Condition | Published | Clean | Augmented | Without case loss |',
                              '| --- | ---: | ---: | ---: | ---: |'])
                for group in groups:
                    cells = []
                    for arm in execution['model_order']:
                        row = next((row for row in aggregate if row['cohort'] == cohort
                            and row['condition_group'] == group and row['arm_or_contrast'] == arm
                            and row['metric'] == 'punctuation_macro.f1'), None)
                        if row is None:
                            cells.append('Outside frozen scope')
                        else:
                            cells.append(f"{100 * row['mean']:.4f}" + (
                                f" ± {100 * row['sample_sd']:.4f}" if row['sample_sd'] is not None else ''))
                    lines.append('| ' + ' | '.join([group, *cells]) + ' |')
                lines.append('')
            lines.extend(['## Reporting limits', '', *['- ' + item for item in plan['limitations']]])
            (staging / 'report.md').write_text('\n'.join(lines) + '\n')
            write(staging / 'complete.json', {'identity': analysis_identity, 'files': file_hashes(staging)})
            args.output.mkdir()
            for path in staging.iterdir():
                if path.name != 'complete.json':
                    shutil.copyfile(path, args.output / path.name)
            shutil.copyfile(staging / 'complete.json', args.output / 'complete.json')
            verify_commit(args.output)
    print('HELD_OUT_ANALYSIS_COMPLETE: 122 raw-target audits; 244 scores; no inference.', flush=True)


if __name__ == '__main__':
    main()
