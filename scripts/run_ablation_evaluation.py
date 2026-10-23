"""Run five ablation conditions, verify saved targets and compare fixed GPU references."""

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path

from turkish_punctuation.data import read_split
from turkish_punctuation.gpu_evaluation import condition_committed, write_in_place
from turkish_punctuation.preprocessing import partition_records
from turkish_punctuation.robustness import read_predictions
from turkish_punctuation.training_io import sha256, utc_now

ROOT = Path(__file__).resolve().parents[1]


def verify_results(input_root, storage, config):
    current = storage / config['output_directory']
    analysis = current / 'analysis'
    reference_configs = {'published': 'published', 'clean': 'clean', 'augmented': 'augmented'}
    configs = {arm: json.loads((ROOT / f'configs/{name}-robustness-grid.json').read_text())
               for arm, name in reference_configs.items()}
    configs['without_case_loss'] = config
    directories = {arm: storage / settings['output_directory'] for arm, settings in configs.items()}
    summaries = {arm: json.loads((folder / 'summary.json').read_text()) for arm, folder in directories.items()}
    names = ['clean-anchor', *config['conditions']]
    release = json.loads((ROOT / 'docs/data-manifest.json').read_text())
    source = next(x for x in release['files'] if x['split'] == 'validation')
    raw = input_root / release['local_directory'] / source['filename']
    if sha256(raw) != source['sha256']:
        raise ValueError('Raw validation differs from its pinned hash.')
    eligible, _ = partition_records(read_split(raw))
    originals = {r['row_id']: r for r in eligible.to_dict(orient='records')}
    with (current / 'conditions/clean-anchor/predictions.jsonl').open() as stream:
        order = [json.loads(line)['row_id'] for line in stream]
    if len(order) != len(set(order)) or set(order) != set(originals):
        raise ValueError('Ablation does not cover all original validation rows once.')
    originals = {key: originals[key] for key in order}
    anchor = summaries['without_case_loss']
    rows = []
    for arm, summary in summaries.items():
        settings = configs[arm]
        archived_config = directories[arm] / 'source_snapshot/noise-evaluation-config.json'
        if (sha256(archived_config) != summary['identity']['config_sha256']
            or json.loads(archived_config.read_text()) != settings
            or summary['run'] != settings):
            raise ValueError(f'Frozen evaluation configuration differs: {arm}')
        if 'trained_model' in settings:
            provenance = summary['identity']['trained_model']
            if (provenance['training_identity'] != settings['trained_model']['run_identity']
                or provenance['model_tensors_sha256'] != settings['trained_model']['model_tensors_sha256']):
                raise ValueError(f'Selected training/model identity differs: {arm}')
        for key in ('packages', 'gpu_runtime', 'noise_manifest_sha256'):
            if summary['identity'][key] != anchor['identity'][key]:
                raise ValueError(f'Cross-model identity differs: {arm}/{key}')
        for key in ('batch_size', 'device', 'cpu_threads', 'seed', 'batch_order'):
            if summary['run'][key] != config[key]:
                raise ValueError(f'Cross-model settings differ: {arm}/{key}')
        if summary['identity']['batching']['row_ids_sha256'] != anchor['identity']['batching']['row_ids_sha256']:
            raise ValueError('Cross-model row order differs.')
        for name, expected in summary['identity']['source_sha256'].items():
            if sha256(directories[arm] / 'source_snapshot' / name) != expected:
                raise ValueError(f'Archived evaluation source changed: {arm}/{name}')
            if name != 'scripts/evaluate_noise.py' and expected != anchor['identity']['source_sha256'][name]:
                raise ValueError(f'Shared inference/data code differs: {arm}/{name}')
        digest = hashlib.sha256(json.dumps(summary['identity'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        for name in names:
            folder = directories[arm] / 'conditions' / name
            if not condition_committed(folder, digest):
                raise ValueError(f'Uncommitted condition: {arm}/{name}')
            expected = summary['clean_anchor'] if name == 'clean-anchor' else summary['conditions'][name]
            saved = json.loads((folder / 'metrics.json').read_text())
            if saved != expected:
                raise ValueError(f'Summary and saved report differ: {arm}/{name}')
            if (summary['identity']['condition_input_sha256'][name] != anchor['identity']['condition_input_sha256'][name]
                or summary['identity']['batching']['padded_tokens'][name] != anchor['identity']['batching']['padded_tokens'][name]):
                raise ValueError(f'Input bytes/batch membership differ: {arm}/{name}')
            _, metrics = read_predictions(folder / 'predictions.jsonl', originals)
            if metrics != saved['metrics'] or metrics['evaluated_words'] != 179164:
                raise ValueError(f'Raw-target metric audit failed: {arm}/{name}')
            row = {'model': arm, 'condition': name,
                   'macro_f1_percent': 100 * metrics['punctuation_macro']['f1'],
                   'micro_f1_percent': 100 * metrics['punctuation_micro']['f1'],
                   'drop_from_own_clean_pp': 100 * (summary['clean_anchor']['metrics']['punctuation_macro']['f1'] - metrics['punctuation_macro']['f1'])}
            for label in ('PERIOD', 'COMMA', 'QUESTION_MARK'):
                for metric in ('precision', 'recall', 'f1'):
                    row[f'{label}_{metric}_percent'] = 100 * metrics['per_class'][label][metric]
            rows.append(row)
        print(f'RAW_TARGET_RESULTS_VERIFIED {arm}: 5 saved conditions; no inference repeated.', flush=True)
    result = {'status': 'verified_complete', 'created_at_utc': utc_now(),
              'summary_sha256': {arm: sha256(folder / 'summary.json') for arm, folder in directories.items()},
              'analysis_source_sha256': sha256(Path(__file__)), 'results': rows,
              'new_evaluations': 5, 'reused_reference_evaluations': 15,
              'training_seed_count': 1, 'corruption_seeds': [586],
              'removed_noise_budget_redistributed': False, 'test_evaluated': False,
              'limitations': ['one training seed', 'one corruption seed in this ablation comparison',
                              'removing case loss lowers total noise exposure',
                              'validation results; no held-out test', 'dataset license/split provenance unresolved']}
    analysis.mkdir(parents=True, exist_ok=True)
    if (analysis / 'complete.json').exists():
        old = json.loads((analysis / 'comparison.json').read_text())
        if old['summary_sha256'] != result['summary_sha256'] or old['analysis_source_sha256'] != result['analysis_source_sha256']:
            raise ValueError('Existing analysis belongs to different inputs/source.')
        for name, expected in json.loads((analysis / 'complete.json').read_text())['files'].items():
            if sha256(analysis / name) != expected:
                raise ValueError('Committed analysis changed.')
        return
    write_in_place(analysis / 'comparison.json', result)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (analysis / 'comparison.csv').write_text(stream.getvalue())
    table = ['# Case-loss ablation: matched GPU validation', '',
             'One training seed (586), medium noise / corruption seed 586. No held-out test.', '',
             '| Condition | Published | Clean | Augmented | Without case loss | Without − augmented (pp) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in names:
        values = {r['model']: r['macro_f1_percent'] for r in rows if r['condition'] == name}
        table.append(f'| {name} | ' + ' | '.join(f'{values[a]:.4f}' for a in configs)
                     + f" | {values['without_case_loss'] - values['augmented']:+.4f} |")
    table += ['', 'Removing case loss lowers total corruption exposure; no removed budget was redistributed.',
              'All checkpoints were selected on clean validation, not noisy scores.']
    (analysis / 'comparison.md').write_text('\n'.join(table) + '\n')
    (analysis / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    write_in_place(analysis / 'complete.json', {'files': {
        name: sha256(analysis / name) for name in ('comparison.json', 'comparison.csv', 'comparison.md', Path(__file__).name)}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--storage-root', type=Path, required=True)
    args = parser.parse_args()
    config_relative = 'configs/ablation-case-loss-evaluation.json'
    config = json.loads((ROOT / config_relative).read_text())
    preparation = args.storage_root / 'preparation/case-loss-ablation-evaluation-v1/seed-586'
    readiness = preparation / 'calibration.json'
    preparation.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, '-u', 'scripts/prepare_ablation_evaluation.py',
                    '--input-root', str(args.input_root), '--storage-root', str(args.storage_root),
                    '--output', str(readiness)], cwd=ROOT, check=True)
    command = [sys.executable, '-u', 'scripts/evaluate_noise.py', '--config', config_relative,
               '--input-root', str(args.input_root), '--output-root', str(args.storage_root),
               '--storage-root', str(args.storage_root), '--calibration', str(readiness)]
    destination = args.storage_root / config['output_directory']
    if destination.exists():
        command.append('--resume')
    subprocess.run(command, cwd=ROOT, check=True)
    verify_results(args.input_root, args.storage_root, config)
    write_in_place(preparation / 'progress.json', {'status': 'complete', 'new_evaluations': 5,
                    'saved_reference_evaluations_verified': 15, 'training_updates': 0,
                    'test_evaluated': False, 'completed_at_utc': utc_now()})
    print('ABLATION_MEDIUM_GPU_EVALUATION_COMPLETE — 5 new conditions; 15 fixed reference results verified.', flush=True)


if __name__ == '__main__':
    main()
