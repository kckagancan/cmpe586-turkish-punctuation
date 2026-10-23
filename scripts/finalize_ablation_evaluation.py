"""Verify downloaded ablation evidence, archive it and audit saved metrics on CPU."""

import argparse
import csv
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from run_ablation_evaluation import verify_results

from turkish_punctuation.gpu_evaluation import condition_committed
from turkish_punctuation.training_io import sha256, utc_now, value_digest

ROOT = Path(__file__).resolve().parents[1]
PREPARATION = ROOT / 'artifacts/evaluation-preparation/ablation-case-loss-medium-gpu-v1'


def read(path):
    return json.loads(path.read_text())


def evaluation_digest(identity):
    # The evaluator uses compact JSON; training/readiness value_digest uses spaces.
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def verify_archive(path, manifest):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names))
            or set(names) != set(manifest['files']) | {'archive-manifest.json'}
            or json.loads(archive.read('archive-manifest.json')) != manifest):
            raise ValueError('Archive member set/manifest differs.')
        for name, expected in manifest['files'].items():
            relative = PurePosixPath(name)
            member = archive.getinfo(name)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f'Unsafe archive path: {name}')
            with archive.open(name) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if member.file_size != expected['bytes'] or digest != expected['sha256']:
                raise ValueError(f'Archive member differs: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion-directory', type=Path,
                        default=PREPARATION / 'completion-2026-10-22')
    args = parser.parse_args()
    completion = args.completion_directory.resolve()
    download = completion / 'drive-download'
    transfer = read(completion / 'drive-transfer-verification.json')
    config = read(ROOT / 'configs/ablation-case-loss-evaluation.json')
    run_relative = config['output_directory']
    preparation_relative = 'preparation/case-loss-ablation-evaluation-v1/seed-586'
    allowed = (run_relative + '/', preparation_relative + '/')
    members = {}
    for item in transfer['files']:
        name = item['path']
        relative = PurePosixPath(name)
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name
            or not name.startswith(allowed) or name in members
            or relative.suffix not in {'.json', '.jsonl', '.py', '.csv', '.md', '.log'}):
            raise ValueError(f'Unexpected transfer member: {name}')
        path = download / name
        if (not path.is_file() or path.is_symlink()
            or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']):
            raise ValueError(f'Local download differs from its receipt: {name}')
        members[name] = {'bytes': item['bytes'], 'sha256': item['sha256']}
    actual = {str(p.relative_to(download)) for p in download.rglob('*') if p.is_file()}
    if actual != set(members) or len(members) != transfer['file_count']:
        raise ValueError('Transfer inventory differs.')

    current = download / run_relative
    remote_prep = download / preparation_relative
    summary, run = read(current / 'summary.json'), read(current / 'run.json')
    readiness = read(remote_prep / 'calibration.json')
    if readiness != read(PREPARATION / 'readiness.json'):
        raise ValueError('Downloaded readiness differs from launch readback.')
    identity = summary['identity']
    specification = readiness['specification']
    if (run['identity'] != identity or run['run'] != config or summary['run'] != config
        or identity['config_sha256'] != specification['config_sha256']
        or sha256(current / 'source_snapshot/noise-evaluation-config.json') != identity['config_sha256']
        or identity['source_sha256'] != specification['source_sha256']
        or identity['gpu_runtime'] != readiness['gpu_runtime']
        or identity['packages'] != readiness['packages']
        or identity['noise_manifest_sha256'] != readiness['noise_manifest_sha256']
        or identity['trained_model']['training_identity'] != config['trained_model']['run_identity']
        or identity['trained_model']['model_tensors_sha256'] != config['trained_model']['model_tensors_sha256']
        or readiness['identity'] != value_digest(specification)):
        raise ValueError('Completed evaluation identity differs from readiness/model/config.')
    snapshots = remote_prep / 'preflight_source_snapshot'
    expected_sources = {
        **specification['source_sha256'],
        'scripts/prepare_ablation_evaluation.py': specification['preparer_sha256'],
        'scripts/run_ablation_evaluation.py': specification['runner_sha256'],
        'configs/ablation-case-loss-evaluation.json': specification['config_sha256'],
        'configs/ablation-case-loss.json': config['ablation_plan_sha256'],
    }
    for name, expected in expected_sources.items():
        if sha256(snapshots / name) != expected or sha256(ROOT / name) != expected:
            raise ValueError(f'Preflight/local source differs: {name}')
    for name, expected in identity['source_sha256'].items():
        if sha256(current / 'source_snapshot' / name) != expected:
            raise ValueError(f'Evaluation snapshot differs: {name}')
    names = ['clean-anchor', *config['conditions']]
    if list(summary['conditions']) != config['conditions']:
        raise ValueError('Completed condition membership differs.')
    for name in names:
        if not condition_committed(current / 'conditions' / name, evaluation_digest(identity)):
            raise ValueError(f'Uncommitted result: {name}')
    analysis = current / 'analysis'
    marker = read(analysis / 'complete.json')
    for name, expected in marker['files'].items():
        if sha256(analysis / name) != expected:
            raise ValueError(f'Committed comparison changed: {name}')
    if sha256(analysis / 'run_ablation_evaluation.py') != specification['runner_sha256']:
        raise ValueError('Archived comparison runner differs.')
    progress = read(remote_prep / 'progress.json')
    if (progress['status'] != 'complete' or progress['new_evaluations'] != 5
        or progress['saved_reference_evaluations_verified'] != 15
        or progress['training_updates'] != 0 or progress['test_evaluated']):
        raise ValueError('Final progress/scope differs.')

    # Publish only an absent run, or reuse exactly matching local bytes.
    destination = ROOT / run_relative
    if not destination.exists():
        shutil.copytree(current, destination)
    for name, expected in members.items():
        if name.startswith(run_relative + '/') and sha256(ROOT / name) != expected['sha256']:
            raise ValueError(f'Published local result differs: {name}')

    # Frozen runner recomputes 20 saved metrics; its final marker makes this read only.
    verify_results(ROOT, ROOT, config)
    comparison = read(analysis / 'comparison.json')
    configs = {arm: read(ROOT / f'configs/{arm}-robustness-grid.json')
               for arm in ('published', 'clean', 'augmented')}
    configs['without_case_loss'] = config
    summaries = {arm: read(ROOT / settings['output_directory'] / 'summary.json')
                 for arm, settings in configs.items()}
    expected_rows = []
    for arm, item in summaries.items():
        for name in names:
            metrics = (item['clean_anchor'] if name == 'clean-anchor'
                       else item['conditions'][name])['metrics']
            row = {'model': arm, 'condition': name,
                   'macro_f1_percent': 100 * metrics['punctuation_macro']['f1'],
                   'micro_f1_percent': 100 * metrics['punctuation_micro']['f1'],
                   'drop_from_own_clean_pp': 100 * (item['clean_anchor']['metrics']['punctuation_macro']['f1']
                                                   - metrics['punctuation_macro']['f1'])}
            for label in ('PERIOD', 'COMMA', 'QUESTION_MARK'):
                for metric in ('precision', 'recall', 'f1'):
                    row[f'{label}_{metric}_percent'] = 100 * metrics['per_class'][label][metric]
            expected_rows.append(row)
    if (comparison['results'] != expected_rows or comparison['new_evaluations'] != 5
        or comparison['reused_reference_evaluations'] != 15
        or comparison['training_seed_count'] != 1 or comparison['corruption_seeds'] != [586]
        or comparison['test_evaluated'] or comparison['removed_noise_budget_redistributed']):
        raise ValueError('Comparison JSON values/scope differ from audited metrics.')
    csv_rows = list(csv.DictReader(io.StringIO((analysis / 'comparison.csv').read_text())))
    if len(csv_rows) != len(expected_rows):
        raise ValueError('Comparison CSV row count differs.')
    for saved, expected in zip(csv_rows, expected_rows, strict=True):
        if set(saved) != set(expected) or any(
            (saved[key] if isinstance(value, str) else float(saved[key])) != value
            for key, value in expected.items()
        ):
            raise ValueError('Comparison CSV values differ from audited metrics.')

    manifest = {'version': 'case-loss-ablation-evaluation-results-v1', 'files': members,
                'new_evaluations': 5, 'reference_evaluations_reused': 15,
                'training_performed': False, 'inference_repeated': False,
                'weights_included': False, 'test_evaluated': False}
    archive_path = completion / 'ablation-case-loss-medium-gpu-v1-results.zip'
    if not archive_path.exists():
        with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in sorted(members):
                archive.write(download / name, name)
            archive.writestr('archive-manifest.json', json.dumps(manifest, indent=2) + '\n')
    verify_archive(archive_path, manifest)
    (completion / 'archive-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    receipt = {'status': 'verified_complete', 'verified_at_utc': utc_now(),
               'verification_source_sha256': sha256(Path(__file__)),
               'archive': str(archive_path.relative_to(ROOT)), 'archive_sha256': sha256(archive_path),
               'archive_bytes': archive_path.stat().st_size, 'archive_source_files': len(members),
               'local_run': run_relative, 'evaluation_identity': evaluation_digest(identity),
               'readiness_identity': readiness['identity'], 'raw_target_metrics_recomputed': 20,
               'validation_rows_per_condition': 14453, 'word_targets_per_condition': 179164,
               'preflight_source_files_verified': len(expected_sources),
               'analysis_files_verified': len(marker['files']), 'comparison_json_csv_values_verified': True,
               'summary_sha256': comparison['summary_sha256'],
               'remote_completed_at_utc': progress['completed_at_utc'],
               'training_performed': False, 'inference_repeated': False,
               'test_evaluated': False, 'test_suite_run': False, 'drive_deleted': False}
    (completion / 'local-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
