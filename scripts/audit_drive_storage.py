"""Match a complete Drive metadata inventory to local backups; never mutate Drive."""

import argparse
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from turkish_punctuation.training_io import sha256, utc_now, value_digest

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'artifacts/storage-audit/drive-2026-10-23-v1'
HELD = ROOT / 'artifacts/evaluation-preparation/held-out-test-v1'
ABLATION = ROOT / 'artifacts/training/training-v1/ablation-without-case-loss-preparation-seed-586'
GRID = ROOT / 'artifacts/evaluation-preparation/robustness-grid-gpu-v1'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def archive_index(path, expected_sha, manifest_name='archive-manifest.json'):
    if sha256(path) != expected_sha:
        raise ValueError(f'Local archive differs from its original verified receipt: {path}')
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read(manifest_name))
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate archive members.')
        if set(archive.namelist()) != set(manifest['files']) | {manifest_name}:
            raise ValueError('Archive manifest membership differs.')
    return manifest['files']


def archive_augmented_inputs(output):
    base = ROOT / 'data/processed/training-augmentation-v1/seed-586'
    manifest = read(base / 'manifest.json')
    if value_digest(manifest['specification']) != manifest['identity']:
        raise ValueError('Original augmented-input identity differs.')
    files = {str(p.relative_to(base)): {'bytes': p.stat().st_size, 'sha256': sha256(p)}
             for p in sorted(base.rglob('*')) if p.is_file()}
    for epoch, values in manifest['epochs'].items():
        for name, expected in values['files'].items():
            if files[f'epoch-{int(epoch):02d}/{name}']['sha256'] != expected:
                raise ValueError('Audited augmented epoch file changed.')
    for name, expected in manifest['specification']['source_sha256'].items():
        if files['source_snapshot/' + name]['sha256'] != expected:
            raise ValueError('Historical augmented input source changed.')
    archive_manifest = {'version': 'augmented-training-inputs-archive-v1',
                        'prepared_identity': manifest['identity'], 'files': files,
                        'noise_regenerated': False, 'training_performed': False,
                        'weights_included': False}
    path = output / 'augmented-training-inputs-seed586.zip'
    if not path.exists():
        with zipfile.ZipFile(path, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in files:
                archive.write(base / name, name)
            archive.writestr('archive-manifest.json', json.dumps(archive_manifest, indent=2) + '\n')
    if archive_index(path, sha256(path)) != files:
        raise ValueError('New augmented-input archive differs.')
    with zipfile.ZipFile(path) as archive:
        if json.loads(archive.read('archive-manifest.json')) != archive_manifest:
            raise ValueError('Existing input archive identity differs.')
        for name, expected in files.items():
            with archive.open(name) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != expected['sha256'] or archive.getinfo(name).file_size != expected['bytes']:
                raise ValueError(f'New input archive readback differs: {name}')
    receipt = {'status': 'verified', 'archive': str(path.relative_to(ROOT)),
               'archive_sha256': sha256(path), 'archive_bytes': path.stat().st_size,
               'source_files': len(files), 'prepared_identity': manifest['identity'],
               'model_weights_read': False, 'noise_regenerated': False,
               'training_performed': False, 'drive_mutated': False}
    write(output / 'augmented-input-archive-verification.json', receipt)
    return path, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    args = parser.parse_args()
    output = args.directory
    inventory = read(output / 'drive-inventory.json')
    files = inventory['files']
    if (inventory['errors'] or inventory['unlisted'] or inventory['mutations_performed']
        or len({f['id'] for f in files}) != len(files)
        or len({f['path'] for f in files}) != len(files)):
        raise ValueError('A complete unique metadata-only project inventory is required.')
    new_archive, view_files = archive_augmented_inputs(output)
    ablation_input_zip = ABLATION / 'case-loss-ablation-v1-seed586-inputs.zip'
    ablation_spec = read(ROOT / 'configs/ablation-case-loss-inputs.json')
    ablation_files = archive_index(ablation_input_zip, ablation_spec['sha256'], 'transfer-manifest.json')
    local_verification = read(HELD / 'completion-2026-10-23/local-verification.json')
    result_archives = [
        (GRID / 'robustness-grid-gpu-v1-results.zip',
         read(GRID / 'completed-result-verification.json')['archive_sha256']),
        (ROOT / 'artifacts/evaluation-preparation/ablation-case-loss-medium-gpu-v1/completion-2026-10-22/ablation-case-loss-medium-gpu-v1-results.zip',
         read(ROOT / 'artifacts/evaluation-preparation/ablation-case-loss-medium-gpu-v1/completion-2026-10-22/local-verification.json')['archive_sha256']),
        (Path(local_verification['archive']), local_verification['archive_sha256']),
    ]
    result_index = {}
    for path, expected in result_archives:
        for name, values in archive_index(path, expected).items():
            if name.startswith('artifacts/evaluations/'):
                if name in result_index and result_index[name]['sha256'] != values['sha256']:
                    raise ValueError('Result archives disagree about an identical path.')
                result_index[name] = {**values, 'archive': str(path.relative_to(ROOT))}
    transfers = {
        'inputs/case-loss-ablation-v1-seed586-inputs.zip': (ablation_input_zip, ablation_spec['sha256']),
        'preparation/held-out-test-v1/transfer-v1/held-out-inputs-v1.zip':
            (HELD / 'transfer-v1/held-out-inputs-v1.zip', '18953b8eb131c18d261aee909ba7d1092f4178d2384b02d2139eedb4038159f1'),
        'preparation/held-out-test-v1/transfer-v1/held-out-source-v1.zip':
            (HELD / 'transfer-v1/held-out-source-v1.zip', '115877201eb35cb81f3e24e870a9a0acabc44e42933b399518db56ed10ebb9b2'),
    }
    candidates, coverage, unarchived_retained = [], [], []
    for item in files:
        name, local, backup, expected, group = item['path'], None, None, None, None
        if name.startswith('inputs/training-augmentation-v1/seed-586/'):
            relative = name.removeprefix('inputs/training-augmentation-v1/seed-586/')
            local = ROOT / 'data/processed/training-augmentation-v1/seed-586' / relative
            backup, expected, group = new_archive, view_files[relative], 'augmented_training_views'
        elif name.startswith('inputs/training-augmentation-v1/seed-586-without-case_loss/'):
            relative = name.removeprefix('inputs/training-augmentation-v1/seed-586-without-case_loss/')
            local = ROOT / 'data/processed/training-augmentation-v1/seed-586-without-case_loss' / relative
            backup, expected, group = ablation_input_zip, ablation_files['views/' + relative], 'ablation_training_views'
        elif name.startswith('artifacts/evaluations/') and not name.startswith('artifacts/evaluations/robustness-grid-gpu-v1/'):
            if name not in result_index:
                # Historical packages deliberately omitted transient run locks.
                # A new or unarchived file is retained, never inferred to be safe.
                retained = {**item, 'classification': 'retain_not_in_verified_archive'}
                unarchived_retained.append(retained)
                coverage.append(retained)
                continue
            expected, local, group = result_index[name], ROOT / name, 'archived_evaluation_copies'
            backup = ROOT / expected['archive']
        elif name in transfers:
            local, digest = transfers[name]
            expected = {'bytes': local.stat().st_size, 'sha256': digest}
            backup, group = local, 'temporary_transfer_bundles'
        if local is None:
            coverage.append({**item, 'classification': 'retain_existing_policy'})
            continue
        digest = sha256(local)
        if (local.stat().st_size != expected['bytes'] or item['bytes'] != expected['bytes']
            or digest != expected['sha256']):
            raise ValueError(f'Local archive/Drive size/source mapping differs: {name}')
        candidate = {**item, 'group': group, 'local_path': str(local.relative_to(ROOT)),
                     'local_sha256': digest, 'backup_archive': str(backup.relative_to(ROOT)),
                     'backup_verified': True, 'current_remote_content_hash_read': False,
                     'classification': 'candidate_pending_scope_approval_and_remote_readback'}
        candidates.append(candidate)
        coverage.append(candidate)
    totals = defaultdict(lambda: {'files': 0, 'bytes': 0})
    for item in files:
        row = totals[item['path'].split('/')[0]]
        row['files'] += 1
        row['bytes'] += item['bytes']
    groups = defaultdict(lambda: {'files': 0, 'bytes': 0})
    for item in candidates:
        groups[item['group']]['files'] += 1
        groups[item['group']]['bytes'] += item['bytes']
    candidate_bytes = sum(f['bytes'] for f in candidates)
    report = {'status': 'inventory_and_local_backup_matching_complete', 'created_at_utc': utc_now(),
              'project_files': len(files), 'project_folders_including_root': len(inventory['folders']),
              'project_logical_file_bytes': sum(f['bytes'] for f in files), 'top_level': dict(totals),
              'candidate_groups': dict(groups), 'candidate_files': len(candidates),
              'candidate_bytes': candidate_bytes,
              'candidate_identity': value_digest(candidates), 'candidate_files_detail': candidates,
              'retained_file_ids': [f['id'] for f in coverage if f['classification'].startswith('retain_')],
              'unarchived_files_retained': unarchived_retained,
              'original_model_archives_reused': True, 'model_weights_rehashed_or_loaded': False,
              'account_quota_read': False, 'current_remote_content_hash_read': False,
              'drive_mutated': False, 'cleanup_executed': False, 'test_suite_run': False}
    write(output / 'cleanup-candidates.json', report)
    write(output / 'local-backup-matching.json', {'files': coverage, 'drive_mutated': False})
    with (output / 'project-files.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['path', 'id', 'bytes', 'classification'])
        writer.writeheader()
        writer.writerows({key: item[key] for key in writer.fieldnames} for item in coverage)
    print(json.dumps({k: v for k, v in report.items() if k not in
                     {'candidate_files_detail', 'retained_file_ids'}}, indent=2), flush=True)


if __name__ == '__main__':
    main()
