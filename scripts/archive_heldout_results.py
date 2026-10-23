"""Archive completed held-out evidence; restore and compare CPU saved-prediction audits."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath

from turkish_punctuation.gpu_evaluation import condition_committed
from turkish_punctuation.heldout import verify_commit
from turkish_punctuation.training_io import sha256, utc_now, value_digest

ROOT = Path(__file__).resolve().parents[1]
RUN = 'artifacts/evaluations/held-out-test-v1'
LOCAL = ROOT / 'artifacts/evaluation-preparation/held-out-test-v1/completion-2026-10-23'
LAUNCH = LOCAL.parent / 'launch-2026-10-23'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def completed_run(directory):
    provenance = verify_commit(directory / 'provenance')
    run = read(directory / 'provenance/run.json')
    progress = read(directory / 'progress.json')
    identity = run['identity']
    if (identity != value_digest(run['specification']) or provenance['identity'] != identity
        or progress['identity'] != identity or progress['status'] != 'inference_complete'
        or progress['completed_conditions'] != 122 or progress['planned_conditions'] != 122):
        raise ValueError('The complete 122-condition inference receipt is required.')
    marker = verify_commit(directory / 'analysis-v1')
    analysis = read(directory / 'analysis-v1/analysis.json')
    if (analysis['identity'] != marker['identity']
        or value_digest(analysis['specification']) != marker['identity']
        or analysis['specification']['run_identity'] != identity
        or analysis['verified_forward_conditions'] != 122
        or analysis['verified_cohort_score_sets'] != 244
        or analysis['model_inference_performed']):
        raise ValueError('Completed saved-prediction analysis differs.')
    plan = read(directory / 'provenance/source_snapshot/configs/held-out-test.json')
    expected = {}
    for arm in plan['models']:
        names = (plan['run_matrix']['ablation_conditions'] if arm == plan['run_matrix']['ablation_model']
                 else plan['run_matrix']['main_conditions'])
        for name in names:
            condition = directory / 'models' / arm / 'conditions' / name
            if not condition_committed(condition, identity):
                raise ValueError(f'Uncommitted condition: {arm}/{name}')
            expected[f'{arm}/{name}'] = sha256(condition / 'complete.json')
    actual = {str(p.parent.relative_to(directory / 'models')) for p in
              (directory / 'models').glob('*/conditions/*/complete.json')}
    if (len(expected) != 122 or len(actual) != 122
        or expected != analysis['specification']['condition_commits_sha256']):
        raise ValueError('Frozen condition coverage/commit hashes differ.')
    return run, analysis


def verify_archive(path, expected_sha=None):
    if expected_sha is not None and sha256(path) != expected_sha:
        raise ValueError('Downloaded ZIP differs from the remote SHA-256.')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members.')
        manifest = json.loads(archive.read('archive-manifest.json'))
        if set(names) != set(manifest['files']) | {'archive-manifest.json'}:
            raise ValueError('ZIP member inventory differs.')
        for name, expected in manifest['files'].items():
            relative = PurePosixPath(name)
            member = archive.getinfo(name)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                or member.is_dir() or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f'Unsafe ZIP member: {name}')
            with archive.open(name) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != expected['sha256'] or member.file_size != expected['bytes']:
                raise ValueError(f'ZIP member differs: {name}')
    return manifest


def restore(args):
    receipt = read(args.receipt)
    if (sha256(args.archive) != receipt['archive_sha256']
        or args.archive.stat().st_size != receipt['archive_bytes']):
        raise ValueError('Downloaded ZIP differs from its remote receipt.')
    inventory = []
    with zipfile.ZipFile(args.archive) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate native ZIP members.')
        for member in archive.infolist():
            name = member.filename
            relative = PurePosixPath(name)
            if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                or relative.parts[0] != 'held-out-test-v1'
                or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError(f'Unsafe native ZIP member: {name}')
            if member.is_dir():
                continue
            canonical = str(PurePosixPath(RUN).parent / relative)
            target = ROOT / canonical
            data = archive.read(name)  # Also checks ZIP CRC before publishing bytes.
            expected = {'bytes': member.file_size, 'sha256': hashlib.sha256(data).hexdigest()}
            if target.exists():
                if target.is_symlink() or not target.is_file() or sha256(target) != expected['sha256']:
                    raise ValueError(f'Existing local evidence differs: {target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as stream:
                    stream.write(data)
            if target.stat().st_size != expected['bytes'] or sha256(target) != expected['sha256']:
                raise ValueError(f'Local readback differs: {target}')
            inventory.append({'archive_member': name, 'local_path': str(target), **expected})
    run, analysis = completed_run(ROOT / RUN)
    if (run['identity'] != receipt['run_identity'] or analysis['identity'] != receipt['analysis_identity']
        or len(inventory) != receipt['source_files']):
        raise ValueError('Native ZIP/analysis/receipt identities differ.')
    write(args.output / 'restoration.json', {'status': 'local_archive_restored_and_verified',
          'verified_at_utc': utc_now(), 'archive': str(args.archive.resolve()),
          'archive_sha256': receipt['archive_sha256'], 'files': inventory,
          'inference_performed': False, 'drive_deleted': False})
    print(f'LOCAL_ARCHIVE_RESTORED: {len(inventory)} exact files; 122 committed conditions.', flush=True)


def verify_local(args):
    run, remote = completed_run(ROOT / RUN)
    verify_commit(args.local_analysis)
    local = read(args.local_analysis / 'analysis.json')
    if ({k: v for k, v in remote.items() if k != 'created_at_utc'}
        != {k: v for k, v in local.items() if k != 'created_at_utc'}):
        raise ValueError('CPU recomputation differs from the complete Colab analysis.')
    for name in ('aggregates.csv', 'report.md'):
        if sha256(ROOT / RUN / 'analysis-v1' / name) != sha256(args.local_analysis / name):
            raise ValueError(f'CPU aggregate/report differs: {name}')
    archive_receipt = read(args.output / 'remote-archive.json')
    restoration = read(args.output / 'restoration.json')
    if sha256(Path(restoration['archive'])) != archive_receipt['archive_sha256']:
        raise ValueError('Local archive changed after restoration.')
    ready = read(LAUNCH / 'readiness.json')
    ready_marker = read(LAUNCH / 'readiness-complete.json')
    if (ready['specification'] != run['specification']['runtime_binding']
        or ready['identity'] != run['specification']['readiness_identity']
        or sha256(LAUNCH / 'readiness-complete.json') != run['specification']['readiness_commit_sha256']
        or sha256(LAUNCH / 'readiness.json') != ready_marker['files']['readiness.json']):
        raise ValueError('Local original readiness receipts differ from the inference binding.')
    plan = read(ROOT / 'configs/held-out-test.json')
    freeze = ROOT / plan['paths']['freeze_directory']
    if verify_commit(freeze)['identity'] != ready['specification']['freeze_identity']:
        raise ValueError('Local protocol/row-mask freeze differs.')
    members = {str(p.relative_to(ROOT)): p for p in (ROOT / RUN).rglob('*') if p.is_file()}
    members.update({str(p.relative_to(ROOT)): p for p in freeze.rglob('*') if p.is_file()})
    for name in ('readiness.json', 'readiness-complete.json', 'local-readiness-verification.json',
                 'initial-observation.json', 'local-initial-observation-verification.json',
                 'launch.json', 'restoration-r2.json'):
        members['launch_evidence/' + name] = LAUNCH / name
    members['verification_tools/archive_heldout_results.py'] = Path(__file__).resolve()
    members['local_audit/analysis.log'] = args.output / 'local-analysis.log'
    manifest = {'version': 'held-out-test-v1-results-v1', 'run_identity': run['identity'],
                'analysis_identity': local['identity'], 'forward_conditions': 122, 'cohort_score_sets': 244,
                'files': {name: {'bytes': path.stat().st_size, 'sha256': sha256(path)}
                          for name, path in sorted(members.items())},
                'weights_included': False, 'inference_repeated': False, 'drive_deleted': False}
    archive_path = args.output / 'held-out-test-v1-results.zip'
    if not archive_path.exists():
        with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in sorted(members.items()):
                archive.write(path, name)
            archive.writestr('archive-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    if verify_archive(archive_path) != manifest:
        raise ValueError('Final local archive differs; preserve it and version the output.')
    write(args.output / 'archive-manifest.json', manifest)
    receipt = {'status': 'verified_complete', 'verified_at_utc': utc_now(),
               'run_identity': run['identity'], 'analysis_identity': local['identity'],
               'archive': str(archive_path), 'archive_sha256': sha256(archive_path),
               'archive_bytes': archive_path.stat().st_size, 'source_files': len(members),
               'remote_transfer_sha256': archive_receipt['archive_sha256'],
               'raw_target_conditions_recomputed_locally': 122, 'cohort_scores_recomputed_locally': 244,
               'local_analysis_matches_colab': True, 'aggregate_csv_exact_match': True,
               'verification_source_sha256': sha256(Path(__file__)),
               'training_performed': False, 'inference_repeated': False,
               'test_suite_run': False, 'drive_deleted': False}
    write(args.output / 'local-verification.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode', required=True)
    command = sub.add_parser('restore')
    command.add_argument('--archive', type=Path, required=True)
    command.add_argument('--receipt', type=Path, required=True)
    command.add_argument('--output', type=Path, default=LOCAL)
    command.set_defaults(action=restore)
    command = sub.add_parser('verify-local')
    command.add_argument('--local-analysis', type=Path, required=True)
    command.add_argument('--output', type=Path, default=LOCAL)
    command.set_defaults(action=verify_local)
    args = parser.parse_args()
    args.action(args)


if __name__ == '__main__':
    main()
