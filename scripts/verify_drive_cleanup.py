"""Verify a one-time Drive cleanup's inputs and receipts; contains no deletion code."""

import argparse
import json
from pathlib import Path

from turkish_punctuation.training_io import sha256, utc_now, value_digest

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'artifacts/storage-audit/drive-2026-10-23-v1'
LIVE_NOTEBOOK = '1SxfoAohLpM2UoLsz3yBe5M3nwgqDlO_P'
ORIGINAL_IDENTITY = 'db40e93dce484d9dce363ec5994052836d142a515bc73ade52234b42a9e09368'
REVISED_IDENTITY = '7ed5bdd0c1ad24eece5a59fe4ac3111159b7b152174edff26f2a333d4f544a4a'


def reviewed_candidates(output):
    original_path = AUDIT / 'cleanup-candidates.json'
    revised_path = output / 'cleanup-candidates-v2.json'
    if sha256(original_path) != '6fa62a0c18c44dd42c7e2c295fb98bd0e13b074cbbfe8dfa8a0f115ba06fe088':
        raise ValueError('Historical candidate manifest changed.')
    original, revised = read(original_path), read(revised_path)
    selected = revised['candidate_files_detail']
    if (value_digest(original['candidate_files_detail']) != ORIGINAL_IDENTITY
        or revised['supersedes_candidate_identity'] != ORIGINAL_IDENTITY
        or value_digest(selected) != REVISED_IDENTITY
        or revised['candidate_identity'] != REVISED_IDENTITY
        or len(selected) != 1148 or sum(row['bytes'] for row in selected) != 925753181):
        raise ValueError('Versioned candidate identity differs.')
    fields = ('id', 'path', 'bytes', 'parent_id')
    if ([[row[k] for k in fields] for row in selected]
        != [[row[k] for k in fields] for row in original['candidate_files_detail']]):
        raise ValueError('Reviewed Drive deletion scope changed.')
    return revised, revised_path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def file_map(inventory):
    if inventory['errors'] or inventory['unlisted']:
        raise ValueError('Current Drive tree is incomplete.')
    files = inventory['files']
    if len({x['id'] for x in files}) != len(files) or len({x['path'] for x in files}) != len(files):
        raise ValueError('Ambiguous Drive IDs/paths.')
    return {item['id']: item for item in files}


def expected_backups():
    evaluation = ROOT / 'artifacts/evaluation-preparation'
    grid = evaluation / 'robustness-grid-gpu-v1'
    ablation = evaluation / 'ablation-case-loss-medium-gpu-v1/completion-2026-10-22'
    held = evaluation / 'held-out-test-v1'
    recipe = read(ROOT / 'configs/ablation-case-loss-inputs.json')
    new = read(AUDIT / 'augmented-input-archive-verification.json')
    return {
        new['archive']: new['archive_sha256'],
        'artifacts/training/training-v1/ablation-without-case-loss-preparation-seed-586/case-loss-ablation-v1-seed586-inputs.zip': recipe['sha256'],
        str((grid / 'robustness-grid-gpu-v1-results.zip').relative_to(ROOT)):
            read(grid / 'completed-result-verification.json')['archive_sha256'],
        str((ablation / 'ablation-case-loss-medium-gpu-v1-results.zip').relative_to(ROOT)):
            read(ablation / 'local-verification.json')['archive_sha256'],
        str((held / 'completion-2026-10-23/held-out-test-v1-results.zip').relative_to(ROOT)):
            read(held / 'completion-2026-10-23/local-verification.json')['archive_sha256'],
        str((held / 'transfer-v1/held-out-inputs-v1.zip').relative_to(ROOT)):
            '18953b8eb131c18d261aee909ba7d1092f4178d2384b02d2139eedb4038159f1',
        str((held / 'transfer-v1/held-out-source-v1.zip').relative_to(ROOT)):
            '115877201eb35cb81f3e24e870a9a0acabc44e42933b399518db56ed10ebb9b2',
        str((AUDIT / 'cleanup-2026-10-23/augmented-colab-manifest-supplement.zip').relative_to(ROOT)):
            'edd9e56f606285f2d2a5e5c420f267f0dd00bacce920c2a8425ec90ed0cb036a',
    }


def verify_archives(expected):
    for name, digest in expected.items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Local backup changed: {name}')


def before(args):
    candidates, candidate_path = reviewed_candidates(args.output)
    selected = candidates['candidate_files_detail']
    original = file_map(read(AUDIT / 'drive-inventory.json'))
    current = file_map(read(args.metadata))
    content = read(args.readback)
    if sha256(args.readback) != args.readback_sha256:
        raise ValueError('Downloaded readback differs from the observed Colab SHA.')
    if (content['file_count'] != 1148 or content['bytes'] != 925753181
        or content['model_weights_read'] or content['training_or_inference_performed']
        or content['drive_deleted'] or set(content['files']) != {r['path'] for r in selected}):
        raise ValueError('Current mounted-Drive content audit scope differs.')
    for row in selected:
        remote = current[row['id']]
        if (remote['path'] != row['path'] or remote['bytes'] != row['bytes']
            or remote['parent_id'] != row['parent_id']
            or remote['modified_time'] != row['modified_time']
            or content['files'][row['path']] != {'bytes': row['bytes'], 'sha256': row['local_sha256']}):
            raise ValueError(f'Current Drive copy differs: {row["path"]}')
        local = ROOT / row['local_path']
        if local.stat().st_size != row['bytes'] or sha256(local) != row['local_sha256']:
            raise ValueError(f'Local copy changed after backup matching: {row["path"]}')
    for identity in candidates['retained_file_ids']:
        old, new = original[identity], current[identity]
        if old['path'] != new['path'] or (identity != LIVE_NOTEBOOK and old['bytes'] != new['bytes']):
            raise ValueError(f'Protected file changed: {old["path"]}')
    backups = expected_backups()
    if {row['backup_archive'] for row in selected} != set(backups):
        raise ValueError('Reviewed backup set changed.')
    verify_archives(backups)
    receipt = {'status': 'verified_for_scoped_cleanup', 'verified_at_utc': utc_now(),
               'candidate_identity': candidates['candidate_identity'],
               'candidate_manifest_sha256': sha256(candidate_path),
               'original_candidate_identity': ORIGINAL_IDENTITY,
               'readback_sha256': sha256(args.readback), 'metadata_sha256': sha256(args.metadata),
               'before_metadata_path': str(args.metadata.resolve().relative_to(ROOT)),
               'candidate_files': 1148, 'candidate_bytes': 925753181,
               'current_remote_content_hashes_match_local_backups': True,
               'local_file_hashes_rechecked': 1148, 'local_backup_archive_hashes_rechecked': backups,
               'retained_files_present': len(candidates['retained_file_ids']),
               'allowed_live_notebook_metadata_change': LIVE_NOTEBOOK,
               'scope_decision': 'User delegated deletion decisions for backed-up/unneeded project Drive copies; assistant selected the previously reviewed 925.75 MB scope.',
               'verification_source_sha256': sha256(Path(__file__)),
               'training_or_inference_performed': False, 'weights_rehashed_or_loaded': False,
               'deletion_performed_by_this_verifier': False, 'test_suite_run': False}
    write(args.output / 'ready-for-cleanup.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'local_backup_archive_hashes_rechecked'}, indent=2))


def after(args):
    candidates, candidate_path = reviewed_candidates(args.output)
    ready = read(args.output / 'ready-for-cleanup.json')
    if sha256(candidate_path) != ready['candidate_manifest_sha256']:
        raise ValueError('Cleanup manifest changed during execution.')
    rows = [json.loads(line) for line in (args.output / 'deletion-receipts.jsonl').read_text().splitlines()]
    ids = {r['id'] for r in candidates['candidate_files_detail']}
    if (len(rows) != 1148 or {r['id'] for r in rows} != ids
        or len({r['id'] for r in rows}) != len(rows) or any(r['status'] != 'deleted' for r in rows)):
        raise ValueError('The complete exact deletion receipt set is required.')
    original = file_map(read(AUDIT / 'drive-inventory.json'))
    current = file_map(read(args.metadata))
    before_path = ROOT / ready['before_metadata_path']
    if sha256(before_path) != ready['metadata_sha256']:
        raise ValueError('Pre-cleanup inventory changed.')
    before_files = file_map(read(before_path))
    if ids & set(current):
        raise ValueError('Deleted file ID still appears in the current project tree.')
    if set(before_files) - ids - set(current):
        raise ValueError('A file outside the reviewed scope disappeared.')
    for identity in set(before_files) - ids:
        old, new = before_files[identity], current[identity]
        if (old['path'] != new['path'] or old['parent_id'] != new['parent_id']
            or (identity != LIVE_NOTEBOOK and old['bytes'] != new['bytes'])):
            raise ValueError(f'Pre-cleanup protected file changed: {old["path"]}')
    for identity in candidates['retained_file_ids']:
        old, new = original[identity], current[identity]
        if old['path'] != new['path'] or (identity != LIVE_NOTEBOOK and old['bytes'] != new['bytes']):
            raise ValueError(f'Protected file changed or disappeared: {old["path"]}')
    verify_archives(ready['local_backup_archive_hashes_rechecked'])
    receipt = {'status': 'scoped_cleanup_verified_complete', 'verified_at_utc': utc_now(),
               'candidate_identity': ready['candidate_identity'], 'deleted_files': 1148,
               'deleted_logical_file_bytes': 925753181, 'permanent_drive_copies_removed': True,
               'all_candidate_ids_absent_from_fresh_project_tree': True,
               'all_291_original_protected_file_ids_present': True,
               'local_backup_archive_hashes_still_match': True,
               'post_project_files': len(current),
               'post_project_logical_file_bytes': sum(x['bytes'] for x in current.values()),
               'new_preserved_file_ids': sorted(set(current) - set(original)),
               'post_metadata_sha256': sha256(args.metadata),
               'account_quota_read': False, 'account_wide_trash_operation': False,
               'training_or_inference_performed': False, 'test_suite_run': False}
    write(args.output / 'cleanup-verification.json', receipt)
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['before', 'after'])
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--readback', type=Path)
    parser.add_argument('--readback-sha256')
    parser.add_argument('--output', type=Path, default=AUDIT / 'cleanup-2026-10-23')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'before':
        if args.readback is None or args.readback_sha256 is None:
            parser.error('before requires --readback and --readback-sha256')
        before(args)
    else:
        after(args)


if __name__ == '__main__':
    main()
