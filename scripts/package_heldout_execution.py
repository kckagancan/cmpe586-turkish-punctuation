"""Pack verified held-out inputs/source, or safely restore a SHA-pinned bundle."""

import argparse
import hashlib
import json
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def inspect_bundle(path):
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        names = [entry.filename for entry in entries]
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members.')
        for entry in entries:
            name = PurePosixPath(entry.filename)
            if (name.is_absolute() or '..' in name.parts or '\\' in entry.filename
                or str(name) != entry.filename or entry.is_dir()
                or stat.S_ISLNK(entry.external_attr >> 16)):
                raise ValueError(f'Unsafe archive member: {entry.filename}')
        manifest = json.loads(archive.read('bundle-manifest.json'))
        if set(names) != {'bundle-manifest.json', *manifest['files']}:
            raise ValueError('Archive membership differs from its manifest.')
        for name, expected in manifest['files'].items():
            result = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    result.update(block)
            if result.hexdigest() != expected:
                raise ValueError(f'Archive readback differs: {name}')
    return manifest


def restore(path, expected_sha256, destination):
    if digest(path) != expected_sha256:
        raise ValueError('Bundle SHA-256 differs from the pinned transfer receipt.')
    manifest = inspect_bundle(path)
    destination = destination.resolve()
    # Preflight the full set before creating any destination file.
    for name, expected in manifest['files'].items():
        target = destination / name
        if not target.resolve().is_relative_to(destination):
            raise ValueError('Destination escapes the selected root.')
        if target.exists() and (not target.is_file() or digest(target) != expected):
            raise ValueError(f'Existing destination differs; preserve it: {target}')
    with zipfile.ZipFile(path) as archive:
        for name, expected in manifest['files'].items():
            target = destination / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                # Stage on the same filesystem; interruption cannot leave a partial final file.
                with TemporaryDirectory(prefix='.held-out-restore-', dir=target.parent) as temporary:
                    staged = Path(temporary) / 'file'
                    with archive.open(name) as source, staged.open('xb') as output:
                        shutil.copyfileobj(source, output)
                    if digest(staged) != expected:
                        raise ValueError(f'Restored bytes differ: {name}')
                    if target.exists():
                        raise FileExistsError(f'A concurrent writer created {target}')
                    staged.rename(target)
            if digest(target) != expected:
                raise ValueError(f'Restored readback differs: {name}')
    print(f"RESTORED {manifest['kind']}: {len(manifest['files'])} matching files; existing bytes reused.", flush=True)


def build(path, kind, files, freeze_identity):
    manifest = {'version': 'held-out-transfer-v1', 'kind': kind,
                'freeze_identity': freeze_identity,
                'files': {name: digest(source) for name, source in sorted(files.items())}}
    if path.exists():
        if inspect_bundle(path) != manifest:
            raise ValueError('Existing bundle differs; retain it and create a new version.')
    else:
        with TemporaryDirectory(prefix='.held-out-pack-', dir=path.parent) as temporary:
            staging = Path(temporary) / path.name
            with zipfile.ZipFile(staging, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for name, source in sorted(files.items()):
                    archive.write(source, name)
                archive.writestr('bundle-manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
            if inspect_bundle(staging) != manifest:
                raise ValueError('Bundle manifest readback differs.')
            staging.rename(path)
    result = {'filename': path.name, 'sha256': digest(path), 'bytes': path.stat().st_size,
              'files': len(files), 'kind': kind}
    print(json.dumps(result), flush=True)
    return result


def pack(output):
    from turkish_punctuation.heldout import execution_sources, verify_inputs
    from turkish_punctuation.training_io import utc_now
    root = Path(__file__).resolve().parents[1]
    plan, execution, freeze, _, manifest, _ = verify_inputs(root, root)
    source_files = {name: root / name for name in execution_sources(root, freeze)}
    for name in ('scripts/package_heldout_execution.py', 'src/turkish_punctuation/__init__.py',
                 'requirements/colab.constraints.txt'):
        source_files[name] = root / name
    input_files = {}
    for relative in (execution['prepared_directory'], plan['paths']['freeze_directory']):
        for path in (root / relative).rglob('*'):
            if path.is_file():
                input_files[path.relative_to(root).as_posix()] = path
    release = json.loads((root / plan['data_manifest']['path']).read_text())
    raw = root / release['local_directory'] / manifest['raw_source']['filename']
    input_files[raw.relative_to(root).as_posix()] = raw
    reference = root / plan['inference']['reference_runtime_receipt']
    if digest(reference) != plan['inference']['reference_runtime_receipt_sha256']:
        raise ValueError('Pinned reference runtime receipt changed.')
    input_files[reference.relative_to(root).as_posix()] = reference
    output.mkdir(parents=True, exist_ok=True)
    bundles = [build(output / 'held-out-inputs-v1.zip', 'inputs', input_files, freeze['identity']),
               build(output / 'held-out-source-v1.zip', 'source', source_files, freeze['identity'])]
    receipt = {'version': 'held-out-transfer-v1', 'freeze_identity': freeze['identity'],
        'bundles': bundles, 'contains_model_weights': False, 'contains_predictions': False,
        'validation_inputs': 'Reuse audited validation-inputs-noise-v1.zip separately; not duplicated.',
        'created_at_utc': utc_now()}
    target = output / 'transfer.json'
    if target.exists():
        existing = json.loads(target.read_text())
        if {k: v for k, v in existing.items() if k != 'created_at_utc'} != {
            k: v for k, v in receipt.items() if k != 'created_at_utc'
        }:
            raise ValueError('Existing transfer receipt differs; retain it.')
    else:
        target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    pack_parser = commands.add_parser('pack')
    pack_parser.add_argument('--output', type=Path, required=True)
    restore_parser = commands.add_parser('restore')
    restore_parser.add_argument('--archive', type=Path, required=True)
    restore_parser.add_argument('--sha256', required=True)
    restore_parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'pack':
        pack(args.output)
    else:
        restore(args.archive, args.sha256, args.destination)


if __name__ == '__main__':
    main()
