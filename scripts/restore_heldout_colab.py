"""Restore an expired Colab session without generating data or scoring models."""

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

SOURCE_SHA = '115877201eb35cb81f3e24e870a9a0acabc44e42933b399518db56ed10ebb9b2'
INPUT_SHA = '18953b8eb131c18d261aee909ba7d1092f4178d2384b02d2139eedb4038159f1'
VALIDATION_SHA = '579a6a6ff5969ce0406ef7bbd057a5fe56c5058ebb17e6cb4918e6dc360c97eb'


def digest(path):
    import hashlib
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def installed(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--storage', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--validation-packager-sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    transfer = args.storage / 'preparation/held-out-test-v1/transfer-v1'
    source = transfer / 'held-out-source-v1.zip'
    inputs = transfer / 'held-out-inputs-v1.zip'
    validation = args.storage / 'inputs/validation-inputs-noise-v1.zip'
    if any(digest(path) != expected for path, expected in (
        (source, SOURCE_SHA), (inputs, INPUT_SHA), (validation, VALIDATION_SHA)
    )):
        raise ValueError('Pinned source/input/validation archive changed.')
    reference_name = 'artifacts/evaluation-preparation/robustness-grid-gpu-v1/calibration.json'
    with zipfile.ZipFile(inputs) as archive:
        reference = json.loads(archive.read(reference_name))
    runtime = {'python': platform.python_version(), 'torch': torch.__version__,
               'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
               'gpu': torch.cuda.get_device_name(0),
               'capability': list(torch.cuda.get_device_capability(0))}
    if any(runtime[key] != reference['gpu_runtime'][key] for key in runtime):
        raise ValueError('Base Python/Torch/CUDA/GPU differs; no fallback or package substitution.')
    if args.receipt.exists():
        raise FileExistsError('Preserve existing restoration evidence; use a new receipt path.')
    args.project.mkdir(parents=True, exist_ok=True)
    # Bootstrap only from the full source archive whose exact SHA was verified.
    with tempfile.TemporaryDirectory(prefix='held-out-source-') as temporary:
        restorer = Path(temporary) / 'package_heldout_execution.py'
        with zipfile.ZipFile(source) as archive:
            restorer.write_bytes(archive.read('scripts/package_heldout_execution.py'))
        subprocess.run([sys.executable, '-u', str(restorer), 'restore', '--archive', str(source),
            '--sha256', SOURCE_SHA, '--destination', str(args.project)], check=True)
    constraints = args.project / 'requirements/colab.constraints.txt'
    pins = [line.strip() for line in constraints.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith('#')]
    expected = dict(line.split('==') for line in pins)
    before = installed(expected)
    if before != expected:
        print('RESTORING_RECORDED_PACKAGES', json.dumps({'before': before, 'expected': expected}), flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-c', str(constraints), *pins], check=True)
    if installed(expected) != expected:
        raise ValueError('Installed packages differ from the recorded Colab constraints.')
    os.environ['PYTHONPATH'] = str(args.project / 'src')
    sys.path.insert(0, str(args.project / 'src'))
    # PYTHONPATH can name this not-yet-created directory when the process starts.
    # Invalidate its negative importer cache after restoring the source tree.
    importlib.invalidate_caches()
    subprocess.run([sys.executable, '-u', str(args.project / 'scripts/package_heldout_execution.py'),
        'restore', '--archive', str(inputs), '--sha256', INPUT_SHA,
        '--destination', str(args.inputs)], check=True)
    old_packager = transfer / 'package_evaluation_inputs.py'
    if digest(old_packager) != args.validation_packager_sha256:
        raise ValueError('Existing validation restoration implementation changed.')
    target = args.project / 'scripts/package_evaluation_inputs.py'
    if target.exists() and digest(target) != args.validation_packager_sha256:
        raise ValueError('Existing validation restorer differs; preserve it.')
    if not target.exists():
        shutil.copyfile(old_packager, target)
    if args.validation.exists():
        manifest = json.loads((args.validation / 'transfer-manifest.json').read_text())
        with zipfile.ZipFile(validation) as archive:
            if manifest != json.loads(archive.read('transfer-manifest.json')):
                raise ValueError('Existing validation cache differs from the pinned archive.')
        for name, item in manifest['files'].items():
            path = args.validation / name
            if digest(path) != item['sha256'] or path.stat().st_size != item['bytes']:
                raise ValueError('Existing validation cache bytes changed.')
        print('VALIDATION_CACHE_REUSED', flush=True)
    else:
        subprocess.run([sys.executable, '-u', str(target), '--verify', str(validation),
            '--sha256', VALIDATION_SHA, '--extract-to', str(args.validation)], check=True)
    from turkish_punctuation.heldout import load_plan, load_tokenizer
    from turkish_punctuation.heldout_runtime import check_model
    from turkish_punctuation.inference import load_published_model
    from turkish_punctuation.training_io import utc_now
    plan, _, freeze, _ = load_plan(args.project, args.inputs)
    token = json.loads((args.project / 'configs/tokenizer.json').read_text())
    try:
        load_tokenizer(args.project, plan)
        tokenizer_cached = True
    except OSError:
        tokenizer_cached = False
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(token['model_id'], revision=token['revision'],
            cache_dir=str(args.project / token['cache_directory']), use_fast=True,
            trust_remote_code=False, token=False)
    model = load_published_model(token, root=args.project, allow_download=True)
    published_hash = check_model(model, plan['models']['published'])
    del model
    gc.collect()
    result = {'version': 'held-out-colab-restoration-v1', 'status': 'restored_not_readiness',
        'created_at_utc': utc_now(), 'freeze_identity': freeze['identity'], 'base_runtime': runtime,
        'packages_before': before, 'packages_after': installed(expected),
        'source_bundle_sha256': SOURCE_SHA, 'input_bundle_sha256': INPUT_SHA,
        'validation_bundle_sha256': VALIDATION_SHA, 'tokenizer_was_cached': tokenizer_cached,
        'published_model_tensors_sha256': published_hash,
        'restore_script_sha256': digest(Path(__file__)),
        'validation_restorer_sha256': args.validation_packager_sha256,
        'noise_regenerated': False, 'training_updates': 0, 'model_forward_passes': 0,
        'test_predictions_or_scores_computed': False}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2) + '\n')
    if json.loads(args.receipt.read_text()) != result:
        raise OSError('Restoration receipt readback differs.')
    print('COLAB_RESTORATION_COMPLETE', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
