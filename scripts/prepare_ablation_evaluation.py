"""Verify the fixed ablation GPU recipe and probe saved inputs; no training or scores."""

import argparse
import gc
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
from datasets import load_from_disk
from evaluate_noise import ROOT, SOURCE_PATHS
from transformers import AutoTokenizer

from turkish_punctuation.ablation import load_plan
from turkish_punctuation.evaluation_model import load_exported_model, training_reference
from turkish_punctuation.gpu_evaluation import (
    condition_committed,
    configure_runtime,
    write_in_place,
)
from turkish_punctuation.inference import iter_predictions
from turkish_punctuation.training_io import model_digest, sha256, utc_now, value_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--storage-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    config_path = ROOT / 'configs/ablation-case-loss-evaluation.json'
    config = json.loads(config_path.read_text())
    plan, training_config = load_plan()
    if (config['conditions'] != plan['evaluation_after_training']['conditions']
        or config['ablation_plan_sha256'] != sha256(ROOT / 'configs/ablation-case-loss.json')
        or config['comparison_condition_policy'] != 'verified_ablation_subset_v1'
        or config['storage_run_directory'] != plan['storage_training']
        or config['batch_size'] != 16 or config['cpu_threads'] != 4
        or config['seed'] != 586 or config['split'] != 'validation'):
        raise ValueError('Evaluation differs from the fixed ablation plan.')
    runtime = configure_runtime(config)
    previous_path = args.storage_root / 'artifacts/evaluations/robustness-grid-gpu-v1/calibration.json'
    previous = json.loads(previous_path.read_text())
    packages = {name: version(name) for name in ('torch', 'transformers', 'datasets', 'scikit-learn', 'numpy')}
    if runtime != previous['gpu_runtime'] or packages != previous['packages']:
        raise ValueError('GPU evaluation runtime differs from the completed grid.')
    hashes = {name: sha256(ROOT / name) for name in SOURCE_PATHS}
    for name, expected in previous['evaluation_source_sha256'].items():
        if name != 'scripts/evaluate_noise.py' and hashes[name] != expected:
            raise ValueError(f'Shared numerical/data implementation changed: {name}')
    prepared = args.input_root / config['prepared_directory']
    if sha256(prepared / 'manifest.json') != previous['noise_manifest_sha256']:
        raise ValueError('Prepared noise manifest differs from the frozen grid.')
    comparison_path = args.storage_root / config['published_comparison_directory'] / 'summary.json'
    if sha256(comparison_path) != config['published_comparison_summary_sha256']:
        raise ValueError('Frozen comparison summary changed.')
    comparison = json.loads(comparison_path.read_text())
    parent_digest = hashlib.sha256(json.dumps(comparison['identity'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    for name in ['clean-anchor', *config['conditions']]:
        folder = comparison_path.parent / 'conditions' / name
        if not condition_committed(folder, parent_digest):
            raise ValueError(f'Reference condition is not committed: {name}')
    settings = {**config['trained_model'], 'run_directory': str(args.storage_root / config['storage_run_directory'])}
    reference = training_reference(ROOT, settings)
    training = json.loads((Path(settings['run_directory']) / 'run.json').read_text())
    if (training['specification']['configuration'] != training_config
        or training['specification']['without_family'] != plan['without_family']
        or training['specification']['initialization_identity'] != plan['initialization']['identity']):
        raise ValueError('Ablation training recipe/initialization differs.')
    specification = {'config_sha256': sha256(config_path), 'runtime': runtime,
                     'source_sha256': hashes, 'preparer_sha256': sha256(Path(__file__)),
                     'runner_sha256': sha256(ROOT / 'scripts/run_ablation_evaluation.py'),
                     'previous_calibration_sha256': sha256(previous_path),
                     'noise_manifest_sha256': sha256(prepared / 'manifest.json'),
                     'model_tensors_sha256': settings['model_tensors_sha256']}
    identity = value_digest(specification)
    snapshot_paths = ['scripts/prepare_ablation_evaluation.py', 'scripts/run_ablation_evaluation.py', 'configs/ablation-case-loss-evaluation.json',
                      'configs/ablation-case-loss.json', *SOURCE_PATHS]
    snapshot = args.output.parent / 'preflight_source_snapshot'
    if args.output.exists():
        old = json.loads(args.output.read_text())
        if old['identity'] != identity or old['status'] != 'complete':
            raise ValueError('Existing evaluation readiness differs; do not overwrite it.')
        for relative in snapshot_paths:
            if sha256(snapshot / relative) != sha256(ROOT / relative):
                raise ValueError(f'Existing readiness snapshot differs: {relative}')
        print('Verified matching existing ablation evaluation readiness; probes not repeated.', flush=True)
        return
    token = json.loads((ROOT / 'configs/tokenizer.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(token['model_id'], revision=token['revision'],
        cache_dir=str(ROOT / token['cache_directory']), local_files_only=True,
        use_fast=True, trust_remote_code=False, token=False)
    names = ['clean-anchor', *config['conditions']]
    datasets = {name: load_from_disk(str(prepared / ('clean/dataset' if name == 'clean-anchor'
                 else f'conditions/{name}/dataset'))) for name in names}
    clean = datasets['clean-anchor']
    if len(clean) != 14453 or any(list(d['row_id']) != list(clean['row_id']) for d in datasets.values()):
        raise ValueError('Probe inputs have different rows/order.')
    lengths = [len(x) for x in clean['input_ids']]
    ids = list(clean['row_id'])
    order = sorted(range(len(clean)), key=lambda i: (lengths[i], ids[i]))
    batch = config['batch_size']
    count_batches = (len(order) + batch - 1) // batch
    selected = {i * (count_batches - 1) // 15 for i in range(16)}
    # Include the longest input from every planned condition in the same probes.
    position = {index: rank for rank, index in enumerate(order)}
    for dataset in datasets.values():
        sizes = [len(x) for x in dataset['input_ids']]
        worst = max(range(len(sizes)), key=lambda i: sizes[i])
        selected.add(position[worst] // batch)
    indices = [index for b in sorted(selected) for index in order[b * batch:(b + 1) * batch]]
    model = load_exported_model(reference, tokenizer, max_length=512)
    if model.config._attn_implementation != 'sdpa' or model.dtype != torch.float32:
        raise ValueError('Expected FP32 SDPA/math, as in the completed GPU grid.')
    if model_digest(model) != settings['model_tensors_sha256']:
        raise ValueError('Probe model tensors differ from the selected ablation model.')
    for _ in iter_predictions(clean.select(indices[:batch]), tokenizer, model,
            max_length=512, batch_size=batch, device='cuda'):
        pass
    measurements = {}
    for name, dataset in datasets.items():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = perf_counter()
        observed = sum(1 for _ in iter_predictions(dataset.select(indices), tokenizer, model,
            max_length=512, batch_size=batch, device='cuda'))
        torch.cuda.synchronize()
        if observed != len(indices):
            raise ValueError('Probe prediction coverage differs.')
        measurements[name] = {'records': observed, 'seconds': perf_counter() - started,
                              'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    result = {'status': 'complete', 'identity': identity, 'specification': specification,
              'gpu_runtime': runtime, 'packages': packages,
              'configs_sha256': {'configs/ablation-case-loss-evaluation.json': sha256(config_path)},
              'evaluation_source_sha256': hashes, 'noise_manifest_sha256': sha256(prepared / 'manifest.json'),
              'measurements': measurements, 'longest_planned_input_included': True,
              'sample_policy': '16 fixed clean-length batches plus all planned longest-input batches',
              'optimizer_updates': 0, 'metrics_computed': False, 'test_accessed': False,
              'created_at_utc': utc_now()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for relative in snapshot_paths:
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    write_in_place(args.output, result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print('ABLATION_EVALUATION_READY — matched runtime/model; five technical probes; zero training updates.', flush=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
