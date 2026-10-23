"""Separate held-out orchestration gates; frozen development implementations stay intact."""

import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from datasets import load_from_disk
from transformers import AutoTokenizer

from turkish_punctuation.data import read_split
from turkish_punctuation.evaluation_model import load_exported_model, training_reference
from turkish_punctuation.inference import load_published_model
from turkish_punctuation.noise_audit import encode_and_audit
from turkish_punctuation.preprocessing import partition_records
from turkish_punctuation.tokenization import LABEL_TO_ID
from turkish_punctuation.training_io import file_hashes, sha256, value_digest

INPUT_SOURCES = ('src/turkish_punctuation/heldout.py', 'scripts/prepare_heldout_inputs.py',
                 'configs/held-out-execution.json', 'configs/held-out-test.json')
EXECUTION_SOURCES = (*INPUT_SOURCES, 'scripts/ready_heldout_evaluation.py',
                     'scripts/evaluate_heldout.py', 'scripts/analyze_heldout.py',
                     'src/turkish_punctuation/heldout_runtime.py',
                     'src/turkish_punctuation/robustness.py')


def read(path):
    return json.loads(Path(path).read_text())


def verify_commit(directory):
    """Verify an immutable tree's exact file set, not only its named subset."""
    marker = read(directory / 'complete.json')
    actual = file_hashes(directory, exclude=('complete.json',))
    if actual != marker['files']:
        raise ValueError(f'Committed tree changed: {directory}')
    return marker


def load_plan(root, input_root=None):
    execution = read(root / 'configs/held-out-execution.json')
    path = root / execution['protocol_config']
    plan = read(path)
    if sha256(path) != execution['protocol_config_sha256'] or plan['split'] != 'test':
        raise ValueError('Execution does not bind the exact frozen test protocol.')
    base = input_root if input_root is not None else root
    freeze_directory = base / plan['paths']['freeze_directory']
    marker = verify_commit(freeze_directory)
    freeze = read(freeze_directory / 'freeze.json')
    if (freeze['identity'] != value_digest(freeze['specification'])
        or freeze['identity'] != execution['freeze_identity']
        or marker['identity'] != freeze['identity']
        or freeze['specification']['config_sha256'] != sha256(path)
        or execution['prepared_directory'] != plan['paths']['future_inputs']
        or execution['output_directory'] != plan['paths']['future_output_root']
        or execution['model_order'] != list(plan['models'])
        or execution['test_scoring_enabled_by_default']):
        raise ValueError('Protocol/row-mask freeze differs from the execution wiring.')
    for name, expected in freeze['specification']['source_sha256'].items():
        if sha256(freeze_directory / 'source_snapshot' / name) != expected:
            raise ValueError(f'Frozen source snapshot changed: {name}')
        # Historical training metadata lives in the verified snapshot. Actual
        # retained exports are independently checked at model load on Drive.
        if not name.startswith('artifacts/') and sha256(root / name) != expected:
            raise ValueError(f'Frozen implementation/configuration changed: {name}')
    cohorts = read(freeze_directory / 'cohorts.json')
    if value_digest(cohorts) != freeze['specification']['cohorts_sha256']:
        raise ValueError('Frozen test row masks changed.')
    return plan, execution, freeze, cohorts


def load_test(root, input_root, plan, cohorts):
    release = read(root / plan['data_manifest']['path'])
    source = next(item for item in release['files'] if item['split'] == 'test')
    path = input_root / release['local_directory'] / source['filename']
    if sha256(path) != source['sha256'] or path.stat().st_size != source['bytes']:
        raise ValueError('Original held-out CSV differs from its pinned source.')
    clean, quarantine = partition_records(read_split(path))
    if (clean['row_id'].tolist() != cohorts['primary_row_ids']
        or quarantine['row_id'].tolist() != cohorts['quarantined_row_ids']):
        raise ValueError('Test eligibility differs from the frozen original IDs.')
    return clean, source


def load_tokenizer(root, plan):
    config = read(root / 'configs/tokenizer.json')
    if config['label_to_id'] != LABEL_TO_ID or config['max_input_length'] != plan['inference']['max_input_length']:
        raise ValueError('Tokenizer/labels differ from the held-out protocol.')
    return AutoTokenizer.from_pretrained(config['model_id'], revision=config['revision'],
        cache_dir=str(root / config['cache_directory']), local_files_only=True,
        use_fast=True, trust_remote_code=False, token=False)


def condition_directory(prepared, name):
    return prepared / ('clean' if name == 'clean-anchor' else 'conditions/' + name)


def verify_inputs(root, input_root, *, retokenize=False):
    """Audit committed bytes and all original word targets, optionally re-encode."""
    plan, execution, freeze, cohorts = load_plan(root, input_root)
    prepared = input_root / execution['prepared_directory']
    marker = verify_commit(prepared)
    manifest = read(prepared / 'manifest.json')
    if (manifest['freeze_identity'] != freeze['identity']
        or manifest['execution_config_sha256'] != sha256(root / 'configs/held-out-execution.json')
        or marker['identity'] != value_digest(manifest['specification'])
        or manifest['model_evaluation_performed'] or manifest['training_performed']):
        raise ValueError('Test input preparation identity differs.')
    if manifest['specification']['source_sha256'] != {
        name: sha256(root / name) for name in INPUT_SOURCES
    }:
        raise ValueError('Input preparation implementation changed; retain this version.')
    clean, source = load_test(root, input_root, plan, cohorts)
    if manifest['raw_source'] != source or list(manifest['conditions']) != plan['run_matrix']['main_conditions']:
        raise ValueError('Prepared source/condition membership differs from the plan.')
    tokenizer = load_tokenizer(root, plan) if retokenize else None
    for name, audit in manifest['conditions'].items():
        directory = condition_directory(prepared, name)
        dataset = load_from_disk(str(directory / 'dataset'))
        if list(dataset['row_id']) != cohorts['primary_row_ids']:
            raise ValueError(f'Encoded row IDs/order differ: {name}')
        frame = clean if name == 'clean-anchor' else pd.DataFrame(
            json.loads(line) for line in (directory / 'records.jsonl').read_text().splitlines())
        if (frame['row_id'].tolist() != clean['row_id'].tolist()
            or frame['tags'].tolist() != clean['tags'].tolist()
            or frame['tokens'].map(len).tolist() != clean['tokens'].map(len).tolist()):
            raise ValueError(f'Original word targets/IDs differ: {name}')
        if (not audit['changes']['invariants_passed']
            or not audit['tokenization']['all_first_subword_targets_preserved']):
            raise ValueError(f'Saved input audit did not pass: {name}')
        for labels, tags in zip(dataset['labels'], clean['tags'], strict=True):
            if [x for x in labels if x != -100] != [LABEL_TO_ID[tag] for tag in tags]:
                raise ValueError(f'Saved first-subword word targets differ: {name}')
        if retokenize:
            encoded, _ = encode_and_audit(frame, tokenizer, max_length=512, batch_size=256)
            if set(encoded.column_names) != set(dataset.column_names) or any(
                encoded[key][:] != dataset[key][:] for key in encoded.column_names
            ):
                raise ValueError(f'Re-tokenized inputs differ: {name}')
        print(f'INPUTS_VERIFIED {name}: {len(dataset)} rows; retokenized={retokenize}', flush=True)
    return plan, execution, freeze, cohorts, manifest, clean


def runtime_settings(plan):
    return {**plan['inference'], 'runtime_policy': 'fresh_gpu_anchor',
            'recompute_clean_anchor': True}


def packages():
    return {name: version(name) for name in ('torch', 'transformers', 'datasets', 'scikit-learn', 'numpy')}


def execution_sources(root, freeze):
    names = {*EXECUTION_SOURCES, *(name for name in freeze['specification']['source_sha256']
                                 if not name.startswith('artifacts/'))}
    return {name: sha256(root / name) for name in sorted(names)}


def load_model(root, plan, arm, storage_root, tokenizer):
    model = plan['models'][arm]
    if arm == 'published':
        return load_published_model(read(root / 'configs/tokenizer.json'), root=root), model
    directory = (storage_root / model['storage_run_directory'] if storage_root is not None
                 else root / model['local_run_directory'])
    for filename, expected in (('run.json', model['run_sha256']), ('summary.json', model['summary_sha256'])):
        if sha256(directory / filename) != expected:
            raise ValueError(f'Pinned training metadata changed: {arm}/{filename}')
    reference = training_reference(root, {'run_directory': str(directory),
        'run_identity': model['training_identity'], 'arm': model['training_arm'],
        'model_tensors_sha256': model['model_tensors_sha256']})
    export = directory / f"models/epoch-{model['selected_epoch']:02d}" / 'export.json'
    if sha256(export) != model['export_sha256']:
        raise ValueError(f'Pinned selected export changed: {arm}')
    return load_exported_model(reference, tokenizer, max_length=512), reference['metadata']
