"""Runtime bindings and raw-target checks for the separate held-out runner."""

import json
from collections import Counter

import torch
from datasets import load_from_disk

from turkish_punctuation.gpu_evaluation import configure_runtime
from turkish_punctuation.heldout import (
    condition_directory,
    execution_sources,
    packages,
    read,
    runtime_settings,
    verify_commit,
)
from turkish_punctuation.metrics import compute_metrics
from turkish_punctuation.robustness import metric_degradation, read_predictions
from turkish_punctuation.training_io import model_digest, sha256, value_digest


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def runtime_binding(root, input_root, plan):
    path = input_root / plan['inference']['reference_runtime_receipt']
    if sha256(path) != plan['inference']['reference_runtime_receipt_sha256']:
        raise ValueError('Pinned GPU reference receipt changed.')
    reference = read(path)
    runtime = configure_runtime(runtime_settings(plan))
    installed = packages()
    if runtime != reference['gpu_runtime'] or installed != reference['packages']:
        raise ValueError('GPU runtime/packages differ from the frozen T4 comparison.')
    for name, digest in reference['evaluation_source_sha256'].items():
        # This is a new launcher. All shared numerical implementations remain exact.
        if name != 'scripts/evaluate_noise.py' and sha256(root / name) != digest:
            raise ValueError(f'Calibrated shared implementation changed: {name}')
    return {'gpu_runtime': runtime, 'packages': installed,
            'reference_runtime_receipt_sha256': sha256(path)}


def check_model(model, expected):
    if model.dtype != torch.float32 or model.config._attn_implementation != 'sdpa':
        raise ValueError('Expected the frozen FP32 SDPA/math model execution.')
    actual = model_digest(model)
    if actual != expected['model_tensors_sha256']:
        raise ValueError('Loaded model tensors differ from the fixed checkpoint.')
    if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise ValueError('Model has non-finite tensors.')
    return actual


def prepared_binding(input_root, execution):
    directory = input_root / execution['prepared_directory']
    return {'manifest_sha256': sha256(directory / 'manifest.json'),
            'commit_sha256': sha256(directory / 'complete.json')}


def runtime_specification(root, input_root, plan, execution, freeze, runtime):
    return {'version': 'held-out-runtime-v1', 'freeze_identity': freeze['identity'],
            'execution_config_sha256': sha256(root / 'configs/held-out-execution.json'),
            'inputs': prepared_binding(input_root, execution),
            'source_sha256': execution_sources(root, freeze), 'runtime': runtime,
            'models': plan['models'], 'inference': plan['inference']}


def verify_readiness(root, input_root, path, plan, execution, freeze, runtime):
    marker = verify_commit(path)
    receipt = read(path / 'readiness.json')
    expected = runtime_specification(root, input_root, plan, execution, freeze, runtime)
    if (receipt['status'] != 'ready_for_test_scoring'
        or receipt['specification'] != expected
        or receipt['identity'] != value_digest(expected)
        or marker['identity'] != receipt['identity']
        or receipt['optimizer_updates'] != 0 or receipt['metrics_computed']
        or receipt['test_predictions_computed']):
        raise ValueError('Completed readiness does not bind these inputs/models/runtime/sources.')
    for name, digest in expected['source_sha256'].items():
        if sha256(path / 'source_snapshot' / name) != digest:
            raise ValueError(f'Readiness source snapshot changed: {name}')
    return receipt


def order_and_records(prepared, clean):
    dataset = load_from_disk(str(prepared / 'clean/dataset'))
    lengths, ids = [len(x) for x in dataset['input_ids']], list(dataset['row_id'])
    order = sorted(range(len(dataset)), key=lambda i: (lengths[i], ids[i]))
    records = {row['row_id']: row for row in clean.to_dict(orient='records')}
    return order, {ids[i]: records[ids[i]] for i in order}


def variant(prepared, name, order, clean_records):
    directory = condition_directory(prepared, name)
    dataset = load_from_disk(str(directory / 'dataset')).select(order)
    if name == 'clean-anchor':
        return dataset, clean_records
    rows = {row['row_id']: row for row in (
        json.loads(line) for line in (directory / 'records.jsonl').read_text().splitlines())}
    return dataset, {row_id: rows[row_id] for row_id in clean_records}


def score_cohorts(path, originals, cohorts):
    predictions, primary = read_predictions(path, originals)
    selected = set(cohorts['sensitivity_row_ids'])
    reference, predicted = [], []
    for row_id, row in predictions.items():
        if row_id in selected:
            reference.extend(row['reference_ids'])
            predicted.extend(row['prediction_ids'])
    if len(selected) != sum(row_id in selected for row_id in predictions):
        raise ValueError('Sensitivity row coverage differs from the fixed mask.')
    return predictions, {'primary': primary, 'sensitivity': compute_metrics(reference, predicted)}


def diagnostics(plan, name, originals, noisy, baseline, current, sensitivity_ids):
    """Fixed source-order diagnostic samples; no selection by score magnitude."""
    examples, counts, per_class = [], Counter(), {}
    groups = Counter()
    selected = set(sensitivity_ids)
    for row_id in sorted(originals):
        clean, changed = originals[row_id], noisy[row_id]
        edits = [i for i, (old, new) in enumerate(zip(clean['tokens'], changed['tokens'], strict=True)) if old != new]
        for i, (true, old, new) in enumerate(zip(current[row_id]['reference_ids'],
            baseline[row_id]['prediction_ids'], current[row_id]['prediction_ids'], strict=True)):
            category = ('both_correct' if true == old == new else 'new_error' if true == old
                        else 'recovered_error' if true == new else 'changed_error' if old != new
                        else 'unchanged_error')
            label = clean['tags'][i]
            counts[category] += 1
            per_class.setdefault(label, Counter())[category] += 1
            key = (category, label)
            if (name in plan['qualitative_analysis']['conditions']
                and category in plan['qualitative_analysis']['categories']
                and label in plan['qualitative_analysis']['classes']
                and groups[key] < plan['qualitative_analysis']['examples_per_category_and_class']):
                examples.append({'row_id': row_id, 'word_index': i, 'reference_class': label,
                    'category': category, 'clean_words': clean['tokens'], 'noisy_words': changed['tokens'],
                    'clean_prediction_ids': baseline[row_id]['prediction_ids'],
                    'noisy_prediction_ids': current[row_id]['prediction_ids'],
                    'input_changed_word_indices': edits, 'error_word_itself_changed': i in edits,
                    'unchanged_word_with_edited_context': i not in edits and bool(edits),
                    'in_sensitivity_cohort': row_id in selected})
                groups[key] += 1
    return {'outcomes': dict(counts), 'outcomes_per_reference_class': per_class}, examples


def condition_report(identity, arm, name, metrics, clean_metrics, audit, paired):
    return {'identity': identity, 'arm': arm, 'condition': name, 'split': 'test',
            'cohorts': metrics,
            'own_clean_to_noisy_drop_pp': {cohort: metric_degradation(clean_metrics[cohort], value)
                                           for cohort, value in metrics.items()},
            'realized_input_changes': {'primary': audit['changes'],
                                       'sensitivity': audit['sensitivity_changes']},
            'paired_outcomes_primary': paired, 'model_selection_performed': False}
