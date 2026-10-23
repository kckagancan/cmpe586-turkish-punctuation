"""Prepare the frozen held-out noise views once, without loading prediction models."""

import json
import platform
import shutil
import time
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import disable_progress_bars, load_from_disk

from turkish_punctuation.heldout import INPUT_SOURCES, load_plan, load_test, load_tokenizer
from turkish_punctuation.noise import FAMILIES, WordNoise, apply_noise, build_plans, stable_digest
from turkish_punctuation.noise_audit import audit_changes, encode_and_audit, validate_source
from turkish_punctuation.training_io import (
    directory_lock,
    file_hashes,
    sha256,
    utc_now,
    value_digest,
)

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def jsonl(path, rows):
    with path.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')


def main():
    plan, execution, freeze, cohorts = load_plan(ROOT)
    destination = ROOT / execution['prepared_directory']
    from turkish_punctuation.heldout import verify_commit
    if destination.exists():
        verify_commit(destination)
        old = json.loads((destination / 'manifest.json').read_text())
        if (old['freeze_identity'] != freeze['identity']
            or old['execution_config_sha256'] != sha256(ROOT / 'configs/held-out-execution.json')):
            raise ValueError('Existing held-out views belong to different inputs/settings.')
        for name, digest in old['specification']['source_sha256'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'Existing preparation source changed: {name}')
        print('Verified matching completed held-out input preparation; views not regenerated.')
        return
    clean, source = load_test(ROOT, ROOT, plan, cohorts)
    validate_source(clean)
    tokenizer = load_tokenizer(ROOT, plan)
    recipe = json.loads((ROOT / plan['noise']['recipe']).read_text())
    noise = WordNoise(recipe)
    source_hashes = {name: sha256(ROOT / name) for name in INPUT_SOURCES}
    specification = {'freeze_identity': freeze['identity'], 'source_sha256': source_hashes,
        'raw_source_sha256': source['sha256'], 'noise_recipe_sha256': sha256(ROOT / plan['noise']['recipe']),
        'tokenizer_config_sha256': sha256(ROOT / 'configs/tokenizer.json'),
        'packages': {name: version(name) for name in ('transformers', 'tokenizers', 'datasets', 'pandas')},
        'python': platform.python_version()}
    manifest = {'version': 'held-out-inputs-v1', 'split': 'test', 'specification': specification,
        'freeze_identity': freeze['identity'], 'raw_source': source,
        'execution_config_sha256': sha256(ROOT / 'configs/held-out-execution.json'),
        'conditions': {}, 'started_at_utc': utc_now(), 'model_weights_loaded': False,
        'model_evaluation_performed': False, 'training_performed': False, 'test_text_read': True}
    sensitivity_ids = set(cohorts['sensitivity_row_ids'])
    sensitive_clean = clean[clean['row_id'].isin(sensitivity_ids)]
    started = time.monotonic()
    disable_progress_bars()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with directory_lock(ROOT / 'artifacts/evaluation-preparation/held-out-test-v1/input-lock'):
        if destination.exists():
            raise FileExistsError('Another preparer published this destination; inspect it.')
        with TemporaryDirectory(prefix='.held-out-inputs-', dir=destination.parent) as temporary:
            staging = Path(temporary) / 'prepared'
            staging.mkdir()
            for name in source_hashes:
                target = staging / 'source_snapshot' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            encoded, token_audit = encode_and_audit(clean, tokenizer, max_length=512, batch_size=256)
            saved_clean = load_from_disk(str(ROOT / 'data/processed/clean-v1/dataset/test'))
            if set(encoded.column_names) != set(saved_clean.column_names) or any(
                encoded[name][:] != saved_clean[name][:] for name in encoded.column_names
            ):
                raise ValueError('Clean test encoding differs from the original clean-v1 preparation.')
            directory = staging / 'clean'
            encoded.save_to_disk(str(directory / 'dataset'))
            changes, _ = audit_changes(clean, clean, [], example_seed=0, condition_name='clean-anchor', example_count=0)
            sensitivity, _ = audit_changes(sensitive_clean, sensitive_clean, [], example_seed=0,
                                          condition_name='clean-anchor', example_count=0)
            manifest['conditions']['clean-anchor'] = {'changes': changes, 'sensitivity_changes': sensitivity,
                'tokenization': token_audit, 'families': [], 'seed': None, 'level': 'clean'}
            print('Prepared clean-anchor: original clean-v1 test inputs/targets reproduced.', flush=True)
            del encoded, saved_clean

            def prepare(name, families, plans, rate, level, seed):
                noisy, edits = apply_noise(clean, noise, plans, families=families, coverage_percent=rate)
                changes, examples = audit_changes(clean, noisy, edits, example_seed=seed or 0,
                    condition_name=name, example_count=recipe['audit']['examples_per_condition'])
                for family in families:
                    pool = plans.positions[family]
                    count = len(pool) if rate is None else plans.total_words * rate // 100
                    actual = changes['families'][family]
                    if (actual['selected_words'] != count
                        or actual['selected_positions_sha256'] != stable_digest(sorted(pool[:count])).hex()
                        or (len(families) == 1 and changes['changed_words'] != count)):
                        raise ValueError(f'Original noise budget/mask differs: {name}/{family}')
                encoded, token_audit = encode_and_audit(noisy, tokenizer, max_length=512, batch_size=256)
                # Filter the same edits, not a new plan with a reduced denominator.
                sensitivity, _ = audit_changes(sensitive_clean, noisy[noisy['row_id'].isin(sensitivity_ids)],
                    [edit for edit in edits if edit['row_id'] in sensitivity_ids], example_seed=seed or 0,
                    condition_name=name, example_count=0)
                output = staging / 'conditions' / name
                output.mkdir(parents=True)
                jsonl(output / 'records.jsonl', noisy.to_dict(orient='records'))
                jsonl(output / 'edits.jsonl', edits)
                write(output / 'examples.json', examples)
                encoded.save_to_disk(str(output / 'dataset'))
                audit = {'families': families, 'seed': seed, 'level': level, 'coverage_percent': rate,
                         'changes': changes, 'sensitivity_changes': sensitivity, 'tokenization': token_audit}
                write(output / 'audit.json', audit)
                manifest['conditions'][name] = audit
                print(f"Prepared {name}: changed={changes['changed_words']}; "
                      f"max_subwords={token_audit['length_including_special_tokens']['max']}", flush=True)

            for seed in plan['noise']['seeds']:
                plans = build_plans(clean, noise, split='test', seed=seed)
                previous = {family: set() for family in FAMILIES}
                for level, rate in plan['noise']['levels'].items():
                    for family in (*FAMILIES, 'combined'):
                        families = recipe['combined']['apply_order'] if family == 'combined' else [family]
                        name = f'{family}-{level}-seed{seed}'
                        prepare(name, families, plans, rate, level, seed)
                        if family != 'combined':
                            mask = set(plans.positions[family][:plans.total_words * rate // 100])
                            if not previous[family].issubset(mask):
                                raise ValueError('Nested severity prefix changed.')
                            previous[family] = mask
                        else:
                            for component in FAMILIES:
                                single = manifest['conditions'][f'{component}-{level}-seed{seed}']
                                if (single['changes']['families'][component]['selected_positions_sha256']
                                    != manifest['conditions'][name]['changes']['families'][component]['selected_positions_sha256']):
                                    raise ValueError('Combined/single component plans differ.')
            for item in recipe['full_stress_conditions']:
                prepare(item['name'], [item['family']], plans, None, 'full', None)
            if list(manifest['conditions']) != plan['run_matrix']['main_conditions']:
                raise ValueError('Prepared test condition membership/order differs from the frozen plan.')
            manifest['max_planned_input_length'] = max(
                row['tokenization']['length_including_special_tokens']['max'] for row in manifest['conditions'].values())
            manifest['elapsed_seconds'] = time.monotonic() - started
            manifest['completed_at_utc'] = utc_now()
            write(staging / 'manifest.json', manifest)
            load_plan(ROOT)  # Detect edits to any frozen implementation/config during work.
            if source_hashes != {name: sha256(ROOT / name) for name in INPUT_SOURCES}:
                raise ValueError('Preparation source changed during execution.')
            write(staging / 'complete.json', {'identity': value_digest(specification),
                                            'files': file_hashes(staging)})
            staging.rename(destination)
    print(f'HELD_OUT_INPUTS_PREPARED: 39 views; no models/scores; {destination}', flush=True)


if __name__ == '__main__':
    main()
