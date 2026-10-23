"""Freeze held-out protocol, model metadata and input-overlap row masks; no scoring."""

import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from inspect_data import fingerprint

from turkish_punctuation.data import read_split
from turkish_punctuation.noise import WordNoise
from turkish_punctuation.preprocessing import partition_records
from turkish_punctuation.training_io import sha256, utc_now, value_digest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def main():
    config_path = ROOT / 'configs/held-out-test.json'
    config = read(config_path)
    if (config['version'] != 'held-out-test-v1' or config['split'] != 'test'
        or config['phase'] != 'protocol_and_cohort_freeze_only'
        or config['execution']['training_enabled']
        or config['execution']['test_scoring_enabled_by_default']):
        raise ValueError('Expected a preparation-only held-out protocol.')
    if (ROOT / config['paths']['future_output_root']).exists():
        raise FileExistsError('Test output namespace exists; do not freeze after scoring.')
    matrix = config['run_matrix']
    noise = read(ROOT / config['noise']['recipe'])
    WordNoise(noise)  # Validate the unchanged operations; do not build/apply noise.
    expected = ['clean-anchor', *[
        f'{family}-{level}-seed{seed}' for seed in config['noise']['seeds']
        for level in config['noise']['levels'] for family in config['noise']['families']
    ], *config['noise']['full_stress_conditions']]
    ablation = ['clean-anchor', *read(ROOT / 'configs/ablation-case-loss.json')['evaluation_after_training']['conditions']]
    count = len(matrix['main_models']) * len(expected) + len(ablation)
    if (matrix['main_models'] != ['published', 'clean', 'augmented']
        or matrix['main_conditions'] != expected or len(expected) != 39
        or matrix['ablation_conditions'] != ablation or len(ablation) != 5
        or len(set(expected)) != len(expected)
        or count != matrix['primary_model_condition_evaluations'] or count != 122
        or matrix['score_sets_including_sensitivity'] != 2 * count):
        raise ValueError('Held-out matrix differs from the fixed validation-derived scope.')

    source_paths = [config['protocol'], 'configs/held-out-test.json',
                    config['data_manifest']['path'], config['clean_preprocessing']['config'],
                    config['noise']['recipe'], 'configs/tokenizer.json',
                    'scripts/freeze_heldout_protocol.py', 'scripts/inspect_data.py',
                    'src/turkish_punctuation/data.py', 'src/turkish_punctuation/preprocessing.py',
                    'src/turkish_punctuation/tokenization.py', 'src/turkish_punctuation/noise.py',
                    'src/turkish_punctuation/noise_audit.py', 'src/turkish_punctuation/metrics.py',
                    'src/turkish_punctuation/inference.py', 'src/turkish_punctuation/gpu_evaluation.py',
                    'src/turkish_punctuation/evaluation_model.py', 'src/turkish_punctuation/training_io.py']
    for item, key in ((config['data_manifest'], 'path'),
                      (config['clean_preprocessing'], 'config'),
                      (config['noise'], 'recipe')):
        expected_hash = item['sha256'] if key == 'path' else item[key + '_sha256']
        if sha256(ROOT / item[key]) != expected_hash:
            raise ValueError(f'Pinned input changed: {item[key]}')
    clean_settings = config['clean_preprocessing']
    clean_path = ROOT / clean_settings['existing_manifest']
    if sha256(clean_path) != clean_settings['existing_manifest_sha256']:
        raise ValueError('Existing clean-v1 manifest changed.')
    clean = read(clean_path)
    release = read(ROOT / config['data_manifest']['path'])
    if clean['source'] != release:
        raise ValueError('Clean preparation/released sources differ.')

    models, initializations = {}, set()
    for arm, model in config['models'].items():
        reference_path = ROOT / model['reference_config']
        summary_path = ROOT / model['reference_summary']
        if (sha256(reference_path) != model['reference_config_sha256']
            or sha256(summary_path) != model['reference_summary_sha256']):
            raise ValueError(f'Existing model reference changed: {arm}')
        source_paths.append(model['reference_config'])
        reference = read(summary_path)
        if arm == 'published':
            if read(reference_path)['model_tensors_sha256'] != model['model_tensors_sha256']:
                raise ValueError('Published model tensor identity differs.')
            models[arm] = {'metadata_verified': True, 'historical_test_exposure': 'unknown'}
            continue
        directory = ROOT / model['local_run_directory']
        run_path, summary_path = directory / 'run.json', directory / 'summary.json'
        export_path = ROOT / model['local_model_directory'] / 'export.json'
        for path, key in ((run_path, 'run_sha256'), (summary_path, 'summary_sha256'), (export_path, 'export_sha256')):
            if sha256(path) != model[key]:
                raise ValueError(f'Frozen training metadata changed: {arm}/{key}')
            source_paths.append(str(path.relative_to(ROOT)))
        run, summary, export = read(run_path), read(summary_path), read(export_path)
        selected = min(summary['history'], key=lambda row: (-row['validation']['metrics']['punctuation_macro']['f1'], row['epoch']))
        if (run['identity'] != model['training_identity']
            or value_digest(run['specification']) != model['training_identity']
            or summary['identity'] != model['training_identity']
            or selected['epoch'] != summary['best_epoch'] or selected['epoch'] != model['selected_epoch']
            or summary['global_step'] != model['selected_update']
            or summary['best_model'] != f"models/epoch-{model['selected_epoch']:02d}"
            or selected['model_export'] != export or export['identity'] != model['training_identity']
            or export['model_tensors_sha256'] != model['model_tensors_sha256']
            or export['files'] != model['export_file_sha256']
            or reference['identity']['trained_model']['model_tensors_sha256'] != model['model_tensors_sha256']
            or run['test_accessed'] or summary['test_accessed']):
            raise ValueError(f'Training selection/model identity differs: {arm}')
        if not all((export_path.parent / name).is_file() for name in export['files']):
            raise FileNotFoundError(f'Selected model export is incomplete: {arm}')
        initializations.add(run['specification']['initialization_identity'])
        models[arm] = {'metadata_verified': True, 'selected_epoch': selected['epoch'],
                       'selected_update': summary['global_step'], 'weights_rehashed': False}
    if len(initializations) != 1:
        raise ValueError('Fine-tuned models do not share initialization.')

    # Membership depends on original input words only; parsing also validates labels.
    records, raw_hashes, fingerprints = {}, {}, {}
    for item in release['files']:
        path = ROOT / release['local_directory'] / item['filename']
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise ValueError(f'Released input changed: {item["split"]}')
        frame = read_split(path)
        records[item['split']] = frame
        raw_hashes[item['split']] = item['sha256']
        fingerprints[item['split']] = {fingerprint(tokens) for tokens in frame['tokens']}
    primary, quarantine = partition_records(records['test'])
    primary_ids = primary['row_id'].tolist()
    quarantine_ids = quarantine['row_id'].tolist()
    if (primary_ids != clean['splits']['test']['eligible_row_ids']
        or quarantine_ids != clean['splits']['test']['quarantined_row_ids']
        or len(primary) != clean_settings['expected_eligible_records']
        or int(primary['tokens'].map(len).sum()) != clean_settings['expected_eligible_word_targets']
        or len(quarantine) != clean_settings['expected_quarantined_records']):
        raise ValueError('Test eligibility differs from the already prepared clean-v1 split.')
    overlaps = {'train': [], 'validation': []}
    for row_id, tokens in primary[['row_id', 'tokens']].itertuples(index=False, name=None):
        key = fingerprint(tokens)
        for split in overlaps:
            if key in fingerprints[split]:
                overlaps[split].append(row_id)
    excluded = set(overlaps['train']) | set(overlaps['validation'])
    sensitivity_ids = [i for i in primary_ids if i not in excluded]
    sensitivity = primary[primary['row_id'].isin(sensitivity_ids)]
    if not sensitivity_ids:
        raise ValueError('Empty sensitivity cohort.')
    cohorts = {'primary_row_ids': primary_ids, 'quarantined_row_ids': quarantine_ids,
               'sensitivity_row_ids': sensitivity_ids, 'excluded_exact_overlap_row_ids': sorted(excluded),
               'exact_train_overlap_row_ids': overlaps['train'],
               'exact_validation_overlap_row_ids': overlaps['validation']}
    aggregate = {'version': config['version'], 'split': 'test',
                 'primary': {'records': len(primary), 'word_targets': int(primary['tokens'].map(len).sum())},
                 'quarantined_records': len(quarantine),
                 'sensitivity': {'records': len(sensitivity), 'word_targets': int(sensitivity['tokens'].map(len).sum()),
                                 'excluded_union_records': len(excluded),
                                 'exact_train_overlap_records': len(overlaps['train']),
                                 'exact_validation_overlap_records': len(overlaps['validation']),
                                 'overlap_with_both_records': len(set(overlaps['train']) & set(overlaps['validation']))},
                 'cohort_row_ids_sha256': {key: value_digest(ids) for key, ids in cohorts.items()},
                 'raw_source_sha256': raw_hashes, 'model_condition_evaluations': count,
                 'score_sets_including_sensitivity': 2 * count,
                 'test_inputs_read_for_structural_audit': True, 'noise_generated': False,
                 'models_loaded': False, 'test_predictions_or_scores_computed': False}
    hashes = {name: sha256(ROOT / name) for name in source_paths}
    specification = {'config_sha256': sha256(config_path), 'source_sha256': hashes,
                     'clean_manifest_sha256': sha256(clean_path), 'raw_source_sha256': raw_hashes,
                     'cohorts_sha256': value_digest(cohorts), 'models': models}
    identity = value_digest(specification)
    destination = ROOT / config['paths']['freeze_directory']
    result = {'status': 'protocol_frozen_not_runtime_ready', 'identity': identity,
              'specification': specification, 'created_at_utc': utc_now(),
              'test_inputs_read': True, 'test_labels_parsed_by_shared_reader': True,
              'cohort_membership_uses_labels': False, 'models_loaded': False,
              'weights_rehashed': False, 'noise_generated': False, 'test_scored': False}
    if destination.exists():
        old = read(destination / 'freeze.json')
        if old['identity'] != identity:
            raise ValueError('Existing protocol freeze differs; preserve it and version changes.')
        for name, expected_hash in read(destination / 'complete.json')['files'].items():
            if sha256(destination / name) != expected_hash:
                raise ValueError(f'Committed freeze changed: {name}')
        print('Verified matching existing freeze; no model or scoring run.', flush=True)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.protocol-freeze-', dir=destination.parent) as temporary:
        staging = Path(temporary) / 'freeze'
        staging.mkdir()
        for name in hashes:
            target = staging / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            if sha256(target) != hashes[name]:
                raise OSError(f'Snapshot readback differs: {name}')
        for name, value in (('freeze.json', result), ('cohorts.json', cohorts), ('aggregate.json', aggregate)):
            (staging / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        files = {str(p.relative_to(staging)): sha256(p) for p in sorted(staging.rglob('*')) if p.is_file()}
        (staging / 'complete.json').write_text(json.dumps({'identity': identity, 'files': files}, indent=2) + '\n')
        staging.rename(destination)
    public = ROOT / 'docs/held-out-test-cohort-summary.json'
    if public.exists() and read(public) != aggregate:
        raise ValueError('Existing public aggregate differs; do not overwrite it.')
    public.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'freeze_identity': identity, **aggregate}, indent=2), flush=True)


if __name__ == '__main__':
    main()
