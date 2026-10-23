"""Explicit, resumable held-out inference; historical validation commands stay intact."""

import argparse
import gc
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from datasets import disable_progress_bars

from turkish_punctuation.gpu_evaluation import (
    condition_committed,
    publish_condition,
    write_in_place,
)
from turkish_punctuation.heldout import load_model, load_tokenizer, verify_commit, verify_inputs
from turkish_punctuation.heldout_runtime import (
    check_model,
    condition_report,
    diagnostics,
    order_and_records,
    read,
    runtime_binding,
    runtime_specification,
    score_cohorts,
    variant,
    verify_readiness,
    write,
)
from turkish_punctuation.inference import iter_predictions
from turkish_punctuation.tokenization import LABEL_TO_ID
from turkish_punctuation.training_io import (
    directory_lock,
    file_hashes,
    sha256,
    utc_now,
    value_digest,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--storage-root', type=Path, required=True)
    parser.add_argument('--readiness', type=Path, required=True)
    parser.add_argument('--allow-test-scoring', action='store_true')
    args = parser.parse_args()
    if not args.allow_test_scoring:
        raise ValueError('Held-out scoring is off. An explicit --allow-test-scoring is required.')
    disable_progress_bars()
    plan, execution, freeze, cohorts, manifest, clean = verify_inputs(ROOT, args.input_root)
    runtime = runtime_binding(ROOT, args.input_root, plan)
    readiness = verify_readiness(ROOT, args.input_root, args.readiness, plan, execution, freeze, runtime)
    prepared = args.input_root / execution['prepared_directory']
    order, originals = order_and_records(prepared, clean)
    batch = plan['inference']['batch_size']
    batching = {'source_row_ids': list(originals), 'batch_size': batch, 'padded_tokens': {}}
    for name in plan['run_matrix']['main_conditions']:
        dataset, _ = variant(prepared, name, order, originals)
        lengths = [len(row) for row in dataset['input_ids']]
        batching['padded_tokens'][name] = sum(
            max(lengths[i:i + batch]) * len(lengths[i:i + batch]) for i in range(0, len(lengths), batch))
    specification = {'version': 'held-out-evaluation-v1', 'readiness_identity': readiness['identity'],
        'readiness_commit_sha256': sha256(args.readiness / 'complete.json'),
        'runtime_binding': runtime_specification(ROOT, args.input_root, plan, execution, freeze, runtime),
        'batching_sha256': value_digest(batching)}
    identity = value_digest(specification)
    output = args.storage_root / execution['output_directory']
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(ROOT, plan)
    completed = 0
    with directory_lock(output.parent / 'held-out-test-v1-writer-lock'):
        provenance = output / 'provenance'
        if provenance.exists():
            marker = verify_commit(provenance)
            run = read(provenance / 'run.json')
            if marker['identity'] != identity or run['identity'] != identity or run['specification'] != specification:
                raise ValueError('Existing run identity differs; preserve it and create a new version.')
        else:
            with TemporaryDirectory(prefix='held-out-provenance-') as temporary:
                staging = Path(temporary)
                write(staging / 'run.json', {'identity': identity, 'specification': specification,
                                           'started_at_utc': utc_now(), 'training_performed': False})
                write(staging / 'batching.json', batching)
                for name in specification['runtime_binding']['source_sha256']:
                    target = staging / 'source_snapshot' / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, target)
                write(staging / 'complete.json', {'identity': identity, 'files': file_hashes(staging)})
                provenance.mkdir()
                for path in staging.rglob('*'):
                    if path.is_file() and path.name != 'complete.json':
                        target = provenance / path.relative_to(staging)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
                shutil.copyfile(staging / 'complete.json', provenance / 'complete.json')
            verify_commit(provenance)
        # Recheck every actual checkpoint before exposing the first test score.
        # This is file/tensor verification only; it never runs test inputs.
        for arm in execution['model_order']:
            model, _ = load_model(ROOT, plan, arm, args.storage_root, tokenizer)
            check_model(model, plan['models'][arm])
            del model
            gc.collect()
            print(f'CHECKPOINT_REVERIFIED {arm}', flush=True)
        for arm in execution['model_order']:
            names = (plan['run_matrix']['ablation_conditions'] if arm == plan['run_matrix']['ablation_model']
                     else plan['run_matrix']['main_conditions'])
            model = None
            baseline, clean_metrics = None, None
            try:
                for name in names:
                    destination = output / 'models' / arm / 'conditions' / name
                    if (destination / 'complete.json').exists():
                        read(destination / 'complete.json')  # A damaged marker is never a retry authority.
                    if condition_committed(destination, identity):
                        current, metrics = score_cohorts(destination / 'predictions.jsonl', originals, cohorts)
                        saved = read(destination / 'metrics.json')
                        if (saved['cohorts'] != metrics or saved['identity'] != identity
                            or saved['arm'] != arm or saved['condition'] != name):
                            raise ValueError('Committed condition metrics/targets/identity differ.')
                        print(f'REUSED {arm} {name}: verified targets and both cohorts', flush=True)
                    else:
                        if model is None:
                            model, _ = load_model(ROOT, plan, arm, args.storage_root, tokenizer)
                            check_model(model, plan['models'][arm])
                        dataset, noisy = variant(prepared, name, order, originals)
                        with TemporaryDirectory(prefix='held-out-condition-') as temporary:
                            staging = Path(temporary)
                            started = perf_counter()
                            write_in_place(output / 'progress.json', {'identity': identity, 'status': 'running',
                                'arm': arm, 'condition': name, 'completed_conditions': completed,
                                'planned_conditions': 122, 'updated_at_utc': utc_now()})
                            with (staging / 'predictions.jsonl').open('x') as stream:
                                count = 0
                                for row, expected in zip(iter_predictions(dataset, tokenizer, model,
                                    max_length=512, batch_size=batch, device='cuda'), originals, strict=True):
                                    targets = [LABEL_TO_ID[tag] for tag in originals[expected]['tags']]
                                    if row['row_id'] != expected or row['reference_ids'] != targets:
                                        raise ValueError('Test inference IDs/first-subword targets changed.')
                                    stream.write(json.dumps(row, separators=(',', ':')) + '\n')
                                    count += 1
                                    if count % execution['progress_every_records'] == 0 or count == len(originals):
                                        print(f'PROGRESS {arm} {name}: {count}/{len(originals)} rows', flush=True)
                            current, metrics = score_cohorts(staging / 'predictions.jsonl', originals, cohorts)
                            paired, examples = diagnostics(plan, name, originals, noisy,
                                current if name == 'clean-anchor' else baseline, current, cohorts['sensitivity_row_ids'])
                            report = condition_report(identity, arm, name, metrics,
                                metrics if name == 'clean-anchor' else clean_metrics, manifest['conditions'][name], paired)
                            report['inference_loop_seconds_including_io'] = perf_counter() - started
                            report['predictions_sha256'] = sha256(staging / 'predictions.jsonl')
                            write(staging / 'metrics.json', report)
                            write(staging / 'error_examples.json', {'selection': plan['qualitative_analysis'],
                                                                  'examples': examples})
                            publish_condition(staging, destination, identity)
                        print(f'COMMITTED {arm} {name}', flush=True)
                    if name == 'clean-anchor':
                        baseline, clean_metrics = current, metrics
                    completed += 1
                    write_in_place(output / 'progress.json', {'identity': identity, 'status': 'running',
                        'completed_conditions': completed, 'planned_conditions': 122,
                        'last_committed': [arm, name], 'updated_at_utc': utc_now()})
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
        if completed != plan['run_matrix']['primary_model_condition_evaluations']:
            raise ValueError('The frozen evaluation matrix is incomplete.')
        if runtime_specification(ROOT, args.input_root, plan, execution, freeze, runtime) != specification['runtime_binding']:
            raise ValueError('Inputs/sources/runtime binding changed during inference.')
        write_in_place(output / 'progress.json', {'identity': identity, 'status': 'inference_complete',
            'completed_conditions': completed, 'planned_conditions': 122, 'updated_at_utc': utc_now(),
            'next_action': 'Run analyze_heldout.py against saved predictions; no repeated inference.'})
    print('HELD_OUT_INFERENCE_COMPLETE: 122 conditions; saved predictions ready for raw-target audit.', flush=True)


if __name__ == '__main__':
    main()
