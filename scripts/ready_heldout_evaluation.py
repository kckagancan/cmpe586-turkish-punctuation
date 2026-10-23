"""Separate held-out readiness: byte/target audits and validation-only GPU probes."""

import argparse
import gc
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from datasets import Dataset, disable_progress_bars, load_from_disk

from turkish_punctuation.heldout import load_model, load_tokenizer, verify_inputs
from turkish_punctuation.heldout_runtime import (
    check_model,
    read,
    runtime_binding,
    runtime_specification,
    verify_readiness,
    write,
)
from turkish_punctuation.inference import iter_predictions
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
    parser.add_argument('--validation-input-root', type=Path, required=True)
    parser.add_argument('--storage-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    disable_progress_bars()
    plan, execution, freeze, _, manifest, _ = verify_inputs(ROOT, args.input_root, retokenize=True)
    runtime = runtime_binding(ROOT, args.input_root, plan)
    specification = runtime_specification(ROOT, args.input_root, plan, execution, freeze, runtime)
    identity = value_digest(specification)
    validation = args.validation_input_root / execution['validation_probe_directory']
    if sha256(validation / 'manifest.json') != execution['validation_probe_manifest_sha256']:
        raise ValueError('Validation technical-probe manifest changed.')
    clean = load_from_disk(str(validation / 'clean/dataset'))
    # Re-encode raw validation inputs from the previously audited original CSV.
    # Its source bytes and the prepared dataset must both agree with that receipt.
    from turkish_punctuation.data import read_split
    from turkish_punctuation.noise_audit import encode_and_audit
    from turkish_punctuation.preprocessing import partition_records
    receipt = read(validation / 'manifest.json')
    release = read(ROOT / plan['data_manifest']['path'])
    item = next(row for row in release['files'] if row['split'] == 'validation')
    raw = args.validation_input_root / release['local_directory'] / item['filename']
    if sha256(raw) != item['sha256']:
        raise ValueError('Original validation probe source changed.')
    records, _ = partition_records(read_split(raw))
    tokenizer = load_tokenizer(ROOT, plan)
    encoded, _ = encode_and_audit(records, tokenizer, max_length=512, batch_size=256)
    if set(encoded.column_names) != set(clean.column_names) or any(
        encoded[key][:] != clean[key][:] for key in clean.column_names
    ) or receipt['split'] != 'validation':
        raise ValueError('Technical-probe inputs differ from original validation encoding.')
    validation_binding = {'manifest_sha256': sha256(validation / 'manifest.json'),
                          'dataset_sha256': file_hashes(validation / 'clean/dataset'),
                          'raw_sha256': sha256(raw)}
    if args.output.exists():
        old = verify_readiness(ROOT, args.input_root, args.output, plan, execution, freeze, runtime)
        if old['validation_inputs'] != validation_binding:
            raise ValueError('Validation probe input bytes changed since readiness.')
        print('Matching completed readiness verified; GPU probes not repeated.', flush=True)
        return
    lengths, ids = [len(row) for row in clean['input_ids']], list(clean['row_id'])
    order = sorted(range(len(clean)), key=lambda i: (lengths[i], ids[i]))
    batch = plan['inference']['batch_size']
    batches = (len(order) + batch - 1) // batch
    selected = {i * (batches - 1) // 15 for i in range(16)}
    selected.add(order.index(max(range(len(clean)), key=lambda i: lengths[i])) // batch)
    indices = [i for b in sorted(selected) for i in order[b * batch:(b + 1) * batch]]
    probes = {'validation_fixed_clean_length_batches': clean.select(indices)}
    maximum = max(max(lengths), manifest['max_planned_input_length'])
    if maximum > 512:
        raise ValueError('Planned maximum shape exceeds the frozen position limit.')
    padded = clean.select(order[-batch:]).to_dict()
    for key in ('input_ids', 'attention_mask', 'token_type_ids', 'labels'):
        if key in padded:
            padding = tokenizer.pad_token_id if key == 'input_ids' else -100 if key == 'labels' else 0
            padded[key] = [row + [padding] * (maximum - len(row)) for row in padded[key]]
    probes['validation_only_padded_to_planned_maximum_shape'] = Dataset.from_dict(padded)
    measurements = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with directory_lock(args.output.parent / 'readiness-lock'):
        if args.output.exists():
            raise FileExistsError('A readiness receipt appeared concurrently; inspect it.')
        for arm in execution['model_order']:
            model, _ = load_model(ROOT, plan, arm, args.storage_root, tokenizer)
            tensor_hash = check_model(model, plan['models'][arm])
            for _ in iter_predictions(clean.select(indices[:batch]), tokenizer, model,
                max_length=512, batch_size=batch, device='cuda'):
                pass
            measurements[arm] = {'model_tensors_sha256': tensor_hash, 'probes': {}}
            for name, dataset in probes.items():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = perf_counter()
                count = sum(1 for _ in iter_predictions(dataset, tokenizer, model,
                    max_length=512, batch_size=batch, device='cuda'))
                torch.cuda.synchronize()
                if count != len(dataset):
                    raise ValueError('Validation-only probe coverage differs.')
                measurements[arm]['probes'][name] = {'rows': count, 'seconds': perf_counter() - started,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                print(f'VALIDATION_PROBE_OK {arm} {name}: {count} rows', flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        if runtime_specification(ROOT, args.input_root, plan, execution, freeze, runtime) != specification:
            raise ValueError('Readiness sources/inputs changed during the probes.')
        result = {'identity': identity, 'status': 'ready_for_test_scoring', 'specification': specification,
            'validation_inputs': validation_binding, 'measurements': measurements,
            'planned_maximum_padded_length': maximum, 'optimizer_updates': 0, 'metrics_computed': False,
            'test_predictions_computed': False, 'test_inputs_structurally_audited': True,
            'probe_data_split': 'validation', 'created_at_utc': utc_now()}
        with TemporaryDirectory(prefix='.held-out-ready-', dir=args.output.parent) as temporary:
            staging = Path(temporary) / 'ready'
            staging.mkdir()
            for name in specification['source_sha256']:
                target = staging / 'source_snapshot' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            write(staging / 'readiness.json', result)
            write(staging / 'complete.json', {'identity': identity, 'files': file_hashes(staging)})
            # Publish the commit marker last, including on a mounted Drive.
            args.output.mkdir()
            for path in staging.rglob('*'):
                if path.is_file() and path != staging / 'complete.json':
                    target = args.output / path.relative_to(staging)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            shutil.copyfile(staging / 'complete.json', args.output / 'complete.json')
        verify_readiness(ROOT, args.input_root, args.output, plan, execution, freeze, runtime)
    print('HELD_OUT_READINESS_COMPLETE: zero training updates; no test predictions/scores.', flush=True)


if __name__ == '__main__':
    main()
