"""Verify actual checkpoints and validation-only GPU probes before transfer scores."""

import argparse
import gc
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from datasets import Dataset, disable_progress_bars

from turkish_punctuation.data import read_split
from turkish_punctuation.heldout import load_model
from turkish_punctuation.heldout_runtime import check_model, runtime_binding
from turkish_punctuation.inference import iter_predictions
from turkish_punctuation.noise_audit import encode_and_audit
from turkish_punctuation.preprocessing import partition_records
from turkish_punctuation.training_io import (
    directory_lock,
    file_hashes,
    sha256,
    utc_now,
    value_digest,
    write_json,
)
from turkish_punctuation.transfer_audit import audit_transfer_changes
from turkish_punctuation.transfer_execution import (
    read,
    runtime_specification,
    verify_inputs,
    verify_readiness,
)
from turkish_punctuation.transfer_noise import (
    CONTROL,
    FAMILIES,
    apply_transfer,
    build_transfer_plans,
)
from turkish_punctuation.transfer_preparation import transfer_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    disable_progress_bars()
    context = verify_inputs(ROOT, args.input_root, retokenize=True)
    config, execution, original, _, _, manifest, _, noise = context
    runtime = runtime_binding(ROOT, args.input_root, original)
    specification = runtime_specification(ROOT, args.input_root, context, runtime)
    identity = value_digest(specification)
    release = read(ROOT / original["data_manifest"]["path"])
    source = next(x for x in release["files"] if x["split"] == "validation")
    raw = args.input_root / release["local_directory"] / source["filename"]
    if sha256(raw) != source["sha256"] or raw.stat().st_size != source["bytes"]:
        raise ValueError("Pinned validation CSV differs")
    records, quarantine = partition_records(read_split(raw, nrows=execution["validation"]["source_prefix_rows"]))
    validation_binding = {"source_sha256": source["sha256"], "row_ids": records["row_id"].tolist(),
        "quarantined_ids": quarantine["row_id"].tolist(), "probe_seed": execution["validation"]["probe_corruption_seed"]}
    if args.output.exists():
        old = verify_readiness(ROOT, args.input_root, args.output, context, runtime)
        if old["validation_inputs"] != validation_binding:
            raise ValueError("Readiness validation inputs differ")
        print("Matching completed transfer readiness reused; no repeated probes", flush=True)
        return
    tokenizer = transfer_tokenizer(ROOT, config)
    clean, _ = encode_and_audit(records, tokenizer, max_length=512, batch_size=256)
    lengths, ids = [len(x) for x in clean["input_ids"]], clean["row_id"][:]
    order = sorted(range(len(clean)), key=lambda i: (lengths[i], ids[i]))
    batch = config["inference"]["batch_size"]
    batches = (len(order) + batch - 1) // batch
    selected = {i * (batches - 1) // 15 for i in range(16)}
    selected.add(order.index(max(range(len(clean)), key=lambda i: lengths[i])) // batch)
    indices = [i for b in sorted(selected) for i in order[b * batch:(b + 1) * batch]]
    probes = {"validation_clean": clean.select(indices)}
    plans = build_transfer_plans(records, noise, split="validation", seed=validation_binding["probe_seed"])
    input_audits = {}
    for family in (*FAMILIES, CONTROL):
        noisy, edits = apply_transfer(records, noise, plans, family)
        changes = audit_transfer_changes(records, noisy, edits, noise, family, plans)
        encoded, token_audit = encode_and_audit(noisy, tokenizer, max_length=512, batch_size=256)
        probes["validation_" + family] = encoded.select(indices)
        input_audits[family] = {"changes": changes, "tokenization": token_audit}
    maximum = max(manifest["max_input_length"], max(len(x) for p in probes.values() for x in p["input_ids"]))
    if maximum > 512:
        raise ValueError("Planned padded shape exceeds the frozen limit")
    padded = clean.select(order[-batch:]).to_dict()
    for key in ("input_ids", "attention_mask", "token_type_ids", "labels"):
        if key in padded:
            pad = tokenizer.pad_token_id if key == "input_ids" else -100 if key == "labels" else 0
            padded[key] = [row + [pad] * (maximum - len(row)) for row in padded[key]]
    probes["validation_only_padded_maximum_shape"] = Dataset.from_dict(padded)
    measurements = {}
    with directory_lock(args.output.parent / "transfer-readiness-lock"):
        if args.output.exists():
            raise FileExistsError("Concurrent readiness destination appeared")
        for arm in execution["model_order"]:
            model, _ = load_model(ROOT, original, arm, args.storage_root, tokenizer)
            tensor_hash = check_model(model, original["models"][arm])
            for _ in iter_predictions(clean.select(indices[:batch]), tokenizer, model,
                                      max_length=512, batch_size=batch, device="cuda"):
                pass
            measurements[arm] = {"model_tensors_sha256": tensor_hash, "probes": {}}
            for name, dataset in probes.items():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = perf_counter()
                count = sum(1 for _ in iter_predictions(dataset, tokenizer, model,
                    max_length=512, batch_size=batch, device="cuda"))
                torch.cuda.synchronize()
                if count != len(dataset):
                    raise ValueError("Validation-only probe coverage differs")
                measurements[arm]["probes"][name] = {"rows": count, "seconds": perf_counter()-started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                print(f"TRANSFER_VALIDATION_PROBE_OK {arm} {name}: {count} rows", flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        if specification != runtime_specification(ROOT, args.input_root, context, runtime):
            raise ValueError("Runtime/source binding changed during readiness")
        result = {"identity": identity, "status": "ready_for_transfer_scoring", "specification": specification,
            "validation_inputs": validation_binding, "validation_input_audits": input_audits,
            "measurements": measurements, "planned_maximum_padded_length": maximum,
            "optimizer_updates": 0, "metrics_computed": False, "test_predictions_computed": False,
            "probe_data_split": "validation", "created_at_utc": utc_now()}
        with TemporaryDirectory(prefix="transfer-ready-") as temporary:
            staging = Path(temporary)
            for name in specification["source_sha256"]:
                target = staging / "source_snapshot" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            write_json(staging / "readiness.json", result)
            write_json(staging / "complete.json", {"identity": identity, "files": file_hashes(staging)})
            args.output.mkdir(parents=True)
            for path in staging.rglob("*"):
                if path.is_file() and path.name != "complete.json":
                    target = args.output / path.relative_to(staging)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
            shutil.copyfile(staging / "complete.json", args.output / "complete.json")
        verify_readiness(ROOT, args.input_root, args.output, context, runtime)
    print("TRANSFER_READINESS_COMPLETE: no test scores, no optimizer updates", flush=True)


if __name__ == "__main__":
    main()
