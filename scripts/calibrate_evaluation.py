"""Time fixed representative batches without producing validation scores or updates."""

import argparse
import gc
import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
from datasets import load_from_disk
from transformers import AutoTokenizer

from evaluate_noise import ROOT, SOURCE_PATHS
from turkish_punctuation.evaluation_model import load_exported_model, training_reference
from turkish_punctuation.gpu_evaluation import configure_runtime, write_in_place
from turkish_punctuation.inference import iter_predictions, load_published_model
from turkish_punctuation.training_io import model_digest, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads((ROOT / "configs/robustness-grid.json").read_text())
    configs = {name: json.loads((ROOT / name).read_text()) for name in matrix["configuration_order"]}
    first = next(iter(configs.values()))
    runtime = configure_runtime(first)
    paired = {}
    for name,config in configs.items():
        if "trained_model" in config:
            run = json.loads((args.storage_root/config["storage_run_directory"]/"run.json").read_text())
            paired[config["trained_model"]["arm"]] = run
    clean,augmented = paired["clean"]["specification"],paired["augmented"]["specification"]
    for key in ("seed","initialization_identity","initialization_tensors_sha256","environment"):
        if clean[key] != augmented[key]:
            raise ValueError(f"Training pair differs in {key}.")
    for key in ("optimization","selection","model","data","runtime"):
        if clean["configuration"][key] != augmented["configuration"][key]:
            raise ValueError(f"Training recipe differs in {key}.")
    if paired["clean"]["budget"] != paired["augmented"]["budget"] or augmented["paired_clean_run"]["identity"] != paired["clean"]["identity"]:
        raise ValueError("Training budgets or paired clean identity differ.")
    token_config = json.loads((ROOT / "configs/tokenizer.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(
        token_config["model_id"], revision=token_config["revision"],
        cache_dir=str(ROOT / token_config["cache_directory"]), local_files_only=False,
        use_fast=True, trust_remote_code=False, token=False,
    )
    prepared = args.input_root / matrix["prepared_directory"]
    clean = load_from_disk(str(prepared / "clean/dataset"))
    lengths = [len(ids) for ids in clean["input_ids"]]
    ids = list(clean["row_id"])
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], ids[i]))
    batch = matrix["batch_size"]
    batches = (len(order) + batch - 1) // batch
    selected = sorted({i * (batches - 1) // 15 for i in range(16)})
    indices = [order[i] for b in selected for i in range(b * batch, min((b + 1) * batch, len(order)))]
    probes = {
        "clean-anchor": clean,
        "combined-heavy-seed586": load_from_disk(str(prepared / "conditions/combined-heavy-seed586/dataset")),
        "diacritic_loss_full": load_from_disk(str(prepared / "conditions/diacritic_loss_full/dataset")),
    }
    padded = {}
    for name in ["clean-anchor", *first["conditions"]]:
        dataset = clean if name == "clean-anchor" else load_from_disk(str(prepared / "conditions" / name / "dataset"))
        sizes = [len(value) for value in dataset["input_ids"]]
        padded[name] = sum(max(sizes[i] for i in order[b:b+batch]) * len(order[b:b+batch])
                           for b in range(0, len(order), batch))
    results = {}
    for config_name, config in configs.items():
        if configure_runtime(config) != runtime or config["conditions"] != first["conditions"]:
            raise ValueError("Matrix configurations have different numerical policies or inputs.")
        if "trained_model" in config:
            settings = dict(config["trained_model"])
            settings["run_directory"] = str(args.storage_root / config["storage_run_directory"])
            reference = training_reference(ROOT, settings)
            model = load_exported_model(reference, tokenizer, max_length=512)
            expected = settings["model_tensors_sha256"]
        else:
            model = load_published_model(token_config, root=ROOT, allow_download=True)
            expected = config["model_tensors_sha256"]
        if model_digest(model) != expected or model.dtype != torch.float32 or model.config._attn_implementation != "sdpa":
            raise ValueError("Calibration loaded different model tensors or attention/dtype.")
        for _ in iter_predictions(clean.select(indices[:batch]), tokenizer, model,
                                  max_length=512, batch_size=batch, device="cuda"):
            pass
        measurements = {}
        for name, dataset in probes.items():
            subset = dataset.select(indices)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = perf_counter()
            count = sum(1 for _ in iter_predictions(subset, tokenizer, model,
                        max_length=512, batch_size=batch, device="cuda"))
            torch.cuda.synchronize()
            if count != len(indices):
                raise ValueError("Calibration prediction coverage changed.")
            sample_sizes = [len(value) for value in subset["input_ids"]]
            work = sum(max(sample_sizes[b:b+batch]) * len(sample_sizes[b:b+batch])
                       for b in range(0, len(sample_sizes), batch))
            measurements[name] = {"seconds": perf_counter()-start, "records": count,
                "padded_tokens": work, "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
        # An advisory padded-work estimate, not a timing guarantee or model score.
        rates = [m["seconds"] / m["padded_tokens"] for m in measurements.values()]
        estimate = sum(padded.values()) * max(rates) / 60
        results[config["run_name"]] = {"model_tensors_sha256": expected,
            "measurements": measurements, "advisory_grid_minutes": estimate}
        print(f"{config['run_name']}: advisory grid {estimate:.1f} min; "
              f"peak {max(m['peak_allocated_bytes'] for m in measurements.values())/2**30:.3f} GiB", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    result = {"status": "complete", "gpu_runtime": runtime,
        "paired_training_initialization_recipe_budget_verified":True,
        "packages": {key:version(key) for key in ("torch","transformers","datasets","scikit-learn","numpy")},
        "configs_sha256": {name:sha256(ROOT/name) for name in configs},
        "evaluation_source_sha256": {name:sha256(ROOT/name) for name in SOURCE_PATHS},
        "calibration_source_sha256": sha256(Path(__file__)),
        "sample_policy": "16 evenly spaced whole clean-length batches; same source IDs for all probes/models",
        "noise_manifest_sha256": sha256(prepared/"manifest.json"),
        "models": results, "optimizer_updates": 0, "metrics_computed": False, "test_accessed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_in_place(args.output, result)
    print("GPU_EVALUATION_CALIBRATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
