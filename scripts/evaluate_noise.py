"""Evaluate published or fine-tuned weights on fixed paired validation variants."""

import argparse
import fcntl
import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from datasets import Dataset, disable_progress_bars, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from turkish_punctuation.data import read_split
from turkish_punctuation.evaluation_model import load_exported_model, training_reference
from turkish_punctuation.gpu_evaluation import (
    condition_committed, configure_runtime, publish_condition, write_in_place,
)
from turkish_punctuation.inference import load_published_model
from turkish_punctuation.metrics import PUNCTUATION_LABELS
from turkish_punctuation.preprocessing import encode_records, partition_records
from turkish_punctuation.robustness import evaluate_variant, metric_degradation, read_predictions
from turkish_punctuation.tokenization import LABEL_TO_ID

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "scripts/evaluate_noise.py",
    "src/turkish_punctuation/evaluation_model.py",
    "src/turkish_punctuation/gpu_evaluation.py",
    "src/turkish_punctuation/training_io.py",
    "src/turkish_punctuation/robustness.py",
    "src/turkish_punctuation/inference.py",
    "src/turkish_punctuation/metrics.py",
    "src/turkish_punctuation/preprocessing.py",
    "src/turkish_punctuation/data.py",
    "src/turkish_punctuation/tokenization.py",
    "configs/tokenizer.json",
    "docs/data-manifest.json",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object, *, in_place: bool = False) -> None:
    if in_place:
        write_in_place(path, value)
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/noise-evaluation.json"))
    parser.add_argument("--input-root", type=Path, default=ROOT)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument(
        "--resume", action="store_true", help="Resume an interrupted identical run."
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    gpu_runtime = configure_runtime(config)
    executed_sources = {name: (ROOT / name).read_bytes() for name in SOURCE_PATHS}
    if gpu_runtime is not None:
        if args.calibration is None:
            raise ValueError("GPU evaluation requires the completed matched calibration receipt.")
        calibration = json.loads(args.calibration.read_text())
        if (
            calibration["status"] != "complete"
            or calibration["gpu_runtime"] != gpu_runtime
            or calibration["configs_sha256"][str(config_path.relative_to(ROOT))]
            != hashlib.sha256(config_bytes).hexdigest()
            or calibration["evaluation_source_sha256"] != {
                name: hashlib.sha256(content).hexdigest() for name, content in executed_sources.items()
            }
        ):
            raise ValueError("GPU runtime/configuration/source differs from calibration.")
    if (
        config["split"] != "validation"
        or config["punctuation_labels"] != list(PUNCTUATION_LABELS)
        or config["primary_metric"] != "punctuation_macro.f1"
        or config["zero_division"] != 0
    ):
        raise ValueError(
            "Expected validation and the existing fixed punctuation metric convention."
        )
    if (
        config["batch_order"] != "clean_length_ascending"
        or config["recompute_clean_anchor"] is not True
    ):
        raise ValueError("Use shared clean-length ordering with a recomputed clean anchor.")
    for key in (
        "batch_size",
        "cpu_threads",
        "progress_every_records",
        "error_examples_per_category_and_class",
    ):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    names = config["conditions"]
    if (
        not names
        or len(set(names)) != len(names)
        or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in names)
    ):
        raise ValueError("Choose distinct simple condition names from the prepared manifest.")
    output_root, input_root = args.output_root.resolve(), args.input_root.resolve()
    destination = (output_root / config["output_directory"]).resolve()
    if not destination.is_relative_to(output_root / "artifacts/evaluations"):
        raise ValueError("Results must be inside artifacts/evaluations/.")
    if destination.exists() and (
        not args.resume or (gpu_runtime is None and (destination / "summary.json").exists())
    ):
        raise FileExistsError(
            "Output exists. Use a new directory, or --resume for an interrupted run."
        )
    if args.resume and not destination.exists():
        raise FileNotFoundError("No interrupted run exists at the configured destination.")
    prepared = input_root / config["prepared_directory"]
    preparation_path = prepared / "manifest.json"
    preparation = json.loads(preparation_path.read_text())
    tokenizer_config = json.loads(executed_sources["configs/tokenizer.json"])
    packages = {
        name: version(name)
        for name in ("torch", "transformers", "datasets", "scikit-learn", "numpy")
    }
    if gpu_runtime is not None and packages != calibration["packages"]:
        raise ValueError("Evaluation packages changed after GPU calibration.")
    trained = None
    comparison = None
    if "trained_model" in config:
        trained_settings = dict(config["trained_model"])
        if args.storage_root is not None:
            trained_settings["run_directory"] = str(
                args.storage_root.resolve() / config["storage_run_directory"]
            )
        trained = training_reference(ROOT, trained_settings)
        clean_report_path = trained["report_path"]
        clean_report = trained["report"]
        baseline_path = trained["prediction_path"]
        reference_tokenizer = trained["tokenizer_config"]
        validation_source_sha256 = trained["validation_source_sha256"]
        model_provenance = trained["metadata"]
        comparison_path = output_root / config["published_comparison_directory"] / "summary.json"
        comparison = json.loads(comparison_path.read_text())
        if (
            packages != comparison["identity"]["packages"]
            or sha256(preparation_path) != comparison["identity"]["noise_manifest_sha256"]
            or names != comparison["run"]["conditions"]
        ):
            raise ValueError("Published comparison has different runtime or noise preparation.")
        for key in ("batch_size", "device", "cpu_threads", "seed", "batch_order"):
            if config[key] != comparison["run"][key]:
                raise ValueError(f"Inference setting differs from published comparison: {key}")
        if gpu_runtime != comparison["identity"].get("gpu_runtime"):
            raise ValueError("Cross-model GPU hardware or numerical policy differs.")
    else:
        clean_directory = input_root / config["clean_reference_directory"]
        clean_report_path = clean_directory / "metrics.json"
        clean_report = json.loads(clean_report_path.read_text())
        baseline_path = clean_directory / "predictions.jsonl"
        reference_tokenizer = clean_report["model"]
        validation_source_sha256 = clean_report["validation_source_sha256"]
        model_provenance = dict(tokenizer_config)
        if gpu_runtime is None and packages != clean_report["packages"]:
            raise ValueError("Runtime packages differ from clean reference.")
        for key in ("batch_size", "device", "cpu_threads", "seed"):
            if gpu_runtime is None and config[key] != clean_report["run"][key]:
                raise ValueError(f"Inference setting differs from clean reference: {key}")
        if clean_report["run"]["split"] != "validation":
            raise ValueError("Clean reference must belong to validation.")
        if preparation["clean_manifest_sha256"] != clean_report["prepared_manifest_sha256"]:
            raise ValueError("Noise variants and clean scores have different clean preparation.")
    if preparation["split"] != "validation":
        raise ValueError("All comparison inputs must belong to validation.")
    if tokenizer_config["label_to_id"] != LABEL_TO_ID:
        raise ValueError("Tokenizer label mapping differs from the implemented mapping.")
    for key in (
        "model_id",
        "revision",
        "max_input_length",
        "label_to_id",
        "alignment",
        "ignore_index",
    ):
        if (
            not tokenizer_config[key]
            == reference_tokenizer[key]
            == preparation["tokenizer_config"][key]
        ):
            raise ValueError(f"Clean/noisy/current model-tokenizer identities differ: {key}")
    if sha256(baseline_path) != clean_report["predictions_sha256"]:
        raise ValueError("Saved clean predictions differ from their recorded hash.")
    source = json.loads(executed_sources["docs/data-manifest.json"])
    item = next(item for item in source["files"] if item["split"] == "validation")
    raw_path = input_root / source["local_directory"] / item["filename"]
    if (
        sha256(raw_path) != item["sha256"]
        or raw_path.stat().st_size != item["bytes"]
        or item != preparation["raw_source"]
        or item["sha256"] != validation_source_sha256
    ):
        raise ValueError("Validation source provenance differs between comparisons.")
    eligible, _ = partition_records(read_split(raw_path))
    clean_records = {row["row_id"]: row for row in eligible.to_dict(orient="records")}
    reference_records = clean_records
    if trained is not None:
        # Training validation stores its batching order; pair by source IDs and
        # verify full unique coverage before reading its historical predictions.
        with baseline_path.open(encoding="utf-8") as stream:
            reference_ids = [json.loads(line)["row_id"] for line in stream]
        if len(reference_ids) != len(set(reference_ids)) or set(reference_ids) != set(clean_records):
            raise ValueError("Training validation does not cover each eligible source row once.")
        reference_records = {row_id: clean_records[row_id] for row_id in reference_ids}
    clean_predictions, clean_metrics = read_predictions(baseline_path, reference_records)
    if clean_metrics != clean_report["metrics"]:
        raise ValueError("Recomputed clean metrics differ from the archived reference.")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["model_id"],
        revision=tokenizer_config["revision"],
        cache_dir=str(ROOT / tokenizer_config["cache_directory"]),
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
        token=False,
    )
    if (
        not isinstance(tokenizer, PreTrainedTokenizerFast)
        or tokenizer.model_max_length != tokenizer_config["max_input_length"]
    ):
        raise ValueError("Expected the same pinned fast tokenizer and position limit.")
    disable_progress_bars()
    variants, input_hashes = {}, {}
    for name in names:
        audit = preparation["conditions"][name]
        path = prepared / "conditions" / name
        if (
            not audit["changes"]["invariants_passed"]
            or not audit["tokenization"]["all_first_subword_targets_preserved"]
        ):
            raise ValueError(f"{name}: data audit did not pass.")
        hashes = {}
        for file, expected in audit["output_sha256"].items():
            hashes[file] = sha256(path / file)
            if hashes[file] != expected:
                raise ValueError(f"{name}/{file}: differs from the preparation manifest.")
        if json.loads((path / "audit.json").read_text()) != audit:
            raise ValueError(f"{name}: local audit differs from the manifest.")
        hashes["audit.json"] = sha256(path / "audit.json")
        with (path / "records.jsonl").open(encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        frame = pd.DataFrame(records)
        if frame["row_id"].tolist() != list(clean_records):
            raise ValueError(f"{name}: source row IDs/order changed.")
        for row in records:
            original = clean_records[row["row_id"]]
            if row["tags"] != original["tags"] or len(row["tokens"]) != len(original["tokens"]):
                raise ValueError(f"{name}: word counts or targets changed.")
        dataset = load_from_disk(str(path / "dataset"))
        if not isinstance(dataset, Dataset):
            raise ValueError(f"{name}: expected one encoded Dataset.")
        regenerated = encode_records(
            frame, tokenizer, max_length=tokenizer_config["max_input_length"]
        )
        if set(dataset.column_names) != set(regenerated.column_names):
            raise ValueError(f"{name}: encoded columns differ from re-tokenization.")
        for column in dataset.column_names:
            if dataset[column][:] != regenerated[column][:]:
                raise ValueError(
                    f"{name}: stored {column} differs from re-tokenized words/targets."
                )
        for file in sorted((path / "dataset").rglob("*")):
            if file.is_file():
                hashes[str(file.relative_to(path))] = sha256(file)
        input_hashes[name] = hashes
        variants[name] = (dataset, {row["row_id"]: row for row in records})
        print(
            f"Preflight passed: {name}; {len(dataset)} rows, exact re-tokenization/targets.",
            flush=True,
        )
    clean_dataset = load_from_disk(str(prepared / "clean/dataset"))
    if not isinstance(clean_dataset, Dataset):
        raise ValueError("Expected the prepared clean validation Dataset.")
    regenerated = encode_records(
        eligible, tokenizer, max_length=tokenizer_config["max_input_length"]
    )
    if set(clean_dataset.column_names) != set(regenerated.column_names):
        raise ValueError("Clean-anchor columns differ from re-tokenization.")
    for column in clean_dataset.column_names:
        if clean_dataset[column][:] != regenerated[column][:]:
            raise ValueError(f"Clean-anchor {column} differs from re-tokenization.")
    input_hashes["clean-anchor"] = {
        str(file.relative_to(prepared)): sha256(file)
        for file in sorted((prepared / "clean/dataset").rglob("*"))
        if file.is_file()
    }
    lengths = [len(ids) for ids in clean_dataset["input_ids"]]
    original_ids = list(clean_records)
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], original_ids[i]))
    ordered_ids = [original_ids[i] for i in order]
    clean_records = {row_id: clean_records[row_id] for row_id in ordered_ids}
    variants["clean-anchor"] = (clean_dataset, clean_records)
    batching = {
        "mode": config["batch_order"],
        "row_ids_sha256": hashlib.sha256(
            json.dumps(ordered_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "padded_tokens": {},
    }
    for name, (dataset, records) in list(variants.items()):
        sizes = [len(ids) for ids in dataset["input_ids"]]

        def padded_work(values: list[int]) -> int:
            size = config["batch_size"]
            return sum(
                max(values[i : i + size]) * len(values[i : i + size])
                for i in range(0, len(values), size)
            )

        batching["padded_tokens"][name] = {
            "source_order": padded_work(sizes),
            "evaluation_order": padded_work([sizes[i] for i in order]),
        }
        variants[name] = (
            dataset.select(order),
            {row_id: records[row_id] for row_id in ordered_ids},
        )
    if comparison is not None and batching != comparison["identity"]["batching"]:
        raise ValueError("Published comparison uses different row order or batch membership.")
    identity = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in executed_sources.items()
        },
        "noise_manifest_sha256": sha256(preparation_path),
        "clean_metrics_sha256": sha256(clean_report_path),
        "clean_predictions_sha256": sha256(baseline_path),
        "condition_input_sha256": input_hashes,
        "batching": batching,
        "packages": packages,
    }
    if gpu_runtime is not None:
        identity["gpu_runtime"] = gpu_runtime
    if trained is not None:
        identity["trained_model"] = model_provenance
        identity["published_comparison_summary_sha256"] = sha256(comparison_path)
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".noise-evaluation-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "run"
            (staging / "source_snapshot").mkdir(parents=True)
            for name, content in executed_sources.items():
                snapshot = staging / "source_snapshot" / name
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(content)
            (staging / "source_snapshot/noise-evaluation-config.json").write_bytes(config_bytes)
            write_json(
                staging / "run.json",
                {
                    "identity": identity,
                    "run": config,
                    "model": model_provenance,
                    "original_clean_reference_metrics": clean_metrics,
                    "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "python": platform.python_version(),
                    "git_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                    ).strip(),
                    "git_dirty": bool(
                        subprocess.check_output(
                            ["git", "status", "--porcelain"], cwd=ROOT, text=True
                        ).strip()
                    ),
                    "training_performed": False,
                    "test_evaluated": False,
                },
            )
            staging.rename(destination)
    run = json.loads((destination / "run.json").read_text())
    if run["identity"] != identity:
        raise ValueError(
            "Resume inputs/code/settings differ from the interrupted run; use a new run."
        )
    identity_digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if gpu_runtime is not None and (destination / "summary.json").exists():
        try:
            completed_summary = json.loads((destination / "summary.json").read_text())
        except json.JSONDecodeError:
            completed_summary = None
        if completed_summary is not None:
            if completed_summary["identity"] != identity:
                raise ValueError("Completed summary identity changed.")
            for name in ["clean-anchor", *names]:
                folder = destination / "conditions" / name
                if not condition_committed(folder, identity_digest):
                    raise ValueError("Completed summary refers to an uncommitted condition.")
                saved = json.loads((folder / "metrics.json").read_text())
                expected = completed_summary["clean_anchor"] if name == "clean-anchor" else completed_summary["conditions"][name]
                if saved != expected:
                    raise ValueError("Completed summary/condition report differs.")
            print(f"Verified already completed GPU run: {destination}", flush=True)
            return
    with (destination / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another evaluator is already using this run directory.") from error
        model = (
            load_exported_model(trained, tokenizer, max_length=tokenizer_config["max_input_length"])
            if trained is not None
            else load_published_model(tokenizer_config, root=ROOT)
        )
        if gpu_runtime is not None:
            from turkish_punctuation.training_io import model_digest
            if model.config._attn_implementation != "sdpa" or model.dtype != torch.float32:
                raise ValueError("Expected the shared FP32 SDPA/math inference implementation.")
            actual = model_digest(model)
            expected = config.get("model_tensors_sha256") or config["trained_model"]["model_tensors_sha256"]
            if actual != expected:
                raise ValueError("GPU evaluation loaded a different model tensor identity.")
        reports = {}
        for name in ["clean-anchor", *names]:
            output = destination / "conditions" / name
            committed = output.exists() and (
                gpu_runtime is None or condition_committed(output, identity_digest)
            )
            if committed:
                saved = json.loads((output / "metrics.json").read_text())
                if (
                    sha256(output / "predictions.jsonl") != saved["predictions_sha256"]
                    or sha256(output / "error_examples.json") != saved["error_examples_sha256"]
                    or saved["condition"] != name
                ):
                    raise ValueError(f"Completed output differs from saved hashes: {name}")
                reports[name] = saved
                if name == "clean-anchor":
                    clean_predictions, clean_metrics = read_predictions(
                        output / "predictions.jsonl", clean_records
                    )
                    if clean_metrics != saved["metrics"]:
                        raise ValueError("Saved clean-anchor metrics do not reproduce.")
                print(f"Resuming: retained completed condition {name}.", flush=True)
                continue
            if output.exists() and not args.resume:
                raise FileExistsError(f"Uncommitted condition exists; use --resume: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            print(f"Evaluating {name} on {config['device']}.", flush=True)

            def progress(completed: int, elapsed: float) -> None:
                write_json(
                    destination / "progress.json",
                    {
                        "status": "running",
                        "condition": name,
                        "completed_records": completed,
                        "total_records": len(eligible),
                        "condition_elapsed_seconds": elapsed,
                        "completed_conditions": list(reports),
                        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    },
                    in_place=gpu_runtime is not None,
                )
                print(f"{name}: {completed}/{len(eligible)} records; {elapsed:.1f}s", flush=True)

            with TemporaryDirectory(
                prefix=f".{name}-", dir=output.parent if gpu_runtime is None else None
            ) as temporary:
                staging = Path(temporary) / "result"
                staging.mkdir()
                dataset, noisy_records = variants[name]
                result = evaluate_variant(
                    dataset,
                    tokenizer,
                    model,
                    clean_records=clean_records,
                    noisy_records=noisy_records,
                    clean_predictions=clean_predictions,
                    config=config,
                    max_length=tokenizer_config["max_input_length"],
                    prediction_path=staging / "predictions.jsonl",
                    progress=progress,
                )
                write_json(staging / "error_examples.json", result.pop("error_examples"))
                result.update(
                    {
                        "condition": name,
                        "input_audit": preparation["conditions"][name]
                        if name != "clean-anchor"
                        else {
                            "tokenization": preparation["clean_tokenization"],
                            "changes": {"changed_words": 0, "changed_word_percent": 0.0},
                        },
                        "degradation_pp": metric_degradation(clean_metrics, result["metrics"]),
                        "predictions_sha256": sha256(staging / "predictions.jsonl"),
                        "error_examples_sha256": sha256(staging / "error_examples.json"),
                        "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                write_json(staging / "metrics.json", result)
                if gpu_runtime is None:
                    staging.rename(output)
                else:
                    publish_condition(staging, output, identity_digest)
                reports[name] = result
                if name == "clean-anchor":
                    clean_predictions, clean_metrics = read_predictions(
                        output / "predictions.jsonl", clean_records
                    )
                    if clean_metrics != result["metrics"]:
                        raise ValueError("New clean-anchor metrics do not reproduce.")
                print(
                    f"Completed {name}: macro-F1={result['metrics']['punctuation_macro']['f1']:.6f}",
                    flush=True,
                )
        summary = {
            "run": config,
            "model": model_provenance,
            "identity": identity,
            "clean_reference_metrics": clean_metrics,
            "original_clean_reference_metrics": clean_report["metrics"],
            "clean_anchor": reports["clean-anchor"],
            "conditions": {name: reports[name] for name in names},
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": (
                "Fixed clean-v1 validation; corruption realizations are identified by condition "
                "names. No held-out test or additional training seeds. Degradation uses this "
                "run's recomputed clean control. Historical clean predictions are diagnostics."
            ),
            "training_performed": False,
            "test_evaluated": False,
        }
        write_json(destination / "summary.json", summary, in_place=gpu_runtime is not None)
        write_json(
            destination / "progress.json",
            {
                "status": "complete",
                "completed_conditions": list(reports),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            in_place=gpu_runtime is not None,
        )
        print(
            f"Completed clean anchor and {len(names)} noise conditions: {destination}", flush=True
        )


if __name__ == "__main__":
    main()
