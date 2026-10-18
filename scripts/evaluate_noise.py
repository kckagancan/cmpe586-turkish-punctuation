"""Evaluate saved validation noise variants against a verified clean reference."""

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
from turkish_punctuation.inference import load_published_model
from turkish_punctuation.metrics import PUNCTUATION_LABELS
from turkish_punctuation.preprocessing import encode_records, partition_records
from turkish_punctuation.robustness import evaluate_variant, metric_degradation, read_predictions
from turkish_punctuation.tokenization import LABEL_TO_ID

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "scripts/evaluate_noise.py",
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


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/noise-evaluation.json"))
    parser.add_argument(
        "--resume", action="store_true", help="Resume an interrupted identical run."
    )
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    executed_sources = {name: (ROOT / name).read_bytes() for name in SOURCE_PATHS}
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
    destination = (ROOT / config["output_directory"]).resolve()
    if not destination.is_relative_to(ROOT / "artifacts/evaluations"):
        raise ValueError("Results must be inside artifacts/evaluations/.")
    if destination.exists() and (not args.resume or (destination / "summary.json").exists()):
        raise FileExistsError(
            "Output exists. Use a new directory, or --resume for an interrupted run."
        )
    if args.resume and not destination.exists():
        raise FileNotFoundError("No interrupted run exists at the configured destination.")
    prepared = ROOT / config["prepared_directory"]
    preparation_path = prepared / "manifest.json"
    preparation = json.loads(preparation_path.read_text())
    clean_directory = ROOT / config["clean_reference_directory"]
    clean_report_path = clean_directory / "metrics.json"
    clean_report = json.loads(clean_report_path.read_text())
    baseline_path = clean_directory / "predictions.jsonl"
    tokenizer_config = json.loads(executed_sources["configs/tokenizer.json"])
    packages = {
        name: version(name)
        for name in ("torch", "transformers", "datasets", "scikit-learn", "numpy")
    }
    if packages != clean_report["packages"]:
        raise ValueError(
            "Runtime packages differ from clean reference; establish a matching clean run."
        )
    for key in ("batch_size", "device", "cpu_threads", "seed"):
        if config[key] != clean_report["run"][key]:
            raise ValueError(f"Inference setting differs from clean reference: {key}")
    if clean_report["run"]["split"] != "validation" or preparation["split"] != "validation":
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
            == clean_report["model"][key]
            == preparation["tokenizer_config"][key]
        ):
            raise ValueError(f"Clean/noisy/current model-tokenizer identities differ: {key}")
    if sha256(baseline_path) != clean_report["predictions_sha256"]:
        raise ValueError("Saved clean predictions differ from their recorded hash.")
    if preparation["clean_manifest_sha256"] != clean_report["prepared_manifest_sha256"]:
        raise ValueError("Noise variants and clean scores derive from different clean preparation.")
    source = json.loads(executed_sources["docs/data-manifest.json"])
    item = next(item for item in source["files"] if item["split"] == "validation")
    raw_path = ROOT / source["local_directory"] / item["filename"]
    if (
        sha256(raw_path) != item["sha256"]
        or raw_path.stat().st_size != item["bytes"]
        or item != preparation["raw_source"]
        or item["sha256"] != clean_report["validation_source_sha256"]
    ):
        raise ValueError("Validation source provenance differs between comparisons.")
    eligible, _ = partition_records(read_split(raw_path))
    clean_records = {row["row_id"]: row for row in eligible.to_dict(orient="records")}
    clean_predictions, clean_metrics = read_predictions(baseline_path, clean_records)
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
                    "model": tokenizer_config,
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
    with (destination / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another evaluator is already using this run directory.") from error
        model = load_published_model(tokenizer_config, root=ROOT)
        reports = {}
        for name in ["clean-anchor", *names]:
            output = destination / "conditions" / name
            if output.exists():
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
                )
                print(f"{name}: {completed}/{len(eligible)} records; {elapsed:.1f}s", flush=True)

            with TemporaryDirectory(prefix=f".{name}-", dir=output.parent) as temporary:
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
                staging.rename(output)
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
            "model": tokenizer_config,
            "identity": identity,
            "clean_reference_metrics": clean_metrics,
            "original_clean_reference_metrics": clean_report["metrics"],
            "clean_anchor": reports["clean-anchor"],
            "conditions": {name: reports[name] for name in names},
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Published checkpoint on fixed clean-v1 validation; one corruption seed in the pilot; not independent final-test evidence or a training-augmentation comparison.",
            "training_performed": False,
            "test_evaluated": False,
        }
        write_json(destination / "summary.json", summary)
        write_json(
            destination / "progress.json",
            {
                "status": "complete",
                "completed_conditions": list(reports),
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(
            f"Completed clean anchor and {len(names)} noise conditions: {destination}", flush=True
        )


if __name__ == "__main__":
    main()
