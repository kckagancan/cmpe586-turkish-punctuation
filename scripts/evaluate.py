"""Evaluate the published reference on the fixed clean validation subset."""

import hashlib
import json
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from turkish_punctuation.data import read_split
from turkish_punctuation.inference import iter_predictions, load_published_model
from turkish_punctuation.metrics import PUNCTUATION_LABELS, compute_metrics
from turkish_punctuation.preprocessing import partition_records
from turkish_punctuation.tokenization import LABEL_TO_ID

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "scripts/evaluate.py",
    "src/turkish_punctuation/inference.py",
    "src/turkish_punctuation/metrics.py",
    "src/turkish_punctuation/preprocessing.py",
    "src/turkish_punctuation/data.py",
    "src/turkish_punctuation/tokenization.py",
    "configs/evaluation.json",
    "configs/tokenizer.json",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    executed_sources = {name: (ROOT / name).read_bytes() for name in SOURCE_PATHS}
    config = json.loads((ROOT / "configs/evaluation.json").read_text())
    tokenizer_config = json.loads((ROOT / "configs/tokenizer.json").read_text())
    if config["split"] != "validation" or config["condition"] != "clean":
        raise ValueError("This initial evaluator supports clean validation only.")
    if (
        config["punctuation_labels"] != list(PUNCTUATION_LABELS)
        or config["primary_metric"] != "punctuation_macro.f1"
        or config["zero_division"] != 0
    ):
        raise ValueError("Evaluation config differs from the implemented metric convention.")
    if config["cpu_threads"] < 1 or config["progress_every_records"] < 1:
        raise ValueError("CPU threads and progress intervals must be positive.")
    destination = ROOT / config["output_directory"]
    if destination.exists():
        raise FileExistsError(
            f"Evaluation already exists; use a new run name/output: {destination}"
        )
    prepared = ROOT / config["prepared_directory"]
    preparation = json.loads((prepared / "manifest.json").read_text())
    for key in (
        "model_id",
        "revision",
        "max_input_length",
        "label_to_id",
        "alignment",
        "ignore_index",
    ):
        if preparation["tokenizer"][key] != tokenizer_config[key]:
            raise ValueError(f"Prepared tokenizer differs in {key}.")
    source = json.loads((ROOT / "docs/data-manifest.json").read_text())
    if preparation["source"] != source:
        raise ValueError("Prepared data and current source manifest differ.")
    item = next(item for item in source["files"] if item["split"] == "validation")
    raw_path = ROOT / source["local_directory"] / item["filename"]
    if sha256(raw_path) != item["sha256"] or raw_path.stat().st_size != item["bytes"]:
        raise ValueError("Validation source differs from the pinned release.")
    eligible, quarantine = partition_records(read_split(raw_path))
    expected_ids = preparation["splits"]["validation"]["eligible_row_ids"]
    if eligible["row_id"].tolist() != expected_ids:
        raise ValueError("Current eligible validation IDs differ from the prepared manifest.")
    dataset = load_from_disk(str(prepared / "dataset/validation"))
    if not isinstance(dataset, Dataset) or list(dataset["row_id"]) != expected_ids:
        raise ValueError("Saved validation dataset has different source IDs or ordering.")
    originals = {
        row_id: (tokens, tags)
        for row_id, tokens, tags in eligible[["row_id", "tokens", "tags"]].itertuples(
            index=False, name=None
        )
    }
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
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("A fast tokenizer is required.")
    model = load_published_model(tokenizer_config, root=ROOT)
    reference, prediction, errors = [], [], []
    error_counts, sampled_rows = Counter(), set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(dataset)} clean validation records on {config['device']}.", flush=True)
    with TemporaryDirectory(prefix=".evaluate-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        for name, content in executed_sources.items():
            snapshot_path = staging / "source_snapshot" / name
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(content)
        start = perf_counter()
        completed = 0
        with (staging / "predictions.jsonl").open("w", encoding="utf-8") as stream:
            for result in iter_predictions(
                dataset,
                tokenizer,
                model,
                max_length=tokenizer_config["max_input_length"],
                batch_size=config["batch_size"],
                device=config["device"],
            ):
                row_id = result["row_id"]
                if row_id != expected_ids[completed]:
                    raise RuntimeError("Prediction order changed.")
                words, tags = originals[row_id]
                if result["reference_ids"] != [LABEL_TO_ID[tag] for tag in tags]:
                    raise RuntimeError(f"Row {row_id}: prepared labels differ from raw targets.")
                if len(result["prediction_ids"]) != len(words):
                    raise RuntimeError(f"Row {row_id}: prediction/word counts differ.")
                stream.write(json.dumps(result) + "\n")
                reference.extend(result["reference_ids"])
                prediction.extend(result["prediction_ids"])
                for position, (true, pred) in enumerate(
                    zip(result["reference_ids"], result["prediction_ids"], strict=True)
                ):
                    if (
                        true != pred
                        and error_counts[true] < config["error_examples_per_reference_class"]
                        and (row_id, true) not in sampled_rows
                    ):
                        errors.append(
                            {
                                "row_id": row_id,
                                "word_index": position,
                                "words": words,
                                "reference_tags": tags,
                                "prediction_ids": result["prediction_ids"],
                                "reference_id_at_error": true,
                                "prediction_id_at_error": pred,
                            }
                        )
                        error_counts[true] += 1
                        sampled_rows.add((row_id, true))
                completed += 1
                if completed % config["progress_every_records"] == 0:
                    print(f"Predicted {completed}/{len(dataset)} records.", flush=True)
        if completed != len(dataset):
            raise RuntimeError("Not all eligible validation records were evaluated.")
        metrics = compute_metrics(reference, prediction)
        expected_support = preparation["splits"]["validation"]["eligible"]["label_counts"]
        if {
            label: values["support"] for label, values in metrics["per_class"].items()
        } != expected_support:
            raise RuntimeError("Evaluated class supports differ from the prepared manifest.")
        report = {
            "run": config,
            "model": tokenizer_config,
            "metrics": metrics,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "evaluated_records": completed,
            "quarantined_records": len(quarantine),
            "prediction_loop_seconds_including_io": perf_counter() - start,
            "prepared_manifest_sha256": sha256(prepared / "manifest.json"),
            "validation_source_sha256": item["sha256"],
            "code_sha256": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in executed_sources.items()
                if name.endswith(".py")
            },
            "configuration_sha256": {
                name: hashlib.sha256(content).hexdigest()
                for name, content in executed_sources.items()
                if name.endswith(".json")
            },
            "source_snapshot_directory": "source_snapshot",
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()
            ),
            "python": platform.python_version(),
            "packages": {
                name: version(name)
                for name in ("torch", "transformers", "datasets", "scikit-learn")
            },
            "label_to_id": LABEL_TO_ID,
            "predictions_sha256": sha256(staging / "predictions.jsonl"),
            "error_sampling": "First mismatches in source order, at most one per row/reference class.",
            "scope": "Published reference on clean-v1 validation, not the full published test set.",
            "upstream_selection": "The released checkpoint may have used this validation split for model selection.",
            "training_performed": False,
            "test_evaluated": False,
        }
        (staging / "metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (staging / "error_examples.json").write_text(
            json.dumps(errors, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved evaluation: {destination}", flush=True)


if __name__ == "__main__":
    main()
