"""Persist and verify training inputs/initialization without any optimizer updates."""

import argparse
import json
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from turkish_punctuation.training import training_budget, validate_training_config
from turkish_punctuation.training_data import load_training_data
from turkish_punctuation.training_io import (
    directory_lock,
    file_hashes,
    sha256,
    utc_now,
    write_json,
)
from turkish_punctuation.training_model import (
    load_shared_initialization,
    load_training_tokenizers,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/prepare_training.py",
    "scripts/train.py",
    "src/turkish_punctuation/training.py",
    "src/turkish_punctuation/training_data.py",
    "src/turkish_punctuation/training_io.py",
    "src/turkish_punctuation/training_model.py",
    "src/turkish_punctuation/preprocessing.py",
    "src/turkish_punctuation/tokenization.py",
    "src/turkish_punctuation/data.py",
    "src/turkish_punctuation/metrics.py",
    "configs/training.json",
    "configs/tokenizer.json",
    "requirements/colab.constraints.txt",
)


def copy_verified_tree(source: Path, destination: Path) -> None:
    """Publish a byte-identical copy; never replace a different existing directory."""
    expected = file_hashes(source)
    if not expected:
        raise ValueError(f"Refusing to publish an empty directory: {source}")
    if destination.exists():
        if file_hashes(destination) != expected:
            raise ValueError(f"Existing files differ; no overwrite performed: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "complete"
        shutil.copytree(source, staging)
        if file_hashes(staging) != expected:
            raise ValueError(f"Copied files failed verification: {destination}")
        staging.rename(destination)
    if file_hashes(destination) != expected:
        raise ValueError(f"Published files failed verification: {destination}")


def verify_input_cache(cache: Path) -> dict[str, str]:
    expected = json.loads((cache / "files.json").read_text())
    if not expected or file_hashes(cache / "prepared") != expected:
        raise ValueError("Persistent prepared-input cache is incomplete or changed.")
    return expected


def prepare(args: argparse.Namespace) -> dict:
    storage = args.storage_root.resolve()
    if storage.is_relative_to(Path("/content/drive")):
        allowed = Path("/content/drive/MyDrive/cmpe586-turkish-punctuation")
        if storage != allowed or not Path("/content/drive/MyDrive").is_dir():
            raise ValueError("Use the mounted Drive's designated project directory only.")
    config_path = ROOT / "configs/training.json"
    config = json.loads(config_path.read_text())
    validate_training_config(config)
    if args.seed not in config["seeds"]["initial_training_seeds"]:
        raise ValueError("Preparation supports the declared initial training seeds only.")
    source = json.loads((ROOT / config["model"]["source_config"]).read_text())
    cache = storage / "inputs/clean-v1"
    initializations = storage / "initializations" / config["version"]
    calibration_relative = Path(f"training-v1/calibration-clean-seed-{args.seed}-v1")
    calibration_path = storage / calibration_relative
    report_directory = storage / "preparation" / config["version"] / f"seed-{args.seed}"
    with directory_lock(storage / "preparation"):
        if args.reuse_storage_root is not None:
            previous = args.reuse_storage_root.resolve()
            migrations = (
                Path("inputs/clean-v1"),
                Path(f"initializations/{config['version']}/seed-{args.seed}"),
                calibration_relative,
            )
            if previous != storage:
                for relative in migrations:
                    original = previous / relative
                    if original.is_dir():
                        print(f"Preserving {relative}", flush=True)
                        copy_verified_tree(original, storage / relative)

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/download_data.py"),
                "--splits",
                "train",
                "validation",
            ],
            cwd=ROOT,
            check=True,
        )
        tokenizer, base_tokenizer = load_training_tokenizers(
            source, root=ROOT, allow_download=args.download_model
        )
        prepared = ROOT / config["data"]["prepared_directory"]
        if cache.exists():
            expected = verify_input_cache(cache)
            if prepared.exists():
                if file_hashes(prepared) != expected:
                    raise ValueError("Local and persistent prepared inputs differ.")
            else:
                copy_verified_tree(cache / "prepared", prepared)
        else:
            if not prepared.exists():
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/prepare_data.py"),
                        "--splits",
                        "train",
                        "validation",
                    ],
                    cwd=ROOT,
                    check=True,
                )
            cache.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".inputs-", dir=cache.parent) as temporary:
                staging = Path(temporary) / "complete"
                copy_verified_tree(prepared, staging / "prepared")
                write_json(staging / "files.json", file_hashes(staging / "prepared"))
                staging.rename(cache)
            verify_input_cache(cache)

        train, _, inputs = load_training_data(config, source, base_tokenizer, root=ROOT)
        model, initialization = load_shared_initialization(
            config,
            source,
            tokenizer,
            seed=args.seed,
            root=ROOT,
            directory=initializations,
            allow_download=args.download_model,
        )
        del model  # No forward/backward, optimizer, scheduler or training run is created.
        calibration_reference = None
        if (calibration_path / "calibration.json").is_file():
            calibration = json.loads((calibration_path / "calibration.json").read_text())
            recorded = json.loads((calibration_path / "run.json").read_text())
            original_config = dict(recorded["specification"]["configuration"])
            current_config = dict(config)
            original_config.pop("status", None)
            current_config.pop("status", None)
            if (
                calibration["identity"] != recorded["identity"]
                or original_config != current_config
                or recorded["specification"]["seed"] != args.seed
                or recorded["specification"]["inputs"] != inputs
                or recorded["specification"]["initialization_tensors_sha256"]
                != initialization["model_tensors_sha256"]
            ):
                raise ValueError("Calibration reference does not match these assets/settings.")
            calibration_reference = {
                "identity": calibration["identity"],
                "code_commit": recorded["git"]["commit"],
                "optimizer_updates": calibration["optimizer_updates"],
                "path": str(calibration_path),
                "weights_reused": False,
            }
        report = {
            "stage": "inputs_and_initialization_prepared",
            "prepared_at_utc": utc_now(),
            "training_started": False,
            "optimizer_updates_in_preparation": 0,
            "seed": args.seed,
            "git": {
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "dirty": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"], cwd=ROOT, text=True
                    ).strip()
                ),
            },
            "storage_root": str(storage),
            "initialization_directory": str(initializations),
            "initialization_tensors_sha256": initialization["model_tensors_sha256"],
            "input_cache": str(cache),
            "inputs": inputs,
            "budget": training_budget(len(train), config),
            "calibration_reference": calibration_reference,
            "source_hashes": {name: sha256(ROOT / name) for name in SOURCE_FILES},
            "environment": {
                "python": platform.python_version(),
                "packages": {
                    name: version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "datasets",
                        "numpy",
                        "scikit-learn",
                        "tokenizers",
                        "safetensors",
                        "pandas",
                        "accelerate",
                    )
                },
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            },
            "not_exercised": [
                "training updates",
                "full-epoch validation/selection",
                "training checkpoint save/resume",
                "augmented training",
            ],
            "test_accessed": False,
        }
        frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        report_directory.mkdir(parents=True, exist_ok=True)
        (report_directory / "pip-freeze.txt").write_text(frozen)
        write_json(report_directory / "readiness.json", report)
        print(f"Preparation complete; no training started. Report: {report_directory}", flush=True)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--reuse-storage-root", type=Path)
    parser.add_argument("--seed", type=int, default=586)
    parser.add_argument("--download-model", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
