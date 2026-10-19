"""Run paired clean/augmented BERT fine-tuning or CUDA calibration; never load test."""

import argparse
import copy
import hashlib
import json
import os
import platform
import statistics
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from importlib.metadata import version
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import cast

# Required before the first CUDA context for deterministic CUDA matrix operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from turkish_punctuation.augmentation import load_epoch_views, verify_clean_pair
from turkish_punctuation.training import (
    TrainingProgress,
    epoch_permutation,
    evaluate_validation,
    make_optimizer,
    make_scheduler,
    train_epoch,
    training_budget,
    validate_training_config,
)
from turkish_punctuation.training_data import load_training_data
from turkish_punctuation.training_io import (
    capture_rng,
    directory_lock,
    file_hashes,
    load_checkpoint,
    restore_rng,
    save_checkpoint,
    save_rotating_checkpoint,
    seed_everything,
    sha256,
    utc_now,
    value_digest,
    write_json,
)
from turkish_punctuation.training_model import (
    export_model,
    load_shared_initialization,
    load_training_tokenizers,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "scripts/train.py",
    "src/turkish_punctuation/training.py",
    "src/turkish_punctuation/training_data.py",
    "src/turkish_punctuation/training_model.py",
    "src/turkish_punctuation/training_io.py",
    "src/turkish_punctuation/preprocessing.py",
    "src/turkish_punctuation/tokenization.py",
    "src/turkish_punctuation/data.py",
    "src/turkish_punctuation/metrics.py",
    "src/turkish_punctuation/augmentation.py",
    "src/turkish_punctuation/noise.py",
    "src/turkish_punctuation/noise_audit.py",
    "scripts/prepare_augmentation.py",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/training.json")
    parser.add_argument("--mode", choices=("calibrate", "train"), default="calibrate")
    parser.add_argument("--seed", type=int, default=586)
    parser.add_argument("--arm", choices=("clean", "augmented"), default="clean")
    parser.add_argument("--augmentation-directory", type=Path)
    parser.add_argument("--paired-clean-run", type=Path)
    parser.add_argument("--without-family", choices=("case_loss", "diacritic_loss", "typo_swap"))
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--initialization-directory", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--calibration-steps", type=int, default=30)
    parser.add_argument("--download-model", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="Resume an identical full training run."
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text())
    resolved = copy.deepcopy(config)
    if (args.batch_size is None) != (args.gradient_accumulation is None):
        raise ValueError("Specify batch size and gradient accumulation together.")
    if args.batch_size is not None:
        original = [
            config["optimization"]["per_device_train_batch_size"],
            config["optimization"]["gradient_accumulation_steps"],
        ]
        if [args.batch_size, args.gradient_accumulation] not in [
            original,
            *config["runtime"]["allowed_memory_fallback_batches"],
        ]:
            raise ValueError("Batch override is not one of the protocol's memory fallbacks.")
        resolved["optimization"]["per_device_train_batch_size"] = args.batch_size
        resolved["optimization"]["gradient_accumulation_steps"] = args.gradient_accumulation
    validate_training_config(resolved)
    seeds = [
        *config["seeds"]["initial_training_seeds"],
        *config["seeds"]["optional_confirmation_training_seeds"],
    ]
    if args.seed not in seeds:
        raise ValueError("Training seed must be explicitly listed in the protocol.")
    if args.mode == "calibrate" and (args.resume or args.calibration_steps <= 5):
        raise ValueError("Calibration requires >5 updates and a fresh output directory.")
    return resolved


def environment() -> dict:
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This runner supports one training process/GPU only.")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU required. Use the Colab training notebook; no CPU training started."
        )
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    properties = torch.cuda.get_device_properties(0)
    return {
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
            )
        },
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": properties.name,
        "gpu_memory_bytes": properties.total_memory,
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "visible_cuda_devices": torch.cuda.device_count(),
        "precision": "fp32",
        "tf32": False,
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def trim_events(path: Path, through_step: int) -> None:
    """Discard uncommitted updates after resuming an older durable checkpoint."""
    if not path.exists():
        return
    temporary = path.with_suffix(".tmp")
    with path.open() as source, temporary.open("w") as target:
        for line in source:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                break
            if item["global_step"] <= through_step:
                target.write(line)
    os.replace(temporary, path)


def run(
    args: argparse.Namespace, config: dict, destination: Path, *, on_verified: Callable[[], None]
) -> None:
    hardware = environment()
    source_path = ROOT / config["model"]["source_config"]
    source = json.loads(source_path.read_text())
    if (
        source["label_to_id"] != {"O": 0, "PERIOD": 1, "COMMA": 2, "QUESTION_MARK": 3}
        or source["max_input_length"] != config["data"]["max_input_length"]
    ):
        raise ValueError("Source label mapping/length differs from the training protocol.")
    tokenizer, base_tokenizer = load_training_tokenizers(
        source, root=ROOT, allow_download=args.download_model
    )
    train, validation, inputs = load_training_data(config, source, base_tokenizer, root=ROOT)
    budget = training_budget(len(train), config)
    if args.mode == "calibrate" and args.calibration_steps >= budget["updates_per_epoch"]:
        raise ValueError("Calibration must end before one full training epoch.")
    initialization_directory = (
        args.initialization_directory or ROOT / config["model"]["initialization_directory"]
    )
    model, initialization = load_shared_initialization(
        config,
        source,
        tokenizer,
        seed=args.seed,
        root=ROOT,
        directory=initialization_directory,
        allow_download=args.download_model,
    )
    views, augmentation, paired = {}, None, None
    augmentation_directory: Path | None = None
    if args.arm == "augmented":
        paired_directory = args.paired_clean_run or ROOT / config["paired_clean_run"]["directory"]
        paired = verify_clean_pair(
            paired_directory,
            config,
            seed=args.seed,
            inputs=inputs,
            initialization=initialization,
            hardware=hardware,
            budget=budget,
        )
        suffix = "" if args.without_family is None else f"-without-{args.without_family}"
        augmentation_directory = args.augmentation_directory or (
            ROOT / config["augmentation"]["prepared_directory"] / f"seed-{args.seed}{suffix}"
        )
        views, augmentation = load_epoch_views(
            augmentation_directory,
            train,
            base_tokenizer,
            config,
            seed=args.seed,
            without_family=args.without_family,
        )
        inputs["augmentation"] = {
            "identity": augmentation["identity"],
            "manifest_sha256": sha256(augmentation_directory / "manifest.json"),
            "epochs": {
                k: v["audit"]["encoded_content_sha256"] for k, v in augmentation["epochs"].items()
            },
        }
    elif args.without_family is not None or args.augmentation_directory is not None:
        raise ValueError("Augmentation directory/ablation flags require --arm augmented.")
    snapshot = {name: (ROOT / name).read_bytes() for name in SOURCE_PATHS}
    snapshot["training-config.json"] = args.config.read_bytes()
    snapshot["tokenizer-config.json"] = source_path.read_bytes()
    if augmentation is not None:
        snapshot["noise-config.json"] = (ROOT / config["augmentation"]["noise_config"]).read_bytes()
    specification = {
        "mode": args.mode,
        "arm": args.arm,
        "seed": args.seed,
        "calibration_steps": args.calibration_steps if args.mode == "calibrate" else None,
        "configuration": config,
        "inputs": inputs,
        "environment": hardware,
        "initialization_identity": initialization["identity"],
        "initialization_tensors_sha256": initialization["model_tensors_sha256"],
        "paired_clean_run": paired,
        "without_family": args.without_family,
        "source_hashes": {
            name: hashlib.sha256(value).hexdigest() for name, value in snapshot.items()
        },
    }
    identity = value_digest(specification)
    if args.resume:
        recorded = json.loads((destination / "run.json").read_text())
        if recorded["identity"] != identity or recorded["specification"] != specification:
            raise ValueError(
                "Resume requires identical source, config, data, initialization and environment."
            )
    else:
        try:
            git = {
                "commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "dirty": bool(
                    subprocess.check_output(
                        ["git", "status", "--porcelain"], cwd=ROOT, text=True
                    ).strip()
                ),
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            git = {"commit": None, "dirty": None}
        if augmentation is not None:
            assert augmentation_directory is not None
            metadata = destination / "augmentation"
            metadata.mkdir()
            (metadata / "manifest.json").write_bytes(
                (augmentation_directory / "manifest.json").read_bytes()
            )
            for epoch, item in augmentation["epochs"].items():
                write_json(metadata / f"epoch-{int(epoch):02d}-audit.json", item["audit"])
        for name, value in snapshot.items():
            path = destination / "source_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
        write_json(
            destination / "run.json",
            {
                "identity": identity,
                "specification": specification,
                "budget": budget,
                "created_at_utc": utc_now(),
                "git": git,
                "test_accessed": False,
            },
        )
    on_verified()
    cast(torch.nn.Module, model).to(device=torch.device("cuda:0"))
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, budget)
    progress = TrainingProgress()
    order = epoch_permutation(len(train), args.seed, 0)
    latest_checkpoint_path = None
    if args.resume:
        saved, checkpoint_path = load_checkpoint(destination / "checkpoints", identity=identity)
        latest_checkpoint_path = checkpoint_path
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        progress = TrainingProgress(**saved["progress"])
        order = saved["permutation"]
        if not 0 <= progress.epoch <= config["optimization"][
            "epochs"
        ] or not 0 <= progress.next_row <= len(train):
            raise ValueError("Checkpoint epoch/row cursor is out of range.")
        expected_steps = progress.epoch * budget["updates_per_epoch"]
        if progress.next_row:
            expected_steps += ceil(
                progress.next_row / config["optimization"]["effective_batch_size"]
            )
        if progress.global_step != expected_steps or scheduler.last_epoch != progress.global_step:
            raise ValueError("Checkpoint optimizer/scheduler/cursor counters disagree.")
        if len(progress.history) != progress.epoch:
            raise ValueError("Checkpoint validation history differs from completed epochs.")
        if order.numel() and not torch.equal(
            order, epoch_permutation(len(train), args.seed, progress.epoch)
        ):
            raise ValueError("Saved sampler permutation differs from the declared epoch order.")
        restore_rng(saved["rng"])
        trim_events(destination / "events.jsonl", progress.global_step)
        print(
            f"Resuming {checkpoint_path.name}: step {progress.global_step}, epoch {progress.epoch + 1}, next row {progress.next_row}",
            flush=True,
        )
    else:
        seed_everything(args.seed)
    started = perf_counter()
    checkpoint_seconds = 0.0
    step_seconds: list[float] = []
    torch.cuda.reset_peak_memory_stats()

    def checkpoint() -> None:
        nonlocal checkpoint_seconds, latest_checkpoint_path
        tick = perf_counter()
        existing_serials = [
            int(path.name.rsplit("-", 1)[1])
            for path in (destination / "checkpoints").glob("checkpoint-[0-9]*")
        ]
        progress.checkpoint_serial = max([progress.checkpoint_serial, *existing_serials]) + 1
        payload = {
            "identity": identity,
            "serial": progress.checkpoint_serial,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "progress": asdict(progress),
            "permutation": order,
            "rng": capture_rng(),
        }
        if config["artifacts"].get("checkpoint_layout") == "two_reused_slots":
            latest_checkpoint_path = save_rotating_checkpoint(
                destination / "checkpoints",
                payload,
                identity=identity,
                serial=progress.checkpoint_serial,
                protected=latest_checkpoint_path,
            )
        else:
            latest_checkpoint_path = save_checkpoint(
                destination / "checkpoints",
                payload,
                identity=identity,
                serial=progress.checkpoint_serial,
                keep=config["artifacts"]["keep_latest_resumable_checkpoints"],
            )
        checkpoint_seconds += perf_counter() - tick

    if args.mode == "train" and not args.resume:
        checkpoint()
    calibration_epoch = 1
    calibration_train = train
    if args.mode == "calibrate":
        if views:
            calibration_epoch = max(views, key=lambda e: max(map(len, views[e]["input_ids"])))
            calibration_train = views[calibration_epoch]
        # Include the longest prepared input batch in the peak-memory observation.
        lengths = list(map(len, calibration_train["input_ids"]))
        worst = sorted(range(len(train)), key=lambda i: (-lengths[i], i))[
            : config["optimization"]["effective_batch_size"]
        ]
        worst_set = set(worst)
        order = torch.tensor([*worst, *(i for i in order.tolist() if i not in worst_set)])
    with (destination / "events.jsonl").open("a", encoding="utf-8") as events:

        def on_update(item: dict) -> None:
            events.write(json.dumps(item, separators=(",", ":")) + "\n")
            step_seconds.append(item["optimizer_step_seconds"])
            interval = config["artifacts"]["resumable_checkpoint_every_optimizer_steps"]
            if progress.global_step % 25 == 0 or args.mode == "calibrate":
                events.flush()
                write_json(
                    destination / "progress.json",
                    {"status": "running", **item, "updated_at_utc": utc_now()},
                )
                print(
                    f"epoch={item['epoch']} step={item['global_step']} loss={item['window_loss']:.4f} lr={item['lr_used']:.3g}",
                    flush=True,
                )
            if args.mode == "train" and progress.global_step % interval == 0:
                events.flush()
                checkpoint()

        while progress.epoch < config["optimization"]["epochs"]:
            if progress.next_row < len(train):
                epoch_train = (
                    calibration_train
                    if args.mode == "calibrate"
                    else (views[progress.epoch + 1] if views else train)
                )
                train_epoch(
                    epoch_train,
                    tokenizer,
                    model,
                    optimizer,
                    scheduler,
                    progress,
                    order,
                    config,
                    on_update=on_update,
                    stop_at_step=args.calibration_steps if args.mode == "calibrate" else None,
                )
            if args.mode == "calibrate":
                steady = step_seconds[5:]
                report = {
                    "identity": identity,
                    "purpose": "Timing/memory calibration only; not a model-quality result.",
                    "optimizer_updates": progress.global_step,
                    "excluded_startup_updates": 5,
                    "mean_seconds_per_update": statistics.mean(steady),
                    "median_seconds_per_update": statistics.median(steady),
                    "estimated_training_update_seconds": statistics.mean(steady)
                    * budget["total_updates"],
                    "estimate_excludes": [
                        "data/model preparation",
                        "validation",
                        "checkpoint/export I/O",
                    ],
                    "peak_allocated_gpu_bytes": torch.cuda.max_memory_allocated(),
                    "peak_reserved_gpu_bytes": torch.cuda.max_memory_reserved(),
                    "longest_clean_batch_included": args.arm == "clean",
                    "longest_prepared_batch_included": True,
                    "calibration_view_epoch": calibration_epoch,
                    "seconds": perf_counter() - started,
                    "weights_retained": False,
                    "test_accessed": False,
                }
                write_json(destination / "calibration.json", report)
                write_json(
                    destination / "progress.json",
                    {
                        "status": "complete",
                        "mode": "calibrate",
                        "global_step": progress.global_step,
                    },
                )
                print(json.dumps(report, indent=2), flush=True)
                return
            if progress.epoch_words != inputs["train"]["words"]:
                raise RuntimeError("Epoch did not supervise each training word exactly once.")
            checkpoint()  # A validation interruption can resume without repeating training.
            epoch = progress.epoch + 1
            export = export_model(
                model, tokenizer, destination / "models" / f"epoch-{epoch:02d}", identity=identity
            )
            validation_directory = destination / "validation" / f"epoch-{epoch:02d}"
            if validation_directory.exists():
                result = json.loads((validation_directory / "metrics.json").read_text())
                if (
                    result["identity"] != identity
                    or result["model_tensors_sha256"] != export["model_tensors_sha256"]
                    or sha256(validation_directory / "predictions.jsonl")
                    != result["predictions_sha256"]
                ):
                    raise ValueError("Existing validation output differs from this epoch model.")
            else:
                validation_directory.parent.mkdir(parents=True, exist_ok=True)
                write_json(
                    destination / "progress.json",
                    {"status": "validating", "epoch": epoch, "global_step": progress.global_step},
                )
                with TemporaryDirectory(
                    prefix=".validation-", dir=validation_directory.parent
                ) as temporary:
                    staging = Path(temporary) / "complete"
                    staging.mkdir()
                    result = evaluate_validation(
                        validation,
                        tokenizer,
                        model,
                        batch_size=config["runtime"]["evaluation_batch_size"],
                        prediction_path=staging / "predictions.jsonl",
                    )
                    result.update(
                        {
                            "identity": identity,
                            "epoch": epoch,
                            "global_step": progress.global_step,
                            "model_tensors_sha256": export["model_tensors_sha256"],
                            "predictions_sha256": sha256(staging / "predictions.jsonl"),
                        }
                    )
                    write_json(staging / "metrics.json", result)
                    staging.rename(validation_directory)
            if result["metrics"]["evaluated_words"] != inputs["validation"]["words"]:
                raise RuntimeError("Validation word support differs from the fixed eligible split.")
            supports = {
                label: values["support"] for label, values in result["metrics"]["per_class"].items()
            }
            if supports != inputs["validation"]["label_counts"]:
                raise RuntimeError("Validation class supports differ from the prepared targets.")
            score = result["metrics"]["punctuation_macro"]["f1"]
            if score > progress.best_score:
                progress.best_score, progress.best_epoch = score, epoch
            progress.history.append(
                {
                    "epoch": epoch,
                    "global_step": progress.global_step,
                    "train_word_loss": progress.epoch_loss_sum / progress.epoch_words,
                    "validation": result,
                    "validation_report_sha256": sha256(validation_directory / "metrics.json"),
                    "model_export": export,
                }
            )
            print(f"Epoch {epoch}: validation macro-F1={score:.6f}", flush=True)
            progress.epoch += 1
            progress.next_row, progress.epoch_words, progress.epoch_loss_sum = 0, 0, 0.0
            order = (
                epoch_permutation(len(train), args.seed, progress.epoch)
                if progress.epoch < config["optimization"]["epochs"]
                else torch.empty(0, dtype=torch.int64)
            )
            checkpoint()
    if progress.global_step != budget["total_updates"]:
        raise RuntimeError("Actual optimizer update count differs from the planned budget.")
    if progress.best_epoch is None or len(progress.history) != config["optimization"]["epochs"]:
        raise RuntimeError("Training has no complete epoch selection history.")
    for item in progress.history:
        epoch = item["epoch"]
        if (
            file_hashes(destination / "models" / f"epoch-{epoch:02d}", exclude=("export.json",))
            != item["model_export"]["files"]
            or sha256(destination / "validation" / f"epoch-{epoch:02d}" / "metrics.json")
            != item["validation_report_sha256"]
        ):
            raise ValueError("A completed epoch artifact changed after its checkpoint commit.")
    write_json(
        destination / "summary.json",
        {
            "identity": identity,
            "arm": args.arm,
            "best_epoch": progress.best_epoch,
            "best_macro_f1": progress.best_score,
            "best_model": f"models/epoch-{progress.best_epoch:02d}",
            "final_model": f"models/epoch-{config['optimization']['epochs']:02d}",
            "global_step": progress.global_step,
            "history": progress.history,
            "this_invocation_seconds": perf_counter() - started,
            "this_invocation_checkpoint_seconds": checkpoint_seconds,
            "completed_at_utc": utc_now(),
            "test_accessed": False,
        },
    )
    write_json(
        destination / "progress.json",
        {
            "status": "complete",
            "global_step": progress.global_step,
            "best_epoch": progress.best_epoch,
        },
    )


def main() -> None:
    args = parse_arguments()
    config = resolve_config(args)
    destination = args.output_directory or ROOT / config["artifacts"]["root_directory"] / (
        f"{args.mode}-{args.arm}-seed-{args.seed}"
        + (f"-without-{args.without_family}" if args.without_family else "")
    )
    if args.resume and (
        not (destination / "run.json").exists() or (destination / "summary.json").exists()
    ):
        raise ValueError(
            "Resume requires an interrupted training run, not a completed/missing run."
        )
    with directory_lock(destination):
        if not args.resume and any(path.name != ".run.lock" for path in destination.iterdir()):
            raise FileExistsError("Output already contains a run; choose a new output directory.")
        verified = False

        def on_verified() -> None:
            nonlocal verified
            verified = True

        try:
            run(args, config, destination, on_verified=on_verified)
        except BaseException as error:
            if verified:
                write_json(
                    destination / "progress.json",
                    {
                        "status": "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "updated_at_utc": utc_now(),
                        "resume_policy": "Resume from the latest verified committed checkpoint; later updates are replayed.",
                    },
                )
            raise


if __name__ == "__main__":
    main()
