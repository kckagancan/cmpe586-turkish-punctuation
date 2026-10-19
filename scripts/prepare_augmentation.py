"""Prepare/audit epoch training views without a model, optimizer or test access."""

import argparse
import json
import platform
import shutil
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from datasets import Dataset, disable_progress_bars, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from turkish_punctuation.augmentation import (
    build_epoch_view,
    encoded_digest,
    load_epoch_views,
    validate_augmentation,
)
from turkish_punctuation.data import read_split
from turkish_punctuation.noise import FAMILIES, source_words_digest
from turkish_punctuation.preprocessing import encode_records, partition_records
from turkish_punctuation.training import validate_training_config
from turkish_punctuation.training_io import file_hashes, sha256, utc_now, value_digest, write_json

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "scripts/prepare_augmentation.py",
    "src/turkish_punctuation/augmentation.py",
    "src/turkish_punctuation/noise.py",
    "src/turkish_punctuation/noise_audit.py",
    "src/turkish_punctuation/preprocessing.py",
    "src/turkish_punctuation/tokenization.py",
    "src/turkish_punctuation/data.py",
)


def write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training-augmented.json"))
    parser.add_argument("--seed", type=int, default=586)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--without-family", choices=FAMILIES)
    parser.add_argument("--download-tokenizer", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text())
    validate_training_config(config)
    noise_path = ROOT / config["augmentation"]["noise_config"]
    noise_config = json.loads(noise_path.read_text())
    validate_augmentation(config, noise_config)
    if args.seed not in config["seeds"]["initial_training_seeds"]:
        raise ValueError("Use a declared initial training seed.")
    suffix = "" if args.without_family is None else f"-without-{args.without_family}"
    destination = (
        args.output_directory
        or ROOT / config["augmentation"]["prepared_directory"] / f"seed-{args.seed}{suffix}"
    ).resolve()
    if destination.exists() != args.verify_only:
        raise ValueError(
            "Use a new output directory, or --verify-only for existing prepared views."
        )
    release = json.loads((ROOT / "docs/data-manifest.json").read_text())
    item = next(x for x in release["files"] if x["split"] == "train")
    raw = ROOT / release["local_directory"] / item["filename"]
    if sha256(raw) != item["sha256"] or raw.stat().st_size != item["bytes"]:
        raise ValueError("Raw train source differs from the pinned release.")
    clean, _ = partition_records(read_split(raw))
    prepared = ROOT / config["data"]["prepared_directory"]
    clean_manifest = json.loads((prepared / "manifest.json").read_text())
    source = json.loads((ROOT / config["model"]["source_config"]).read_text())
    if (
        clean_manifest["source"] != release
        or clean_manifest["tokenizer"] != source
        or clean["row_id"].tolist() != clean_manifest["splits"]["train"]["eligible_row_ids"]
        or len(clean) != config["data"]["expected_train_records"]
    ):
        raise ValueError("Clean train preparation/source IDs differ.")
    tokenizer = AutoTokenizer.from_pretrained(
        source["model_id"],
        revision=source["revision"],
        cache_dir=str(ROOT / source["cache_directory"]),
        local_files_only=not args.download_tokenizer,
        use_fast=True,
        trust_remote_code=False,
        token=False,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("Expected the pinned fast tokenizer.")
    disable_progress_bars()
    saved = load_from_disk(str(prepared / "dataset/train"))
    if not isinstance(saved, Dataset):
        raise TypeError("Expected the explicit clean train Dataset.")
    if len(saved) != len(clean) or list(saved["row_id"]) != clean["row_id"].tolist():
        raise ValueError("Saved clean train row IDs/counts differ.")
    reproduced = encode_records(clean, tokenizer, max_length=config["data"]["max_input_length"])
    for start in range(0, len(saved), 512):
        if saved[start : start + 512] != reproduced[start : start + 512]:
            raise ValueError("Clean train encodings differ from current pinned tokenization.")
    specification = {
        "training_version": config["version"],
        "training_seed": args.seed,
        "epochs": config["optimization"]["epochs"],
        "without_family": args.without_family,
        "augmentation": config["augmentation"],
        "noise_config": noise_config,
        "train_raw_sha256": item["sha256"],
        "source_words_sha256": source_words_digest(clean),
        "clean_encoded_content_sha256": encoded_digest(saved),
        "tokenizer_config": source,
        "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
    }
    if args.verify_only:
        old = json.loads((destination / "manifest.json").read_text())
        if old["specification"] != specification:
            raise ValueError("Prepared views differ from current code/configuration/inputs.")
        load_epoch_views(
            destination,
            saved,
            tokenizer,
            config,
            seed=args.seed,
            without_family=args.without_family,
        )
        print(f"Verified all epoch views; zero model/optimizer updates: {destination}", flush=True)
        return
    started = perf_counter()
    report = {
        "identity": value_digest(specification),
        "specification": specification,
        "epochs": {},
        "created_at_utc": utc_now(),
        "training_performed": False,
        "model_weights_loaded": False,
        "test_accessed": False,
        "python": platform.python_version(),
        "packages": {
            name: version(name) for name in ("datasets", "transformers", "tokenizers", "pandas")
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".training-views-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        for name in SOURCES:
            target = staging / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        write_json(staging / "source_snapshot/training-config.json", config)
        write_json(staging / "source_snapshot/noise-config.json", noise_config)
        for epoch in range(1, config["optimization"]["epochs"] + 1):
            print(f"Preparing training view: epoch {epoch}", flush=True)
            mixed, encoded, edits, audit, examples = build_epoch_view(
                clean,
                tokenizer,
                config,
                noise_config,
                seed=args.seed,
                epoch=epoch,
                without_family=args.without_family,
            )
            path = staging / f"epoch-{epoch:02d}"
            path.mkdir()
            write_jsonl(path / "records.jsonl", mixed.to_dict(orient="records"))
            write_jsonl(path / "edits.jsonl", edits)
            write_json(path / "examples.json", examples)
            encoded.save_to_disk(str(path / "dataset"))
            write_json(path / "audit.json", audit)
            report["epochs"][str(epoch)] = {"audit": audit, "files": file_hashes(path)}
            print(
                f"Epoch {epoch}: {audit['selected_view_rows']:,} noisy-view rows; "
                f"{audit['changes']['changed_word_percent']:.4f}% words changed; "
                f"max {audit['tokenization']['length_including_special_tokens']['max']} subwords",
                flush=True,
            )
            del mixed, encoded, edits
        report["elapsed_seconds"] = perf_counter() - started
        report["completed_at_utc"] = utc_now()
        write_json(staging / "manifest.json", report)
        load_epoch_views(
            staging, saved, tokenizer, config, seed=args.seed, without_family=args.without_family
        )
        staging.rename(destination)
    print(f"Prepared and reloaded all epochs; zero training updates: {destination}", flush=True)


if __name__ == "__main__":
    main()
