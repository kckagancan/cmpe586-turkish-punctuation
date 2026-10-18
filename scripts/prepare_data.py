"""Prepare local clean-v1 datasets and quarantine; do not train or download weights."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import DatasetDict
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from turkish_punctuation.data import read_split, summarize_split
from turkish_punctuation.preprocessing import encode_records, make_dataloader, partition_records
from turkish_punctuation.tokenization import IGNORE_INDEX, LABEL_TO_ID

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=["train", "validation", "test"],
    )
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/preprocessing.json").read_text())
    tokenizer_config = json.loads((ROOT / "configs/tokenizer.json").read_text())
    manifest = json.loads((ROOT / "docs/data-manifest.json").read_text())
    destination = ROOT / config["output_directory"]
    if destination.exists():
        raise FileExistsError(f"Prepared data already exists; use a new version: {destination}")
    if config["empty_token_policy"] != "quarantine_whole_row":
        raise ValueError("Only whole-row quarantine is implemented.")
    if (
        tokenizer_config["label_to_id"] != LABEL_TO_ID
        or tokenizer_config["ignore_index"] != IGNORE_INDEX
        or tokenizer_config["alignment"] != "first_subword"
    ):
        raise ValueError("Tokenizer configuration differs from the implemented alignment.")
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
        raise ValueError("A fast tokenizer with word_ids() is required.")
    if tokenizer.model_max_length != tokenizer_config["max_input_length"]:
        raise ValueError("Tokenizer length limit differs from the recorded configuration.")
    report = {
        "preprocessing": config,
        "tokenizer": tokenizer_config,
        "source": manifest,
        "packages": {name: version(name) for name in ("torch", "transformers", "datasets")},
        "evaluation_scope": "Fixed eligible subset, not the full published benchmark.",
        "noise_policy": "Derive noise from eligible rows; fail on new coverage/length errors.",
        "splits": {},
        "training_performed": False,
        "prepared_splits": [
            name for name in ("train", "validation", "test") if name in args.splits
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "prepared"
        (staging / "quarantine").mkdir(parents=True)
        encoded_splits = DatasetDict()
        for item in manifest["files"]:
            split = item["split"]
            if split not in args.splits:
                continue
            path = ROOT / manifest["local_directory"] / item["filename"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError(f"Raw file differs from manifest: {path}")
            records = read_split(path)
            eligible, quarantine = partition_records(records)
            quarantine_path = staging / "quarantine" / f"{split}.jsonl"
            with quarantine_path.open("w", encoding="utf-8") as stream:
                columns = quarantine[["row_id", "tokens", "tags"]]
                for row_id, tokens, tags in columns.itertuples(index=False, name=None):
                    record = {
                        "row_id": row_id,
                        "tokens": tokens,
                        "tags": tags,
                        "reason": "empty_token",
                    }
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            encoded = encode_records(
                eligible,
                tokenizer,
                max_length=tokenizer_config["max_input_length"],
                batch_size=config["tokenizer_batch_size"],
            )
            encoded_splits[split] = encoded
            batch = next(
                iter(
                    make_dataloader(
                        encoded,
                        tokenizer,
                        batch_size=config["preview_batch_size"],
                        shuffle=False,
                        seed=config["seed"],
                    )
                )
            )
            report["splits"][split] = {
                "raw": summarize_split(records),
                "eligible": summarize_split(eligible),
                "quarantined": summarize_split(quarantine),
                "eligible_row_ids": eligible["row_id"].tolist(),
                "quarantined_row_ids": quarantine["row_id"].tolist(),
                "preview_batch_shapes": {key: list(value.shape) for key, value in batch.items()},
                "preview_supervised_words": int((batch["labels"] != IGNORE_INDEX).sum().item()),
            }
            print(f"{split}: {len(eligible)} eligible, {len(quarantine)} quarantined", flush=True)
        encoded_splits.save_to_disk(str(staging / "dataset"))
        (staging / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(destination)
    print(f"Saved prepared datasets and manifest: {destination}")


if __name__ == "__main__":
    main()
