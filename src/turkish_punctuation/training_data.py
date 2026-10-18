"""Validate only train/validation inputs and reproduce their saved clean encodings."""

import json
from pathlib import Path

from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerFast

from turkish_punctuation.data import read_split
from turkish_punctuation.preprocessing import encode_records, partition_records
from turkish_punctuation.training_io import file_hashes, sha256


def load_training_data(
    config: dict,
    source: dict,
    base_tokenizer: PreTrainedTokenizerFast,
    *,
    root: Path,
) -> tuple[Dataset, Dataset, dict[str, object]]:
    """Hash raw/prepared train and validation and compare all re-encoded columns.

    Load split subdirectories directly: never load/iterate the parent DatasetDict,
    which can also contain a held-out test split.
    """
    prepared = root / config["data"]["prepared_directory"]
    manifest = json.loads((prepared / "manifest.json").read_text())
    release = json.loads((root / "docs/data-manifest.json").read_text())
    if manifest["source"] != release or manifest["tokenizer"] != source:
        raise ValueError("Prepared source/tokenizer identity differs from project manifests.")
    if manifest["preprocessing"]["empty_token_policy"] != "quarantine_whole_row":
        raise ValueError("Expected the clean-v1 whole-row quarantine policy.")
    datasets: dict[str, Dataset] = {}
    inputs: dict[str, object] = {
        "prepared_manifest_sha256": sha256(prepared / "manifest.json")
    }
    for split in ("train", "validation"):
        item = next(item for item in release["files"] if item["split"] == split)
        raw_path = root / release["local_directory"] / item["filename"]
        if raw_path.stat().st_size != item["bytes"] or sha256(raw_path) != item["sha256"]:
            raise ValueError(f"Raw {split} differs from its pinned release hash.")
        records, _ = partition_records(read_split(raw_path))
        split_path = prepared / "dataset" / split
        saved = load_from_disk(str(split_path))
        if not isinstance(saved, Dataset):
            raise TypeError("Expected one explicit split dataset.")
        expected = manifest["splits"][split]["eligible"]
        expected_count = config["data"][f"expected_{split}_records"]
        if (
            len(saved) != expected_count
            or len(records) != expected_count
            or list(saved["row_id"]) != records["row_id"].tolist()
            or records["row_id"].tolist() != manifest["splits"][split]["eligible_row_ids"]
        ):
            raise ValueError(f"Prepared {split} row IDs/counts differ.")
        words = sum(map(len, records["tokens"].tolist()))
        if words != expected["token_entries"] or (
            split == "train" and words != config["data"]["expected_train_words"]
        ):
            raise ValueError(f"Prepared {split} word counts differ.")
        reproduced = encode_records(
            records,
            base_tokenizer,
            max_length=config["data"]["max_input_length"],
            batch_size=manifest["preprocessing"]["tokenizer_batch_size"],
        )
        if set(saved.column_names) != set(reproduced.column_names):
            raise ValueError(f"Prepared {split} columns differ.")
        for start in range(0, len(saved), 512):
            if saved[start : start + 512] != reproduced[start : start + 512]:
                raise ValueError(f"Prepared {split} encoding differs near record {start}.")
        datasets[split] = saved
        inputs[split] = {
            "raw_sha256": item["sha256"],
            "prepared_files": file_hashes(split_path),
            "records": len(saved),
            "words": words,
            "max_subwords": max(map(len, saved["input_ids"])),
            "label_counts": expected["label_counts"],
        }
        print(f"Verified {split}: {len(saved):,} records, {words:,} words", flush=True)
    return datasets["train"], datasets["validation"], inputs
