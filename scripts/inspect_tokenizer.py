"""Audit the pinned tokenizer and simulate exclusions without preparing training data."""

import argparse
import hashlib
import json
from collections import Counter
from math import ceil
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from turkish_punctuation.data import LABELS, read_split
from turkish_punctuation.tokenization import IGNORE_INDEX, LABEL_TO_ID, align_labels

ROOT = Path(__file__).resolve().parents[1]


def inspect_split(records: pd.DataFrame, tokenizer: PreTrainedTokenizerFast, limit: int) -> dict:
    """Measure coverage and lengths before filtering; count all labels in affected rows."""
    counts, missing_labels, excluded_labels = Counter(), Counter(), Counter()
    lengths, affected_ids = [], []
    for start in range(0, len(records), 256):
        batch = records.iloc[start : start + 256]
        encodings = tokenizer(
            batch["tokens"].tolist(),
            is_split_into_words=True,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        columns = batch[["row_id", "tokens", "tags"]]
        for i, (row_id, tokens, tags) in enumerate(columns.itertuples(index=False, name=None)):
            ids = encodings.data["input_ids"][i]
            word_ids = encodings.word_ids(batch_index=i)
            represented = {word_id for word_id in word_ids if word_id is not None}
            missing = set(range(len(tokens))) - represented
            too_long = len(ids) > limit
            counts["records"] += 1
            counts["word_slots"] += len(tokens)
            counts["missing_word_slots"] += len(missing)
            counts["missing_nonempty_word_slots"] += sum(tokens[j] != "" for j in missing)
            counts["rows_with_missing_words"] += bool(missing)
            counts["rows_over_limit"] += too_long
            counts["unknown_subword_tokens"] += ids.count(tokenizer.unk_token_id)
            missing_labels.update(tags[j] for j in missing)
            lengths.append(len(ids))
            if missing or too_long:
                counts["candidate_excluded_rows"] += 1
                excluded_labels.update(tags)
                affected_ids.append(row_id)
            else:
                aligned = align_labels(word_ids, tags)
                supervised = [value for value in aligned if value != IGNORE_INDEX]
                if supervised != [LABEL_TO_ID[tag] for tag in tags]:
                    raise RuntimeError(f"Supervision changed in row {row_id}.")
                counts["fully_aligned_rows"] += 1
    ordered = sorted(lengths)
    return {
        **dict(counts),
        "candidate_excluded_row_percent": 100 * counts["candidate_excluded_rows"] / len(records),
        "missing_word_label_counts": {label: missing_labels[label] for label in LABELS},
        "candidate_excluded_label_counts": {label: excluded_labels[label] for label in LABELS},
        "length_including_special_tokens": {
            "max": max(lengths),
            "p50": ordered[ceil(len(ordered) * 0.50) - 1],
            "p95": ordered[ceil(len(ordered) * 0.95) - 1],
            "p99": ordered[ceil(len(ordered) * 0.99) - 1],
        },
        "first_affected_row_ids": affected_ids[:10],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-tokenizer", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/tokenizer.json").read_text())
    manifest = json.loads((ROOT / "docs/data-manifest.json").read_text())
    if config["label_to_id"] != LABEL_TO_ID or config["ignore_index"] != IGNORE_INDEX:
        raise ValueError("Tokenizer configuration and alignment constants disagree.")
    if config["alignment"] != "first_subword":
        raise ValueError("Only first-subword alignment is implemented.")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        cache_dir=str(ROOT / config["cache_directory"]),
        local_files_only=not args.download_tokenizer,
        use_fast=True,
        trust_remote_code=False,
        token=False,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("A fast tokenizer with word_ids() is required.")
    if tokenizer.model_max_length != config["max_input_length"]:
        raise ValueError("Tokenizer length limit differs from the recorded configuration.")
    example_words = ["Çocuklarımızla", "İstanbul'da", "buluşabilecek", "miyiz"]
    example_tags = ["O", "O", "O", "QUESTION_MARK"]
    example = tokenizer(example_words, is_split_into_words=True)
    report = {
        "model_id": config["model_id"],
        "tokenizer_revision": config["revision"],
        "data_revision": manifest["revision"],
        "max_input_length": config["max_input_length"],
        "alignment": config["alignment"],
        "do_lower_case": tokenizer.init_kwargs.get("do_lower_case"),
        "synthetic_example": {
            "words": example_words,
            "tags": example_tags,
            "subwords": tokenizer.convert_ids_to_tokens(example.data["input_ids"]),
            "word_ids": example.word_ids(),
            "input_ids": example.data["input_ids"],
            "labels": align_labels(example.word_ids(), example_tags),
        },
        "policy_status": "Row exclusion is simulated only; no processed split was saved.",
        "splits": {},
    }
    for item in manifest["files"]:
        path = ROOT / manifest["local_directory"] / item["filename"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Raw file differs from manifest: {path}")
        summary = inspect_split(read_split(path), tokenizer, config["max_input_length"])
        report["splits"][item["split"]] = summary
        print(item["split"], json.dumps(summary, ensure_ascii=False), flush=True)
    destination = ROOT / "docs/tokenization-audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved tokenizer audit: {destination}")


if __name__ == "__main__":
    main()
