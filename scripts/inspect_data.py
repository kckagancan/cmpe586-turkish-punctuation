"""Audit released CSV files without modifying them or training a model."""

import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path

from turkish_punctuation.data import read_split

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def inspect_file(path: Path) -> tuple[dict, dict]:
    labels, lengths, texts, rows = Counter(), Counter(), Counter(), Counter()
    empty_token_labels = Counter()
    input_lengths, label_variants = {}, {}
    counts = Counter()
    records = read_split(path)
    columns = records[["tokens", "tags"]]
    for tokens, tags in columns.itertuples(index=False, name=None):
        labels.update(tags)
        lengths[len(tokens)] += 1
        counts["rows"] += 1
        counts["word_tokens"] += len(tokens)
        counts["empty_token_entries"] += tokens.count("")
        counts["rows_with_empty_tokens"] += "" in tokens
        empty_token_labels.update(tag for token, tag in zip(tokens, tags) if token == "")
        counts["rows_with_uppercase"] += any(any(c.isupper() for c in t) for t in tokens)
        counts["tokens_with_uppercase"] += sum(any(c.isupper() for c in t) for t in tokens)
        counts["tokens_with_turkish_diacritics"] += sum(
            any(c in "çğıöşüÇĞİÖŞÜ" for c in t) for t in tokens
        )
        counts["tokens_with_internal_whitespace"] += sum(
            any(c.isspace() for c in t) for t in tokens
        )
        counts["rows_with_question_mark"] += "QUESTION_MARK" in tags
        counts["rows_with_multiple_period_labels"] += tags.count("PERIOD") > 1
        counts["rows_over_128_words"] += len(tokens) > 128
        counts["rows_over_256_words"] += len(tokens) > 256
        text_key, row_key = fingerprint(tokens), fingerprint([tokens, tags])
        texts[text_key] += 1
        rows[row_key] += 1
        input_lengths[text_key] = len(tokens)
        label_variants.setdefault(text_key, set()).add(fingerprint(tags))
    stats = {
        **dict(counts),
        "label_counts": dict(sorted(labels.items())),
        "empty_token_label_counts": dict(sorted(empty_token_labels.items())),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "mean_words": counts["word_tokens"] / counts["rows"],
        "unique_input_sequences": len(texts),
        "duplicate_input_rows_beyond_first": sum(texts.values()) - len(texts),
        "duplicate_labeled_rows_beyond_first": sum(rows.values()) - len(rows),
        "inputs_with_multiple_label_sequences": sum(len(v) > 1 for v in label_variants.values()),
        "word_label_alignment": "all token/tag list lengths match; empty tokens counted separately",
    }
    indexes = {"texts": texts, "rows": rows, "lengths": input_lengths}
    return stats, indexes


def main() -> None:
    manifest = json.loads((ROOT / "docs/data-manifest.json").read_text())
    report = {"source_revision": manifest["revision"], "splits": {}, "cross_split_overlap": {}}
    indexes = {}
    for item in manifest["files"]:
        path = ROOT / manifest["local_directory"] / item["filename"]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"File does not match the manifest: {path}")
        stats, index = inspect_file(path)
        report["splits"][item["split"]] = stats
        indexes[item["split"]] = index
        print(item["split"], json.dumps(stats, ensure_ascii=False))
    for left, right in combinations(indexes, 2):
        a, b = indexes[left], indexes[right]
        common_texts = a["texts"].keys() & b["texts"].keys()
        common_rows = a["rows"].keys() & b["rows"].keys()
        overlap = {
            "shared_unique_inputs": len(common_texts),
            "shared_unique_inputs_at_least_10_words": sum(
                a["lengths"][key] >= 10 for key in common_texts
            ),
            "shared_unique_labeled_rows": len(common_rows),
            f"{left}_rows_with_shared_input": sum(a["texts"][key] for key in common_texts),
            f"{right}_rows_with_shared_input": sum(b["texts"][key] for key in common_texts),
        }
        report["cross_split_overlap"][f"{left}__{right}"] = overlap
        print(f"{left} / {right}", json.dumps(overlap))
    destination = ROOT / "docs/data-audit.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Saved aggregate audit: {destination}")


if __name__ == "__main__":
    main()
