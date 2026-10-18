"""Read and validate punctuation records while preserving their original token slots."""

import ast
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

LABELS = ("O", "PERIOD", "COMMA", "QUESTION_MARK")


def _parse_list(value: str, *, column: str, row_id: int) -> list[str]:
    """Parse a serialized list and include its source location in any error."""
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError) as error:
        raise ValueError(f"Row {row_id}, {column}: invalid list representation.") from error
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"Row {row_id}, {column}: expected a list of strings.")
    return parsed


def parse_record(sentence: str, tags: str, *, row_id: int) -> tuple[list[str], list[str]]:
    """Convert one CSV record into aligned tokens and labels; retain empty strings."""
    tokens = _parse_list(sentence, column="sentence", row_id=row_id)
    labels = _parse_list(tags, column="tags", row_id=row_id)
    if not tokens:
        raise ValueError(f"Row {row_id}: the token list is empty.")
    if len(tokens) != len(labels):
        raise ValueError(f"Row {row_id}: token and label counts differ.")
    unknown = sorted(set(labels) - set(LABELS))
    if unknown:
        raise ValueError(f"Row {row_id}: unknown labels {unknown}.")
    return tokens, labels


def read_split(path: str | Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Read a released CSV as row_id/tokens/tags, without cleaning or reordering.

    row_id is the zero-based CSV data-row position, excluding the header.
    nrows optionally reads a prefix for development; None reads the whole split.
    Empty token strings are kept because they can carry punctuation labels.
    """
    if nrows is not None and nrows < 1:
        raise ValueError("nrows must be positive or None.")
    # Without index_col=False, pandas may silently treat an extra field as an index.
    # With it, extra fields can be discarded with a warning; turn that into an error.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", pd.errors.ParserWarning)
            raw = pd.read_csv(
                path,
                encoding="utf-8",
                dtype=str,
                keep_default_na=False,
                skip_blank_lines=False,
                index_col=False,
                nrows=nrows,
            )
    except (pd.errors.ParserWarning, pd.errors.ParserError) as error:
        raise ValueError(f"Invalid CSV field structure: {path}") from error
    if raw.columns.tolist() != ["sentence", "tags"]:
        raise ValueError(f"Expected sentence/tags columns in that order: {path}")
    if raw.empty:
        raise ValueError(f"No records found: {path}")

    records = []
    for row_id, (sentence, tags) in enumerate(raw.itertuples(index=False, name=None)):
        tokens, labels = parse_record(sentence, tags, row_id=row_id)
        records.append({"row_id": row_id, "tokens": tokens, "tags": labels})
    return pd.DataFrame(records, columns=["row_id", "tokens", "tags"])


def find_empty_tokens(records: pd.DataFrame) -> pd.DataFrame:
    """Locate empty slots in validated records, retaining their attached labels."""
    findings = []
    columns = records[["row_id", "tokens", "tags"]]
    for row_id, tokens, tags in columns.itertuples(index=False, name=None):
        for token_index, (token, tag) in enumerate(zip(tokens, tags, strict=True)):
            if token == "":
                findings.append({"row_id": row_id, "token_index": token_index, "tag": tag})
    return pd.DataFrame(findings, columns=["row_id", "token_index", "tag"])


def summarize_split(records: pd.DataFrame) -> dict:
    """Summarize validated records; token_entries includes the released empty slots."""
    empty_tokens = find_empty_tokens(records)
    counts = Counter(tag for tags in records["tags"] for tag in tags)
    return {
        "records": len(records),
        "token_entries": int(records["tokens"].map(len).sum()),
        "empty_token_entries": len(empty_tokens),
        "rows_with_empty_tokens": int(empty_tokens["row_id"].nunique()),
        "label_counts": {label: counts[label] for label in LABELS},
    }
