"""Audit corruption exposure and tokenizer alignment without loading a model."""

import unicodedata
from collections import Counter
from math import ceil

import pandas as pd
from datasets import Dataset
from transformers import PreTrainedTokenizerFast

from turkish_punctuation.data import LABELS
from turkish_punctuation.noise import FAMILIES, Position, stable_digest
from turkish_punctuation.tokenization import IGNORE_INDEX, LABEL_TO_ID, align_labels


def validate_source(records: pd.DataFrame) -> None:
    """Require the fixed, nonempty-word clean subset; never normalize source text."""
    if records.empty or records["row_id"].duplicated().any():
        raise ValueError("Expected nonempty records with unique row IDs.")
    columns = records[["row_id", "tokens", "tags"]]
    for row_id, tokens, tags in columns.itertuples(index=False, name=None):
        if not tokens or len(tokens) != len(tags) or set(tags) - set(LABELS):
            raise ValueError(f"Row {row_id}: invalid word/label structure.")
        for index, word in enumerate(tokens):
            if not word or not unicodedata.is_normalized("NFC", word):
                raise ValueError(f"Row {row_id}, word {index}: empty or non-NFC source.")


def _protected_characters(word: str) -> list[tuple[int, str]]:
    return [
        (i, char)
        for i, char in enumerate(word)
        if char.isspace() or unicodedata.category(char).startswith("P")
    ]


def audit_changes(
    clean: pd.DataFrame,
    noisy: pd.DataFrame,
    edits: list[dict],
    *,
    example_seed: int,
    condition_name: str,
    example_count: int,
) -> tuple[dict, list[dict]]:
    """Verify fixed targets, measure realized edits/overlap and sample local examples."""
    if clean["row_id"].tolist() != noisy["row_id"].tolist():
        raise ValueError("Noise changed row IDs or row order.")
    masks: dict[str, set[Position]] = {family: set() for family in FAMILIES}
    stage_changes, stage_noops = Counter(), Counter()
    for edit in edits:
        position = (edit["row_id"], edit["word_index"])
        family = edit["family"]
        if position in masks[family]:
            raise ValueError("The same family edited a word more than once.")
        masks[family].add(position)
        stage_changes[family] += edit["before"] != edit["after"]
        stage_noops[family] += edit["before"] == edit["after"]
    selected = set().union(*masks.values())
    supports, changed_labels = Counter(), Counter()
    selected_labels = {family: Counter() for family in FAMILIES}
    overlap = Counter()
    changed_positions, examples = set(), []
    total_words = changed_characters = changed_rows = selected_characters = 0
    clean_rows = clean[["row_id", "tokens", "tags"]].itertuples(index=False, name=None)
    noisy_rows = noisy[["row_id", "tokens", "tags"]].itertuples(index=False, name=None)
    for (row_id, words, tags), (_, outputs, output_tags) in zip(
        clean_rows, noisy_rows, strict=True
    ):
        if tags != output_tags or len(words) != len(outputs):
            raise ValueError(f"Row {row_id}: noise changed targets or word count.")
        row_changed = False
        for index, (word, output, tag) in enumerate(zip(words, outputs, tags, strict=True)):
            position = (row_id, index)
            total_words += 1
            supports[tag] += 1
            # All v1 operators preserve codepoint length as well as word boundaries.
            if (
                not output
                or len(word) != len(output)
                or not unicodedata.is_normalized("NFC", output)
                or _protected_characters(word) != _protected_characters(output)
            ):
                raise ValueError(f"Row {row_id}, word {index}: invalid character/boundary change.")
            membership = [family for family in FAMILIES if position in masks[family]]
            if membership:
                overlap["+".join(membership)] += 1
                selected_characters += len(word)
            for family in membership:
                selected_labels[family][tag] += 1
            if word == output:
                continue
            if position not in selected:
                raise ValueError("An unselected word changed.")
            changed_positions.add(position)
            changed_labels[tag] += 1
            changed_characters += sum(a != b for a, b in zip(word, output, strict=True))
            row_changed = True
            examples.append(
                {
                    "row_id": row_id,
                    "word_index": index,
                    "clean": word,
                    "noisy": output,
                    "tag": tag,
                    "selected_families": membership,
                }
            )
        changed_rows += row_changed
    if any(sum(selected_labels[f].values()) != len(masks[f]) for f in FAMILIES):
        raise ValueError("Edit log contains a position outside the source words.")
    examples.sort(
        key=lambda item: stable_digest(
            [example_seed, condition_name, item["row_id"], item["word_index"], "audit_example"]
        )
    )
    families = {}
    for family in FAMILIES:
        positions = sorted(masks[family])
        families[family] = {
            "selected_words": len(positions),
            "selected_positions_sha256": stable_digest(positions).hex(),
            "stage_changed_words": stage_changes[family],
            "stage_noop_words": stage_noops[family],
            "selected_label_counts": {tag: selected_labels[family][tag] for tag in LABELS},
        }
    report = {
        "rows": len(clean),
        "words": total_words,
        "changed_rows": changed_rows,
        "changed_words": len(changed_positions),
        "changed_word_percent": 100 * len(changed_positions) / total_words,
        "selected_union_words": len(selected),
        "selected_union_percent": 100 * len(selected) / total_words,
        "selected_but_final_unchanged_words": len(selected - changed_positions),
        "changed_character_positions": changed_characters,
        "selected_word_character_positions": selected_characters,
        "word_length_change": 0,
        "overlap_membership_counts": dict(sorted(overlap.items())),
        "label_support": {tag: supports[tag] for tag in LABELS},
        "final_changed_label_counts": {tag: changed_labels[tag] for tag in LABELS},
        "final_changed_percent_within_label": {
            tag: 100 * changed_labels[tag] / supports[tag] if supports[tag] else 0.0
            for tag in LABELS
        },
        "families": families,
        "invariants_passed": True,
    }
    return report, examples[:example_count]


def encode_and_audit(
    records: pd.DataFrame,
    tokenizer: PreTrainedTokenizerFast,
    *,
    max_length: int,
    batch_size: int,
) -> tuple[Dataset, dict]:
    """Retain word_ids long enough to audit fertility, then save model-ready inputs.

    Use the shared align_labels rule. No truncation, padding or condition-specific
    exclusions are allowed; a coverage/length failure stops this preparation run.
    """
    if batch_size < 1 or not 0 < max_length <= tokenizer.model_max_length:
        raise ValueError("Invalid tokenizer batch size or sequence limit.")
    columns: dict[str, list] = {"row_id": [], "labels": []}
    lengths, fertility = [], Counter()
    unknown_pieces = unknown_words = subwords = total_words = 0
    for start in range(0, len(records), batch_size):
        batch = records.iloc[start : start + batch_size]
        encoded = tokenizer(
            batch["tokens"].tolist(), is_split_into_words=True, truncation=False, padding=False
        )
        rows = batch[["row_id", "tokens", "tags"]].itertuples(index=False, name=None)
        for i, (row_id, words, tags) in enumerate(rows):
            ids = encoded.data["input_ids"][i]
            if len(ids) > max_length:
                raise ValueError(f"Row {row_id}: {len(ids)} tokens exceeds {max_length}.")
            word_ids = encoded.word_ids(batch_index=i)
            try:
                aligned = align_labels(word_ids, tags)
            except ValueError as error:
                raise ValueError(f"Row {row_id}: {error}") from error
            if [value for value in aligned if value != IGNORE_INDEX] != [
                LABEL_TO_ID[t] for t in tags
            ]:
                raise ValueError(f"Row {row_id}: first-subword targets changed.")
            word_pieces = Counter(index for index in word_ids if index is not None)
            fertility.update(word_pieces.values())
            subwords += sum(word_pieces.values())
            total_words += len(words)
            unknown = [
                (token, index)
                for token, index in zip(ids, word_ids, strict=True)
                if index is not None and token == tokenizer.unk_token_id
            ]
            unknown_pieces += len(unknown)
            unknown_words += len({index for _, index in unknown})
            lengths.append(len(ids))
            columns["row_id"].append(row_id)
            columns["labels"].append(aligned)
        for key, values in encoded.data.items():
            columns.setdefault(key, []).extend(values)
    ordered = sorted(lengths)
    if not ordered:
        raise ValueError("Cannot audit empty records.")
    return Dataset.from_dict(columns), {
        "rows": len(records),
        "words": total_words,
        "subwords": subwords,
        "mean_subwords_per_word": subwords / total_words,
        "subwords_per_word_histogram": dict(sorted(fertility.items())),
        "length_including_special_tokens": {
            "max": ordered[-1],
            **{f"p{p}": ordered[ceil(len(ordered) * p / 100) - 1] for p in (50, 95, 99)},
        },
        "unknown_subwords": unknown_pieces,
        "words_with_unknown_subwords": unknown_words,
        "unknown_subword_percent": 100 * unknown_pieces / subwords,
        "missing_words": 0,
        "rows_over_limit": 0,
        "all_first_subword_targets_preserved": True,
    }
