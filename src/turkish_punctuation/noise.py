"""Deterministic word-preserving corruptions; selection never inspects labels."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

FAMILIES = ("case_loss", "diacritic_loss", "typo_swap")
Position = tuple[int, int]


def stable_digest(parts: Sequence[Any]) -> bytes:
    """Use explicit UTF-8 serialization rather than Python's process-randomized hash."""
    payload = json.dumps(list(parts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).digest()


def validate_config(config: dict) -> None:
    """Reject settings whose claimed semantics the current implementation cannot honor."""
    fixed = {
        "sampling": {
            "method": "lowest_sha256_rank_among_eligible_clean_word_positions",
            "selection_scope": "whole_split",
            "rate_denominator": "all_original_word_positions_in_eligible_split",
            "target_count_formula": "total_words * coverage_percent // 100",
            "eligible_pool": "clean_source_words_for_each_family",
            "nested_severity_prefixes": True,
            "key_fields": ["version", "seed", "split", "row_id", "word_index", "family", "purpose"],
            "selection_purpose": "select",
            "key_serialization": "JSON array; ensure_ascii=False; separators=(',', ':'); UTF-8",
            "digest": "sha256_full_digest_bytes",
            "tie_break": ["row_id", "word_index"],
            "excluded_key_fields": ["severity", "condition_name", "model_id", "labels"],
            "insufficient_eligible_words": "fail_condition_without_lowering_budget",
        },
        "invariants": {
            "preserve_raw_files": True,
            "preserve_split_row_ids_and_word_order": True,
            "preserve_word_count_and_tags": True,
            "allow_whitespace_or_punctuation_edits": False,
            "allow_empty_output_words": False,
            "source_unicode_form": "NFC",
            "non_nfc_input_policy": "fail_audit_without_silent_normalization",
            "retokenize_and_realign_each_variant": True,
            "missing_word_or_overlength_policy": "fail_before_model_scoring_no_condition_specific_row_drops",
        },
    }
    for section, expected in fixed.items():
        if config[section] != expected:
            raise ValueError(
                f"Unsupported {section} rules; update implementation/protocol together."
            )
    if set(config["families"]) != set(FAMILIES):
        raise ValueError("Only case_loss, diacritic_loss and typo_swap are implemented.")
    expected_operations = (
        "lowercase_entire_selected_word_with_Turkish_I_rules",
        "translate_all_listed_characters_in_selected_word",
        "one_adjacent_internal_character_swap",
    )
    for family, operation in zip(FAMILIES, expected_operations, strict=True):
        if config["families"][family]["operation"] != operation:
            raise ValueError(f"Unsupported operation for {family}.")
    case = config["families"]["case_loss"]
    if case["remaining_casing"] != "Python_str_lower" or case["normalize_source"] is not False:
        raise ValueError(
            "Case loss requires ordinary lowercase after explicit overrides, no NFC edits."
        )
    if any(
        config["families"][name]["eligible_when"] != "operation_changes_clean_word"
        for name in FAMILIES[:2]
    ):
        raise ValueError("Case/folding eligibility must require an actual clean-word change.")
    if config["families"]["diacritic_loss"]["other_characters"] != "preserve":
        raise ValueError("Unlisted characters must be preserved.")
    swap = config["families"]["typo_swap"]
    expected_swap = {
        "preserve_first_and_last_character": True,
        "require_unequal_adjacent_characters": True,
        "candidate_left_indices": "range(1, len(word) - 2)",
        "position_choice": "lowest_sha256_of_base_key_with_purpose_swap_position_plus_left_index",
        "position_tie_break": "lowest_left_index",
        "maximum_swaps_per_selected_word": 1,
    }
    if any(swap.get(key) != value for key, value in expected_swap.items()):
        raise ValueError("Unsupported swap policy.")
    if type(swap["minimum_word_length"]) is not int or swap["minimum_word_length"] < 4:
        raise ValueError("Interior swaps require minimum_word_length >= 4.")
    alphabet = swap["alphabet"]
    if not alphabet or len(set(alphabet)) != len(alphabet) or not alphabet.isalpha():
        raise ValueError("Swap alphabet must contain unique letters only.")
    for mapping in (case["turkish_overrides"], config["families"]["diacritic_loss"]["mapping"]):
        if not mapping or any(
            len(a) != 1 or len(b) != 1 or not (a + b).isalpha() for a, b in mapping.items()
        ):
            raise ValueError("Character maps must contain single-letter pairs.")
    combined = config["combined"]
    if (
        set(combined["apply_order"]) != set(FAMILIES)
        or len(combined["apply_order"]) != len(FAMILIES)
        or combined["plans"] != "reuse_each_single_family_plan_at_same_level_and_seed"
        or combined["final_noops_or_cancellations"] != "record_without_resampling"
        or combined["report_union_and_overlap"] is not True
        or combined["cross_family_equal_budget_claim"] is not False
    ):
        raise ValueError("Unsupported combined policy.")
    rates = [item["coverage_percent"] for item in config["levels"].values()]
    if (
        not rates
        or any(type(rate) is not int or not 0 <= rate <= 100 for rate in rates)
        or rates != sorted(set(rates))
    ):
        raise ValueError("Levels must have distinct ascending integer percentages in [0, 100].")
    seeds = config["evaluation_seeds"]
    if not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be distinct integers.")


class WordNoise:
    """Keep transformation rules independent of datasets, tokenizers and models."""

    def __init__(self, config: dict):
        validate_config(config)
        self.config = config
        self.case_map = str.maketrans(config["families"]["case_loss"]["turkish_overrides"])
        self.fold_map = str.maketrans(config["families"]["diacritic_loss"]["mapping"])
        self.alphabet = frozenset(config["families"]["typo_swap"]["alphabet"])
        self.minimum_length = config["families"]["typo_swap"]["minimum_word_length"]

    def lower(self, word: str) -> str:
        return word.translate(self.case_map).lower()

    def fold(self, word: str) -> str:
        return word.translate(self.fold_map)

    def swap_candidates(self, word: str) -> list[int]:
        """Return left indices of unequal interior adjacent pairs."""
        if len(word) < self.minimum_length or any(char not in self.alphabet for char in word):
            return []
        return [i for i in range(1, len(word) - 2) if word[i] != word[i + 1]]

    def eligible(self, word: str, family: str) -> bool:
        if family == "case_loss":
            return self.lower(word) != word
        if family == "diacritic_loss":
            return self.fold(word) != word
        if family == "typo_swap":
            return bool(self.swap_candidates(word))
        raise ValueError(f"Unknown noise family: {family}")

    def transform(self, word: str, family: str, *, swap_index: int | None = None) -> str:
        if family == "case_loss":
            return self.lower(word)
        if family == "diacritic_loss":
            return self.fold(word)
        if family == "typo_swap" and swap_index is not None:
            if not 1 <= swap_index < len(word) - 2:
                raise ValueError("Swap index is outside the interior.")
            i = swap_index
            return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        raise ValueError(f"Unknown family or missing swap location: {family}")


@dataclass(frozen=True)
class NoisePlans:
    """Ranked clean positions; every severity uses a prefix of the same ranking."""

    version: str
    split: str
    seed: int
    total_words: int
    source_words_sha256: str
    positions: Mapping[str, tuple[Position, ...]]


def source_words_digest(records: pd.DataFrame) -> str:
    """Bind a plan to exact clean words and IDs, excluding labels from selection."""
    digest = hashlib.sha256()
    for row_id, tokens in records[["row_id", "tokens"]].itertuples(index=False, name=None):
        digest.update(stable_digest([row_id, tokens]))
    return digest.hexdigest()


def build_plans(records: pd.DataFrame, noise: WordNoise, *, split: str, seed: int) -> NoisePlans:
    """Rank eligible positions without reading tags or using traversal-dependent randomness."""
    words = records[["row_id", "tokens"]]
    if words.empty or words["row_id"].duplicated().any():
        raise ValueError("Plans require nonempty records with unique row IDs.")
    version = noise.config["version"]
    ranked = {family: [] for family in FAMILIES}
    total = 0
    for row_id, tokens in words.itertuples(index=False, name=None):
        for word_index, word in enumerate(tokens):
            total += 1
            for family in FAMILIES:
                if noise.eligible(word, family):
                    key = [version, seed, split, row_id, word_index, family, "select"]
                    ranked[family].append((stable_digest(key), row_id, word_index))
    positions = {
        family: tuple((row_id, index) for _, row_id, index in sorted(pool))
        for family, pool in ranked.items()
    }
    return NoisePlans(version, split, seed, total, source_words_digest(records), positions)


def apply_noise(
    records: pd.DataFrame,
    noise: WordNoise,
    plans: NoisePlans,
    *,
    families: Sequence[str],
    coverage_percent: int | None,
) -> tuple[pd.DataFrame, list[dict]]:
    """Materialize a copy and stage edit log. None means all eligible words.

    Eligibility, swap locations and masks come from clean words even in combined
    conditions. The order of families determines composition, with no resampling.
    """
    if not families or len(set(families)) != len(families) or set(families) - set(FAMILIES):
        raise ValueError("Choose distinct supported noise families.")
    if coverage_percent is None and "typo_swap" in families:
        raise ValueError("A full-swap stress condition is not specified.")
    if coverage_percent is not None and (
        type(coverage_percent) is not int or not 0 <= coverage_percent <= 100
    ):
        raise ValueError("Coverage must be an integer percentage or None for full loss.")
    if plans.version != noise.config["version"]:
        raise ValueError("Plan and operator versions differ.")
    if source_words_digest(records) != plans.source_words_sha256:
        raise ValueError("Plan belongs to different clean words or row IDs.")
    columns = records[["row_id", "tokens", "tags"]]
    rows = [
        {"row_id": row_id, "tokens": list(tokens), "tags": list(tags)}
        for row_id, tokens, tags in columns.itertuples(index=False, name=None)
    ]
    row_lookup = {row["row_id"]: row for row in rows}
    clean_lookup = dict(records[["row_id", "tokens"]].itertuples(index=False, name=None))
    if len(row_lookup) != len(rows) or sum(len(row["tokens"]) for row in rows) != plans.total_words:
        raise ValueError("Records differ from the plan's unique IDs/word count.")
    edits = []
    for family in families:
        pool = plans.positions[family]
        count = (
            len(pool) if coverage_percent is None else plans.total_words * coverage_percent // 100
        )
        if count > len(pool):
            raise ValueError(
                f"{family}: requested {count} changed words; only {len(pool)} eligible."
            )
        selected = sorted(pool) if coverage_percent is None else pool[:count]
        for row_id, word_index in selected:
            clean = clean_lookup[row_id][word_index]
            current = row_lookup[row_id]["tokens"][word_index]
            swap_index = None
            if family == "typo_swap":
                key = [
                    plans.version,
                    plans.seed,
                    plans.split,
                    row_id,
                    word_index,
                    family,
                    "swap_position",
                ]
                swap_index = min(
                    noise.swap_candidates(clean),
                    key=lambda index: (stable_digest([*key, index]), index),
                )
            result = noise.transform(current, family, swap_index=swap_index)
            row_lookup[row_id]["tokens"][word_index] = result
            edits.append(
                {
                    "row_id": row_id,
                    "word_index": word_index,
                    "family": family,
                    "swap_index": swap_index,
                    "before": current,
                    "after": result,
                }
            )
    return pd.DataFrame(rows, columns=["row_id", "tokens", "tags"]), edits
