"""Data-only normalization-corpus diagnostics; never load or select a model."""

import hashlib
import itertools
import unicodedata
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

from turkish_punctuation.noise import FAMILIES, WordNoise


@dataclass(frozen=True)
class Slot:
    sentence_id: int
    word_index: int
    original: str
    normalized: str
    boundary: str | None


def read_multilexnorm(text: str) -> tuple[list[list[Slot]], dict]:
    """Preserve empty targets and mark every slot involved in a split or merge."""
    sentences, current = [], []
    empty_lines = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if line == "":
            empty_lines += 1
            if current:
                sentences.append(current)
                current = []
            continue
        fields = line.split("\t")
        if (len(fields) != 2 or not fields[0]
                or any(c.isspace() for c in fields[0])):
            raise ValueError(f"Invalid two-column token record at line {line_number}")
        if any(not unicodedata.is_normalized("NFC", value) for value in fields):
            raise ValueError(f"Non-NFC input at line {line_number}; no normalization applied")
        current.append(tuple(fields))
    if current:
        sentences.append(current)
    if not sentences:
        raise ValueError("Empty corpus")
    rows = []
    for sentence_id, sentence in enumerate(sentences):
        merge_anchors = set()
        for index, (_, target) in enumerate(sentence):
            if target == "":
                anchor = index - 1
                while anchor >= 0 and sentence[anchor][1] == "":
                    anchor -= 1
                if anchor < 0:
                    raise ValueError(f"Orphan merge continuation in sentence {sentence_id}")
                merge_anchors.add(anchor)
        row = []
        for index, (original, target) in enumerate(sentence):
            if target == "":
                boundary = "merge_continuation"
            elif len(target.split()) != 1 or target != target.strip():
                boundary = "split_and_merge_anchor" if index in merge_anchors else "split"
            elif index in merge_anchors:
                boundary = "merge_anchor"
            else:
                boundary = None
            row.append(Slot(sentence_id, index, original, target, boundary))
        rows.append(row)
    return rows, {"empty_physical_lines": empty_lines, "nfc_violations": 0,
                  "malformed_records": 0, "orphan_merge_continuations": 0}


class OperatorReachability:
    """An upper bound over legal word edits, not a simulation of sampled plans."""

    def __init__(self, noise: WordNoise):
        self.noise = noise

    @lru_cache(maxsize=None)
    def variants(self, corrected: str) -> dict[str, frozenset[str]]:
        eligible = [f for f in self.noise.config["combined"]["apply_order"]
                    if self.noise.eligible(corrected, f)]
        outputs = {}
        for count in range(1, len(eligible) + 1):
            for subset in itertools.combinations(eligible, count):
                values = [corrected]
                for family in subset:
                    if family == "typo_swap":
                        values = [self.noise.transform(corrected, family, swap_index=i)
                                  for i in self.noise.swap_candidates(corrected)]
                    else:
                        values = [self.noise.transform(value, family) for value in values]
                outputs["+".join(subset)] = frozenset(values)
        return outputs

    def matches(self, corrected: str, observed: str) -> list[str]:
        if corrected == observed:
            return []
        return [name for name, values in self.variants(corrected).items() if observed in values]


def edit_distance(left: str, right: str) -> int:
    """Unit-cost Levenshtein distance; do not infer a unique human edit sequence."""
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def mechanism_shapes(corrected: str, observed: str, noise: WordNoise,
                     minimum_length: int) -> list[str]:
    """Nonexclusive case-neutral shapes on alphabetic, single-word differences."""
    if (len(corrected) < minimum_length or corrected == observed or not observed
            or any(c not in noise.alphabet for c in corrected + observed)):
        return []
    shapes = []
    if len(corrected) == len(observed):
        if sum(a != b for a, b in zip(corrected, observed, strict=True)) == 1:
            shapes.append("one_character_substitution")
        if (observed != noise.fold(corrected)
                and all(b in {a, a.translate(noise.fold_map)}
                        for a, b in zip(corrected, observed, strict=True))):
            shapes.append("partial_turkish_folding")
    if len(observed) == len(corrected) - 1 and any(
        corrected[:i] + corrected[i + 1:] == observed for i in range(1, len(corrected) - 1)
    ):
        shapes.append("one_internal_character_omission")
    if len(observed) == len(corrected) + 1 and any(
        observed[:i] + observed[i + 1:] == corrected for i in range(len(observed))
    ):
        shapes.append("one_character_insertion")
    clean_runs = [(c, len(list(group))) for c, group in itertools.groupby(corrected)]
    noisy_runs = [(c, len(list(group))) for c, group in itertools.groupby(observed)]
    if (len(clean_runs) == len(noisy_runs) and len(observed) > len(corrected)
            and all(a == b and n <= m
                    for (a, n), (b, m) in zip(clean_runs, noisy_runs, strict=True))):
        shapes.append("character_run_elongation")
    return shapes


def _counts(counter: Counter) -> dict:
    return {str(key): value for key, value in sorted(counter.items(), key=lambda x: str(x[0]))}


def _view(pairs: list[tuple[str, str, str | None]], reach: OperatorReachability,
          minimum_length: int) -> dict:
    partition, boundary_counts, singles, combinations = Counter(), Counter(), Counter(), Counter()
    shapes_all, shapes_unreachable, distances, length_changes = (Counter() for _ in range(4))
    any_shape_unreachable = case_only = 0
    for observed, corrected, boundary in pairs:
        if observed == corrected:
            partition["unchanged"] += 1
            continue
        if boundary:
            partition["boundary_excluded_changed"] += 1
            boundary_counts[boundary] += 1
            continue
        matches = reach.matches(corrected, observed)
        single_matches = [name for name in matches if name in FAMILIES]
        if single_matches:
            partition["single_operator_reachable"] += 1
        elif matches:
            partition["combination_only_reachable"] += 1
        else:
            partition["unreachable"] += 1
        singles.update(single_matches)
        combinations.update(name for name in matches if name not in FAMILIES)
        case_only += reach.noise.lower(observed) == reach.noise.lower(corrected)
        shapes = mechanism_shapes(corrected, observed, reach.noise, minimum_length)
        shapes_all.update(shapes)
        if not matches:
            shapes_unreachable.update(shapes)
            any_shape_unreachable += bool(shapes)
        distances[edit_distance(corrected, observed)] += 1
        length_changes[len(observed) - len(corrected)] += 1
    total = len(pairs)
    if sum(partition.values()) != total:
        raise ValueError("View partition does not conserve input slots")
    eligible = sum(partition[key] for key in
                   ["single_operator_reachable", "combination_only_reachable", "unreachable"])
    reachable = partition["single_operator_reachable"] + partition["combination_only_reachable"]
    if sum(distances.values()) != eligible:
        raise ValueError("Distance denominator does not match one-word changes")
    return {
        "slots_or_pairs": total, "partition": _counts(partition),
        "changed": total - partition["unchanged"], "eligible_one_word_changes": eligible,
        "reachable_one_word_changes": reachable,
        "reachable_fraction_of_eligible_changes": reachable / eligible if eligible else None,
        "boundary_exclusions": _counts(boundary_counts),
        "single_operator_memberships_nonexclusive": {f: singles[f] for f in FAMILIES},
        "combination_memberships_nonexclusive": _counts(combinations),
        "case_only_changed_one_word_pairs": case_only,
        "shape_memberships_nonexclusive": _counts(shapes_all),
        "shape_memberships_among_unreachable_nonexclusive": _counts(shapes_unreachable),
        "unreachable_with_any_listed_shape": any_shape_unreachable,
        "levenshtein_distance_histogram": _counts(distances),
        "observed_minus_corrected_length_histogram": _counts(length_changes),
    }


def _casing(words: list[str]) -> dict:
    counts, punctuation = Counter(), Counter()
    for word in words:
        punctuation.update(c for c in word if unicodedata.category(c).startswith("P"))
        if word.isalpha():
            if word.islower():
                counts["lowercase"] += 1
            elif word.isupper():
                counts["uppercase"] += 1
            elif word.istitle():
                counts["titlecase"] += 1
            else:
                counts["mixed_or_uncased"] += 1
    return {"alphabetic_slots": sum(counts.values()), "casing": _counts(counts),
            "punctuation_glyph_counts_not_gold": _counts(punctuation)}


def audit_rows(rows: list[list[Slot]], noise: WordNoise, minimum_length: int) -> tuple[dict, list]:
    reach = OperatorReachability(noise)
    slots = [slot for row in rows for slot in row]
    pairs = [(slot.original, slot.normalized, slot.boundary) for slot in slots]
    neutral = [(noise.lower(a), noise.lower(b), boundary) for a, b, boundary in pairs]
    structural = Counter(slot.boundary for slot in slots if slot.boundary)
    one_word = [slot for slot in slots if slot.boundary is None]
    clean_case_eligible = [s for s in one_word if noise.eligible(s.normalized, "case_loss")]
    original_lowercases = [s for s in clean_case_eligible
                           if s.original == noise.lower(s.normalized)]
    changed_case_only = [s for s in one_word if s.original != s.normalized
                         and noise.lower(s.original) == noise.lower(s.normalized)]
    row_fingerprints = Counter(tuple((s.original, s.normalized) for s in row) for row in rows)
    views = {}
    for name, values in [("exact", pairs), ("case_neutral", neutral)]:
        views[name] = {
            "occurrence_weighted": _view(values, reach, minimum_length),
            "unique_pair_weighted": _view(sorted(set(values), key=repr), reach, minimum_length),
        }
    trace = []
    for slot in slots:
        exact_matches = [] if slot.boundary else reach.matches(slot.normalized, slot.original)
        target, observed = noise.lower(slot.normalized), noise.lower(slot.original)
        neutral_matches = [] if slot.boundary else reach.matches(target, observed)
        trace.append({
            "sentence_id": slot.sentence_id, "word_index": slot.word_index,
            "pair_sha256": hashlib.sha256(
                (slot.original + "\0" + slot.normalized).encode("utf-8")).hexdigest(),
            "boundary": slot.boundary, "changed_exact": slot.original != slot.normalized,
            "changed_case_neutral": target != observed,
            "exact_operators": exact_matches, "case_neutral_operators": neutral_matches,
            "case_neutral_shapes": [] if slot.boundary else mechanism_shapes(
                target, observed, noise, minimum_length),
        })
    if len(trace) != len(slots):
        raise ValueError("Private trace lost source slots")
    return {
        "sentences": len(rows), "source_slots": len(slots),
        "source_slots_with_letters": sum(any(c.isalpha() for c in s.original) for s in slots),
        "one_word_slots": len(one_word),
        "one_word_corrected_slots_with_letters": sum(
            any(c.isalpha() for c in s.normalized) for s in one_word),
        "one_word_alphabetic_pairs_in_frozen_alphabet": sum(
            bool(s.normalized + s.original)
            and all(c in noise.alphabet for c in s.normalized + s.original) for s in one_word),
        "structural_exclusion_slots": sum(structural.values()),
        "structural_exclusion_types": _counts(structural),
        "unique_original_forms": len({s.original for s in slots}),
        "unique_normalized_fields": len({s.normalized for s in slots}),
        "duplicate_sentence_excess": sum(count - 1 for count in row_fingerprints.values()),
        "original_casing": _casing([s.original for s in slots]),
        "normalized_casing": _casing([s.normalized for s in slots]),
        "case_candidates": {
            "corrected_case_loss_eligible_one_word_slots": len(clean_case_eligible),
            "observed_exact_lowercase_of_corrected": len(original_lowercases),
            "fraction_of_corrected_case_eligible_slots": (
                len(original_lowercases) / len(clean_case_eligible) if clean_case_eligible else None),
            "all_case_only_changed_one_word_slots": len(changed_case_only),
            "other_case_only_changes": len(changed_case_only) - len(original_lowercases),
            "population_prevalence_estimated": False,
        },
        "views": views,
        "interpretation": {
            "coverage_is_word_reachability_upper_bound": True,
            "profile_is_conditional_on_this_corpus_and_eligibility": True,
            "punctuation_gold_available_for_restoration": False,
            "model_performance_measured": False,
            "raw_text_included": False,
        },
    }, trace
