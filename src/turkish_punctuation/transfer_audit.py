"""Independent boundary/shape and exposure audit for variable-length transfer edits."""

import itertools
import unicodedata
from collections import Counter
from fractions import Fraction

from turkish_punctuation.data import LABELS
from turkish_punctuation.noise import source_words_digest, stable_digest
from turkish_punctuation.transfer_noise import CONTROL, FAMILIES


def run_lengths(word):
    return [(char, len(list(group))) for char, group in itertools.groupby(word)]


def audit_transfer_changes(clean, noisy, edits, noise, family=None, plan=None):
    if family not in (None, *FAMILIES, CONTROL):
        raise ValueError("Unsupported transfer audit family")
    if clean["row_id"].tolist() != noisy["row_id"].tolist():
        raise ValueError("Transfer changed row IDs/order")
    selected = {}
    for edit in edits:
        key = (edit["row_id"], edit["word_index"])
        if key in selected or edit["family"] != family:
            raise ValueError("Duplicate or wrong-family edit record")
        selected[key] = edit
    selector = "partial_folding" if family == CONTROL else family
    if plan is not None:
        if source_words_digest(clean) != plan["source_words_sha256"]:
            raise ValueError("Audited source differs from transfer plan")
        expected = plan["families"][selector]
        if set(selected) != set(map(tuple, expected["selected_positions"])):
            raise ValueError("Edit log selection differs from the exact planned mask")
    supports, edited_supports, changed_rows, histogram, donor_usage = Counter(), Counter(), 0, Counter(), Counter()
    words = changed = character_edits = added = removed = eligible = 0
    conditioned = extrapolated = 0
    placement, word_lengths, other = Counter(), Counter(), Counter()
    seen = set()
    original_rows = clean[["row_id", "tokens", "tags"]].itertuples(index=False, name=None)
    output_rows = noisy[["row_id", "tokens", "tags"]].itertuples(index=False, name=None)
    for (row_id, tokens, tags), (_, outputs, targets) in zip(original_rows, output_rows, strict=True):
        if tags != targets or len(tokens) != len(outputs):
            raise ValueError("Transfer changed word count/order or original targets")
        row_changed = False
        for index, (before, after, tag) in enumerate(zip(tokens, outputs, tags, strict=True)):
            position = (row_id, index)
            words += 1
            supports[tag] += 1
            if not after or not unicodedata.is_normalized("NFC", after):
                raise ValueError("Transfer produced an empty/non-NFC word")
            eligible += noise.eligible(before, selector) if selector else 0
            if position not in selected:
                if before != after:
                    raise ValueError("An unselected word changed")
                continue
            edit = selected[position]
            if edit["before"] != before or edit["after"] != after or before == after:
                raise ValueError("Edit record/source/output mismatch or selected noop")
            if not noise.eligible(before, selector) or any(c not in noise.original.alphabet for c in after):
                raise ValueError("Selected word is outside the alphabetic opportunity pool")
            parameters = edit["parameters"]
            if "template_id" in parameters:
                donor_usage[parameters["template_id"]] += 1
            word_lengths[len(before)] += 1
            if family == "elongation":
                left, right = run_lengths(noise.original.lower(before)), run_lengths(noise.original.lower(after))
                if (len(left) != len(right)
                        or any(a != b or n > m for (a, n), (b, m) in zip(left, right, strict=True))):
                    raise ValueError("Elongation changed the original run skeleton")
                indices = parameters["positions"]
                extras = parameters["extras"]
                if len(set(indices)) != len(indices) or any(n <= 0 for n in extras):
                    raise ValueError("Invalid elongation insertion positions/deltas")
                insertions = dict(zip(indices, extras, strict=True))
                reconstructed = "".join(c + c * insertions.get(i, 0) for i, c in enumerate(before))
                if reconstructed != after or len(after) - len(before) != sum(extras):
                    raise ValueError("Elongation does not preserve original surface characters")
                neutral = noise.original.lower(before)
                if any(i < 0 or i >= len(before) or (i < len(before)-1 and neutral[i] == neutral[i+1])
                       for i in indices):
                    raise ValueError("Elongation insertion is not at an original run end")
                amount = len(after) - len(before)
                added += amount
                character_edits += amount
                histogram[amount] += 1
                other[f"changed_runs:{len(indices)}"] += 1
                conditioned += parameters["feasible_template_conditioning"]
                for i in indices:
                    placement[str(Fraction(i, len(before)-1))] += 1
            elif family == "internal_omission":
                index_removed = parameters["position"]
                if (not 0 < index_removed < len(before)-1 or len(after) != len(before)-1
                        or before[:index_removed] + before[index_removed+1:] != after
                        or before[0] != after[0] or before[-1] != after[-1]):
                    raise ValueError("Omission is not exactly one original interior character")
                removed += 1
                character_edits += 1
                histogram[1] += 1
                placement[str(Fraction(index_removed, len(before)-1))] += 1
                other["omitted_letter:" + before[index_removed]] += 1
            else:
                if len(before) != len(after):
                    raise ValueError("Folding changed codepoint length")
                sites = noise.foldable(before)
                actual = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
                if (any(b != a.translate(noise.original.fold_map) for i, (a, b) in
                        enumerate(zip(before, after, strict=True)) if i in actual)
                        or not set(actual).issubset(sites) or actual != parameters["positions"]):
                    raise ValueError("Folding changed an unlisted character or edit index")
                if family == CONTROL:
                    if actual != sites or after != noise.original.fold(before):
                        raise ValueError("Matched control is not the original full-word folding operation")
                elif (not 0 < len(actual) < len(sites)
                      or parameters["mapped_folded"] != len(actual)
                      or parameters["target_foldable"] != len(sites)):
                    raise ValueError("Partial folding is not a nonempty proper subset")
                else:
                    extrapolated += parameters["unseen_foldable_count"]
                character_edits += len(actual)
                histogram[f"{len(actual)}/{len(sites)}"] += 1
                for i in actual:
                    placement[str(Fraction(i, len(before)-1))] += 1
            seen.add(position)
            changed += 1
            edited_supports[tag] += 1
            row_changed = True
        changed_rows += row_changed
    if seen != set(selected) or sum(supports.values()) != words:
        raise ValueError("Some selected positions/targets were not reconciled")
    if plan is not None and (eligible != expected["eligible_words"] or changed != expected["selected_words"]):
        raise ValueError("Realized eligible pool/budget differs from planned exact dose")
    return {
        "rows": len(clean), "words": words, "eligible_words": eligible,
        "changed_rows": changed_rows, "changed_words": changed,
        "changed_word_fraction_all": changed / words,
        "changed_fraction_of_opportunities": changed / eligible if eligible else None,
        "selected_positions_sha256": stable_digest(sorted(selected)).hex(),
        "label_support": {label: supports[label] for label in LABELS},
        "changed_label_support": {label: edited_supports[label] for label in LABELS},
        "declared_character_edit_count": character_edits, "added_characters": added,
        "removed_characters": removed, "net_codepoint_change": added - removed,
        "amplitude_histogram": {str(k): v for k, v in sorted(histogram.items(), key=lambda p: str(p[0]))},
        "template_usage": {str(k): v for k, v in sorted(donor_usage.items())},
        "changed_corrected_word_length_histogram": dict(sorted(word_lengths.items())),
        "relative_placement_histogram": dict(sorted(placement.items())),
        "additional_histogram": dict(sorted(other.items())),
        "elongation_feasible_template_conditioning_words": conditioned,
        "partial_unseen_foldable_count_words": extrapolated,
        "invariants_passed": True, "full_original_targets_preserved": True,
    }
