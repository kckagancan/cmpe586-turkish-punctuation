"""Describe saved corpus shapes, their density and amplitude without model calls."""

import hashlib
import itertools
import json
import math
from collections import Counter

import numpy as np

from turkish_punctuation.corpus_audit import edit_distance

FAMILIES = {
    "elongation": "character_run_elongation",
    "internal_omission": "one_internal_character_omission",
    "partial_folding": "partial_turkish_folding",
}
VOWELS = frozenset("aeıioöuü")


def histogram(values):
    return {str(k): v for k, v in sorted(Counter(values).items())}


def distribution(values, quantiles):
    if not values:
        return {"n": 0, "mean": None, "minimum": None, "maximum": None,
                "quantiles": {}, "histogram": {}}
    return {
        "n": len(values), "mean": float(np.mean(values)),
        "minimum": float(min(values)), "maximum": float(max(values)),
        "quantiles": {f"p{q * 100:g}": float(np.quantile(values, q, method="inverted_cdf"))
                      for q in quantiles},
        "histogram": histogram(values),
    }


def intensity_bins(values):
    lower, upper = (float(np.quantile(values, q, method="inverted_cdf")) for q in [.25, .75])
    counts = {"lower": sum(x <= lower for x in values),
              "middle": sum(lower < x <= upper for x in values),
              "upper": sum(x > upper for x in values)}
    if sum(counts.values()) != len(values):
        raise ValueError("Intensity bins do not conserve observations")
    return {"q25": lower, "q75": upper, "counts": counts,
            "human_severity_labels": False, "three_nonempty_bins": all(counts.values())}


def runs(word):
    result, start = [], 0
    for char, group in itertools.groupby(word):
        length = len(list(group))
        result.append((char, length, start))
        start += length
    return result


def describe_elongation(clean, observed):
    left, right = runs(clean), runs(observed)
    if (len(left) != len(right) or len(observed) <= len(clean)
            or any(a != b or n > m for (a, n, _), (b, m, _) in zip(left, right, strict=True))):
        raise ValueError("Saved elongation does not satisfy its run-length definition")
    changed = []
    for (char, n, start), (_, m, _) in zip(left, right, strict=True):
        if m > n:
            location = ("initial" if start == 0 else
                        "final" if start + n == len(clean) else "interior")
            changed.append({"letter": char, "extra": m - n, "location": location,
                            "relative_run_end": (start + n - 1) / (len(clean) - 1)})
    return {"extra_characters": len(observed) - len(clean), "changed_runs": len(changed),
            "extra_over_corrected_length": (len(observed) - len(clean)) / len(clean),
            "run_details": changed}


def describe_omission(clean, observed):
    indices = [i for i in range(1, len(clean) - 1)
               if clean[:i] + clean[i + 1:] == observed]
    if len(observed) != len(clean) - 1 or not indices:
        raise ValueError("Saved omission has no legal interior deletion alignment")
    letters = {clean[i] for i in indices}
    if len(letters) != 1:
        raise ValueError("Ambiguous deletion changed the omitted character identity")
    return {"candidate_indices": indices, "candidate_count": len(indices),
            "letter": next(iter(letters)), "inverse_corrected_length": 1 / len(clean),
            "unique_relative_position": indices[0] / (len(clean) - 1)
            if len(indices) == 1 else None}


def describe_partial_folding(clean, observed, noise):
    if len(clean) != len(observed):
        raise ValueError("Partial folding changed length")
    available, changed = [], []
    for i, (a, b) in enumerate(zip(clean, observed, strict=True)):
        folded = a.translate(noise.fold_map)
        if b not in {a, folded}:
            raise ValueError("Partial folding contains a non-folding substitution")
        if a != folded:
            available.append(a)
        if a != b:
            changed.append({"mapping": a + "->" + b, "relative_position": i / (len(clean) - 1)})
    if not 0 < len(changed) < len(available):
        raise ValueError("Partial folding must change a nonempty proper subset")
    return {"available": len(available), "folded": len(changed),
            "retained": len(available) - len(changed),
            "folded_fraction": len(changed) / len(available),
            "available_letters": available, "changed_positions": changed}


def make_features(rows, trace, noise, minimum_length):
    lookup = {(item["sentence_id"], item["word_index"]): item for item in trace}
    if len(lookup) != len(trace):
        raise ValueError("Duplicate IDs in saved audit trace")
    features = []
    for row in rows:
        for slot in row:
            item = lookup.pop((slot.sentence_id, slot.word_index))
            pair_hash = hashlib.sha256(
                (slot.original + "\0" + slot.normalized).encode()).hexdigest()
            clean, observed = noise.lower(slot.normalized), noise.lower(slot.original)
            if (item["pair_sha256"] != pair_hash or item["boundary"] != slot.boundary
                    or item["changed_exact"] != (slot.original != slot.normalized)
                    or item["changed_case_neutral"] != (clean != observed)):
                raise ValueError("Saved audit/source row reconciliation failed")
            reason = ("structural" if slot.boundary else
                      "corrected_length_below_minimum" if len(clean) < minimum_length else
                      "nonalphabetic_pair" if any(c not in noise.alphabet for c in clean + observed)
                      else "eligible")
            eligible = reason == "eligible"
            foldable = sum(c != c.translate(noise.fold_map) for c in clean) if eligible else 0
            flags = {f: shape in item["case_neutral_shapes"] for f, shape in FAMILIES.items()}
            if any(flags.values()) and (not eligible or item["case_neutral_operators"]):
                raise ValueError("New shape eligibility/unreachability disagrees with saved audit")
            details = {}
            if flags["elongation"]:
                details["elongation"] = describe_elongation(clean, observed)
            if flags["internal_omission"]:
                details["internal_omission"] = describe_omission(clean, observed)
            if flags["partial_folding"]:
                details["partial_folding"] = describe_partial_folding(clean, observed, noise)
            if sum(flags.values()) > 1:
                raise ValueError("The three configured shapes should be length-disjoint")
            features.append({
                "sentence_id": slot.sentence_id, "word_index": slot.word_index,
                "pair_sha256": pair_hash,
                "neutral_pair_sha256": hashlib.sha256(json.dumps(
                    [observed, clean, slot.boundary], ensure_ascii=False).encode()).hexdigest(),
                "reason": reason, "corrected_length": len(clean), "observed_length": len(observed),
                "changed_exact": item["changed_exact"], "changed_neutral": clean != observed,
                "old_neutral_operators": item["case_neutral_operators"],
                "old_exact_operators": item["exact_operators"],
                "partial_opportunity": eligible and foldable >= 2,
                "full_fold_opportunity": eligible and foldable > 0,
                "literal_full_fold": eligible and clean != observed and observed == noise.fold(clean),
                "families": flags, "details": details,
                "case_neutral_edit_distance": edit_distance(clean, observed) if details else None,
                "exact_edit_distance": edit_distance(slot.normalized, slot.original)
                if details else None,
            })
    if lookup:
        raise ValueError("Saved audit trace contains extra IDs")
    return features


def sentence_features(features, n_sentences):
    result = []
    for sentence_id in range(n_sentences):
        group = [x for x in features if x["sentence_id"] == sentence_id]
        result.append({
            "sentence_id": sentence_id, "source_slots": len(group),
            "exact_changed": sum(x["changed_exact"] for x in group),
            "base_opportunities": sum(x["reason"] == "eligible" for x in group),
            "partial_opportunities": sum(x["partial_opportunity"] for x in group),
            "family_hits": {f: sum(x["families"][f] for x in group) for f in FAMILIES},
            "old_exact_reachable_hits": sum(bool(x["old_exact_operators"]) for x in group),
            "old_neutral_reachable_hits": sum(bool(x["old_neutral_operators"]) for x in group),
        })
    return result


def bootstrap_rates(sentences, config):
    settings = config["bootstrap"]
    if (settings["generator"] != "PCG64" or settings["resampling_unit"] != "whole_sentence"
            or settings["interval_quantile_method"] != "linear"):
        raise ValueError("Unsupported bootstrap specification")
    columns = []
    for family in FAMILIES:
        key = "partial_opportunities" if family == "partial_folding" else "base_opportunities"
        columns += [[s["family_hits"][family] for s in sentences], [s[key] for s in sentences]]
    array = np.array(columns, dtype=np.int64).T
    generator = np.random.Generator(np.random.PCG64(settings["seed"]))
    estimates = []
    for offset in range(0, settings["replicates"], settings["batch_size"]):
        size = min(settings["batch_size"], settings["replicates"] - offset)
        indices = generator.integers(0, len(sentences), size=(size, len(sentences)))
        sampled = array[indices]
        sums = sampled.sum(axis=1)
        rates = np.divide(sums[:, ::2], sums[:, 1::2],
                          out=np.full((size, 3), np.nan), where=sums[:, 1::2] != 0)
        presence = (sampled[:, :, ::2] > 0).mean(axis=1)
        estimates.append(np.concatenate([rates, presence], axis=1))
    values = np.concatenate(estimates)
    result = {}
    for i, family in enumerate(FAMILIES):
        result[family] = {}
        for name, index in [("pooled_opportunity_rate", i), ("sentence_presence_rate", i + 3)]:
            valid = values[:, index][np.isfinite(values[:, index])]
            result[family][name] = {
                "interval": np.quantile(valid, settings["interval_quantiles"],
                                        method="linear").tolist(),
                "defined_replicates": len(valid), "undefined_replicates": len(values) - len(valid),
            }
    return result, values


def family_profile(family, features, sentences, quantiles):
    eligible = [x for x in features if x["partial_opportunity"]] if family == "partial_folding" \
        else [x for x in features if x["reason"] == "eligible"]
    matched = [x for x in eligible if x["families"][family]]
    n = len(matched)
    opportunities = [s["partial_opportunities"] if family == "partial_folding"
                     else s["base_opportunities"] for s in sentences]
    hits = [s["family_hits"][family] for s in sentences]
    rates = [h / d for h, d in zip(hits, opportunities, strict=True) if d]
    positive_rates = [h / d for h, d in zip(hits, opportunities, strict=True) if h and d]
    concentration = Counter(x["neutral_pair_sha256"] for x in matched)
    loo = [(n - h) / (len(eligible) - d)
           for h, d in zip(hits, opportunities, strict=True) if len(eligible) > d]
    result = {
        "occurrences": n, "opportunities": len(eligible), "opportunity_fraction": n / len(eligible),
        "fraction_all_source_slots": n / len(features),
        "fraction_one_to_one_slots": n / sum(x["reason"] != "structural" for x in features),
        "affected_sentences": sum(h > 0 for h in hits),
        "fraction_all_sentences": sum(h > 0 for h in hits) / len(sentences),
        "sentences_with_opportunities": sum(d > 0 for d in opportunities),
        "sentences_without_opportunities": sum(d == 0 for d in opportunities),
        "zero_hit_sentences_with_opportunities": sum(h == 0 and d > 0
                                                      for h, d in zip(hits, opportunities, strict=True)),
        "hits_per_sentence": distribution(hits, quantiles),
        "sentence_rates_including_zeros": distribution(rates, quantiles),
        "affected_sentence_rates": distribution(positive_rates, quantiles),
        "largest_sentence_share_of_hits": max(hits) / n,
        "top_five_sentence_share_of_hits": sum(sorted(hits, reverse=True)[:5]) / n,
        "largest_neutral_pair_share_of_hits": max(concentration.values()) / n,
        "leave_one_sentence_out_pooled_rate_range": [min(loo), max(loo)],
        "corrected_length": distribution([x["corrected_length"] for x in matched], quantiles),
        "case_neutral_edit_distance": distribution(
            [x["case_neutral_edit_distance"] for x in matched], quantiles),
        "exact_edit_distance": distribution([x["exact_edit_distance"] for x in matched], quantiles),
        "unique_pair_sensitivity": {}, "length_strata": [],
    }
    for view, key in [("exact", "pair_sha256"), ("case_neutral", "neutral_pair_sha256")]:
        pairs = {x[key]: x for x in eligible}
        count = sum(x["families"][family] for x in pairs.values())
        result["unique_pair_sensitivity"][view] = {
            "matches": count, "opportunities": len(pairs), "fraction": count / len(pairs)}
    for lo, hi in [(4, 5), (6, 8), (9, 12), (13, math.inf)]:
        pool = [x for x in eligible if lo <= x["corrected_length"] <= hi]
        count = sum(x["families"][family] for x in pool)
        result["length_strata"].append({"minimum": lo, "maximum": None if hi == math.inf else hi,
                                         "opportunities": len(pool), "matches": count,
                                         "fraction": count / len(pool) if pool else None})
    details = [x["details"][family] for x in matched]
    if family == "elongation":
        run_details = [run for d in details for run in d["run_details"]]
        values = [d["extra_characters"] for d in details]
        result["amplitude"] = {
            "extra_characters": distribution(values, quantiles),
            "extra_over_corrected_length": distribution(
                [d["extra_over_corrected_length"] for d in details], quantiles),
            "changed_runs": distribution([d["changed_runs"] for d in details], quantiles),
            "extra_per_changed_run": distribution([d["extra"] for d in run_details], quantiles),
            "ordered_extra_count_patterns": histogram(
                ["+".join(str(run["extra"]) for run in d["run_details"]) for d in details]),
            "single_run_occurrences": sum(d["changed_runs"] == 1 for d in details),
            "extra_distribution_single_run_only": distribution(
                [d["extra_characters"] for d in details if d["changed_runs"] == 1], quantiles),
            "run_locations": histogram([d["location"] for d in run_details]),
            "repeated_letters_per_changed_run": histogram([d["letter"] for d in run_details]),
            "vowel_runs": sum(d["letter"] in VOWELS for d in run_details),
            "total_added_characters": sum(values), "top_five_word_share_of_extra_characters":
                sum(sorted(values, reverse=True)[:5]) / sum(values),
            "empirical_intensity_bins": intensity_bins(values),
        }
    elif family == "internal_omission":
        positions = [d["unique_relative_position"] for d in details
                     if d["unique_relative_position"] is not None]
        values = [d["inverse_corrected_length"] for d in details]
        result["amplitude"] = {
            "deleted_characters_per_match": 1, "inverse_corrected_length": distribution(values, quantiles),
            "candidate_alignment_counts": histogram([d["candidate_count"] for d in details]),
            "ambiguous_alignment_occurrences": sum(d["candidate_count"] > 1 for d in details),
            "unique_alignment_relative_positions": distribution(positions, quantiles),
            "omitted_letters": histogram([d["letter"] for d in details]),
            "vowel_omissions": sum(d["letter"] in VOWELS for d in details),
            "empirical_intensity_bins": intensity_bins(values),
        }
    else:
        available = [letter for d in details for letter in d["available_letters"]]
        mappings = [p["mapping"] for d in details for p in d["changed_positions"]]
        values = [d["folded_fraction"] for d in details]
        result["amplitude"] = {
            "folded_occurrences": distribution([d["folded"] for d in details], quantiles),
            "available_occurrences": distribution([d["available"] for d in details], quantiles),
            "retained_occurrences": distribution([d["retained"] for d in details], quantiles),
            "folded_fraction": distribution(values, quantiles),
            "folded_available_joint": histogram([f'{d["folded"]}/{d["available"]}' for d in details]),
            "available_letters_within_matched_words": histogram(available),
            "changed_mappings": histogram(mappings),
            "changed_relative_positions": distribution(
                [p["relative_position"] for d in details for p in d["changed_positions"]], quantiles),
            "total_folded_characters": len(mappings),
            "total_available_characters_within_matched_words": len(available),
            "empirical_intensity_bins": intensity_bins(values),
        }
    return result


def analyze_depth(rows, trace, noise, config, audit_summary):
    if config["families"] != FAMILIES or config["quantile_method"] != "inverted_cdf":
        raise ValueError("Unsupported family mapping or quantiles")
    features = make_features(rows, trace, noise, config["minimum_corrected_length"])
    sentences = sentence_features(features, len(rows))
    quantiles = config["descriptive_quantiles"]
    profiles = {f: family_profile(f, features, sentences, quantiles) for f in FAMILIES}
    if {f: p["occurrences"] for f, p in profiles.items()} != config["expected_shape_occurrences"]:
        raise ValueError("Depth profile differs from the original saved shape counts")
    bootstrap, replicates = bootstrap_rates(sentences, config)
    for f in profiles:
        profiles[f]["sentence_cluster_bootstrap_stability"] = bootstrap[f]
    combinations = Counter("+".join(f for f in FAMILIES if s["family_hits"][f]) or "none"
                           for s in sentences)
    base_partition, partial_partition = Counter(), Counter()
    for x in features:
        if x["reason"] == "eligible":
            label = ("unchanged_case_neutral" if not x["changed_neutral"] else
                     next((f for f in FAMILIES if x["families"][f]), None) or
                     ("old_operator_reachable" if x["old_neutral_operators"] else "other_unreachable"))
            base_partition[label] += 1
        if x["partial_opportunity"]:
            label = ("unchanged_case_neutral" if not x["changed_neutral"] else
                     "full_folding" if x["literal_full_fold"] else
                     "partial_folding" if x["families"]["partial_folding"] else "other_change")
            partial_partition[label] += 1
    union = sum(any(x["families"].values()) for x in features)
    previous = audit_summary["views"]["case_neutral"]["occurrence_weighted"]
    result = {
        "scope": "data_only_train_split_depth_extension_no_model_scores",
        "sentences": len(rows), "source_slots": len(features),
        "eligibility_partition": dict(sorted(Counter(x["reason"] for x in features).items())),
        "eligible_base_partition": dict(sorted(base_partition.items())),
        "multi_foldable_opportunity_partition": dict(sorted(partial_partition.items())),
        "families": profiles,
        "joint": {
            "word_level_union": union, "word_level_overlap_excess":
                sum(p["occurrences"] for p in profiles.values()) - union,
            "affected_sentences_any_new_shape": sum(k != "none" for s in sentences
                for k in ["+".join(f for f in FAMILIES if s["family_hits"][f]) or "none"]),
            "sentence_membership_combinations": dict(sorted(combinations.items())),
            "new_shape_sentences_also_containing_old_exact_reachable_pattern": sum(
                any(s["family_hits"].values()) and s["old_exact_reachable_hits"] > 0 for s in sentences),
            "new_shape_sentences_also_containing_old_neutral_reachable_pattern": sum(
                any(s["family_hits"].values()) and s["old_neutral_reachable_hits"] > 0 for s in sentences),
            "fraction_saved_unreachable_case_neutral_changes_with_three_shapes":
                union / previous["partition"]["unreachable"],
            "saved_unreachable_case_neutral_changes": previous["partition"]["unreachable"],
        },
        "sentence_context": {
            "source_slots_per_sentence": distribution([s["source_slots"] for s in sentences], quantiles),
            "exact_changed_fraction_all_source_slots": distribution(
                [s["exact_changed"] / s["source_slots"] for s in sentences], quantiles),
            "base_opportunities_per_sentence": distribution(
                [s["base_opportunities"] for s in sentences], quantiles),
        },
        "bootstrap_specification": config["bootstrap"],
        "quantile_method": config["quantile_method"],
        "raw_text_in_public_output": False, "model_calls": False, "source_test_read": False,
        "generated_corruptions": False, "evaluation_recipe_frozen": False,
    }
    if (sum(result["eligibility_partition"].values()) != len(features)
            or sum(base_partition.values()) != result["eligibility_partition"]["eligible"]
            or sum(combinations.values()) != len(rows)
            or sum(partial_partition.values()) != profiles["partial_folding"]["opportunities"]):
        raise ValueError("Depth-analysis counters fail conservation")
    return result, features, sentences, replicates
