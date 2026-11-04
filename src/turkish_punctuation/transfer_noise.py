"""Versioned, empirically anchored transfer noise; never inspect punctuation labels."""

from fractions import Fraction

import pandas as pd

from turkish_punctuation.noise import WordNoise, source_words_digest, stable_digest

VERSION = "transfer-noise-v1"
FAMILIES = ("elongation", "internal_omission", "partial_folding")
CONTROL = "full_folding_matched"
RATES = {"elongation": (89, 4512), "internal_omission": (50, 4174),
         "partial_folding": (43, 617)}


def rational(pair):
    return Fraction(*pair)


def weighted_choice(items, key, weight="count"):
    total = sum(item[weight] for _, item in items)
    if total <= 0:
        raise ValueError("Empirical profile has no applicable positive weight")
    draw = int.from_bytes(stable_digest([*key, "profile_draw"]), "big") % total
    for identifier, item in items:
        draw -= item[weight]
        if draw < 0:
            return identifier, item
    raise RuntimeError("Weighted profile draw did not terminate")


def ordered_assignment(candidates, targets, scale, key):
    """Minimum-distance ordered matching; additive hash tie costs preserve DP optimality."""
    if len(candidates) < len(targets) or scale < 1:
        raise ValueError("Not enough distinct positions for the selected profile")
    states = [None] * (len(targets) + 1)
    states[0] = (Fraction(0), 0, ())
    for position in candidates:
        secondary = int.from_bytes(stable_digest([*key, "assignment_tie", position]), "big")
        for count in range(len(targets), 0, -1):
            previous = states[count - 1]
            if previous is None:
                continue
            candidate = (previous[0] + abs(Fraction(position, scale) - targets[count - 1]),
                         previous[1] + secondary, (*previous[2], position))
            if states[count] is None or candidate < states[count]:
                states[count] = candidate
    return list(states[-1][2])


class TransferNoise:
    def __init__(self, profile, original_recipe):
        self.profile = profile
        self.original = WordNoise(original_recipe)
        if profile["version"] != "transfer-morphology-profile-v1":
            raise ValueError("Unsupported morphology profile")
        if {f: (r["numerator"], r["denominator"]) for f, r in profile["rates"].items()} != RATES:
            raise ValueError("Operational corpus dose differs from this version")
        for field, expected in [("elongation_templates", 89), ("partial_templates", 43),
                                ("omission_positions", 100)]:
            items = profile[field]
            if (not items or sum(x["count"] for x in items) != expected
                    or any(type(x["count"]) is not int or x["count"] <= 0 for x in items)):
                raise ValueError("Empirical profile weights do not conserve source events")
        for item in profile["elongation_templates"]:
            if (len(item["positions"]) != len(item["extras"]) or not item["extras"]
                    or any(type(x) is not int or x < 1 for x in item["extras"])):
                raise ValueError("Invalid elongation template")
        for item in profile["partial_templates"]:
            if not 0 < item["folded"] < item["available"] or len(item["positions"]) != item["folded"]:
                raise ValueError("Invalid folded/available joint profile")
        for field in ["elongation_templates", "partial_templates"]:
            for item in profile[field]:
                values = [rational(p) for p in item["positions"]]
                if values != sorted(set(values)) or not all(0 <= x <= 1 for x in values):
                    raise ValueError("Profile positions must be ordered, distinct relative indices")
        if not all(0 < rational(x["position"]) < 1 for x in profile["omission_positions"]):
            raise ValueError("Omission source positions must be strictly interior")

    def foldable(self, word):
        return [i for i, char in enumerate(word) if char != char.translate(self.original.fold_map)]

    def compatible_casing(self, word):
        return self.original.lower(self.original.fold(word)) == self.original.fold(self.original.lower(word))

    def eligible(self, word, family):
        if family not in FAMILIES:
            raise ValueError("Unknown transfer family")
        minimum = 3 if family == "internal_omission" else 2
        if len(word) < minimum or any(char not in self.original.alphabet for char in word):
            return False
        return family != "partial_folding" or (len(self.foldable(word)) >= 2 and self.compatible_casing(word))

    def parameters(self, word, family, key):
        if not self.eligible(word, family):
            raise ValueError("Cannot edit a word outside the declared opportunity pool")
        if family == "elongation":
            neutral = self.original.lower(word)
            ends = [i for i, char in enumerate(neutral)
                    if i == len(neutral) - 1 or char != neutral[i + 1]]
            applicable = [(i, p) for i, p in enumerate(self.profile["elongation_templates"])
                          if len(p["extras"]) <= len(ends)]
            identifier, template = weighted_choice(applicable, key)
            positions = ordered_assignment(ends, [rational(p) for p in template["positions"]],
                                           len(word) - 1, key)
            return {"template_id": identifier, "positions": positions, "extras": template["extras"],
                    "feasible_template_conditioning": len(applicable) != len(self.profile["elongation_templates"]),
                    "target_runs": len(ends)}
        if family == "internal_omission":
            identifier, template = weighted_choice(list(enumerate(self.profile["omission_positions"])), key)
            target = rational(template["position"])
            position = min(range(1, len(word) - 1), key=lambda i: (
                abs(Fraction(i, len(word) - 1) - target),
                stable_digest([*key, "omission_position_tie", i]), i))
            return {"template_id": identifier, "position": position,
                    "source_relative_position": template["position"]}
        candidates = self.foldable(word)
        templates = list(enumerate(self.profile["partial_templates"]))
        conditional = [(i, p) for i, p in templates if p["available"] == len(candidates)]
        identifier, template = weighted_choice(conditional or templates, key)
        source_fraction = Fraction(template["folded"], template["available"])
        count = template["folded"] if conditional else min(range(1, len(candidates)), key=lambda k: (
            abs(Fraction(k, len(candidates)) - source_fraction),
            stable_digest([*key, "subset_count_tie", k]), k))
        source_positions = [rational(p) for p in template["positions"]]
        targets = [source_positions[(2 * j + 1) * len(source_positions) // (2 * count)]
                   for j in range(count)]
        positions = ordered_assignment(candidates, targets, len(word) - 1, key)
        return {"template_id": identifier, "positions": positions, "target_foldable": len(candidates),
                "source_available": template["available"], "source_folded": template["folded"],
                "mapped_folded": count, "unseen_foldable_count": not bool(conditional)}

    def transform(self, word, family, parameters):
        if family == "elongation":
            additions = dict(zip(parameters["positions"], parameters["extras"], strict=True))
            return "".join(char + char * additions.get(i, 0) for i, char in enumerate(word))
        if family == "internal_omission":
            index = parameters["position"]
            return word[:index] + word[index + 1:]
        if family == "partial_folding":
            positions = set(parameters["positions"])
            return "".join(char.translate(self.original.fold_map) if i in positions else char
                           for i, char in enumerate(word))
        if family == CONTROL:
            return self.original.fold(word)
        raise ValueError("Unknown transfer operation")


def position_key(seed, split, row_id, word_index, family):
    return [VERSION, seed, split, row_id, word_index, family]


def build_transfer_plans(records, noise, *, split, seed):
    if split not in {"validation", "test"} or type(seed) is not int:
        raise ValueError("Expected an explicit split and integer seed")
    pools = {f: [] for f in FAMILIES}
    for row_id, words in records[["row_id", "tokens"]].itertuples(index=False, name=None):
        for index, word in enumerate(words):
            for family in FAMILIES:
                if noise.eligible(word, family):
                    pools[family].append((stable_digest(
                        [*position_key(seed, split, row_id, index, family), "select"]), row_id, index))
    plans = {"version": VERSION, "seed": seed, "split": split,
             "source_words_sha256": source_words_digest(records), "families": {}}
    for family, pool in pools.items():
        numerator, denominator = RATES[family]
        count = len(pool) * numerator // denominator
        selected = [(row_id, index) for _, row_id, index in sorted(pool)[:count]]
        plans["families"][family] = {
            "eligible_words": len(pool), "selected_words": count, "selected_positions": selected,
            "selected_positions_sha256": stable_digest(sorted(selected)).hex()}
    return plans


def apply_transfer(records, noise, plans, family):
    if family not in (*FAMILIES, CONTROL) or plans["version"] != VERSION:
        raise ValueError("Unsupported transfer operation/plan")
    if plans["source_words_sha256"] != source_words_digest(records):
        raise ValueError("Transfer plan belongs to different words or original IDs")
    selector = "partial_folding" if family == CONTROL else family
    rows = [{"row_id": row_id, "tokens": list(words), "tags": list(tags)}
            for row_id, words, tags in records[["row_id", "tokens", "tags"]].itertuples(index=False, name=None)]
    lookup = {row["row_id"]: row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate original row IDs")
    edits = []
    for row_id, index in plans["families"][selector]["selected_positions"]:
        word = lookup[row_id]["tokens"][index]
        key = position_key(plans["seed"], plans["split"], row_id, index, selector)
        parameters = ({"positions": noise.foldable(word), "target_foldable": len(noise.foldable(word))}
                      if family == CONTROL else noise.parameters(word, family, key))
        output = noise.transform(word, family, parameters)
        if output == word or not output:
            raise ValueError("A selected transfer word did not undergo a valid change")
        lookup[row_id]["tokens"][index] = output
        edits.append({"row_id": row_id, "word_index": index, "family": family,
                      "before": word, "after": output, "parameters": parameters})
    return pd.DataFrame(rows, columns=["row_id", "tokens", "tags"]), edits
