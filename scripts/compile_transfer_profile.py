"""Compile anonymous aggregate morphology templates from verified local corpus evidence."""

import hashlib
import json
import math
from collections import Counter
from fractions import Fraction
from pathlib import Path

from turkish_punctuation.corpus_audit import read_multilexnorm
from turkish_punctuation.noise import WordNoise
from turkish_punctuation.transfer_noise import TransferNoise

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = "docs/informal-corpus-calibration-profile.json"
CALIBRATION_SHA = "38a797a61535dbe465d963e41e4c8273962a732106a33eca7beabbeffd3ff21a"
FEATURE_SHA = "16187bc5a7d32434a4fad865a99006c7a036e50259a6bfdb78262fb1ec30f176"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def fraction_pair(value, scale):
    numerator = round(value * scale)
    if abs(value - numerator / scale) > 1e-12:
        raise ValueError("Saved relative position is not an original character index")
    q = Fraction(numerator, scale)
    return [q.numerator, q.denominator]


def compile_profile():
    if digest(ROOT / CALIBRATION) != CALIBRATION_SHA:
        raise ValueError("Completed calibration aggregate changed")
    calibration = read(ROOT / CALIBRATION)
    run = ROOT / "artifacts/corpus-audit/multilexnorm-depth-v1/calibration-f8b903a57451c895"
    committed = read(run / "commit.json")
    if committed["identity"] != calibration["identity"]:
        raise ValueError("Completed calibration identity differs")
    for name, expected in committed["files"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or digest(path) != expected:
            raise ValueError("Completed calibration files changed")
    feature_path = run / "affected-slot-features.jsonl"
    if digest(feature_path) != FEATURE_SHA:
        raise ValueError("Completed affected-slot feature trace changed")
    source = ROOT / calibration["source"]["path"]
    if digest(source) != calibration["source"]["sha256"]:
        raise ValueError("Pinned source train file changed")
    rows, _ = read_multilexnorm(source.read_text())
    noise = WordNoise(read(ROOT / "configs/noise.json"))
    by_id = {(s.sentence_id, s.word_index): s for row in rows for s in row}
    partial_pool = original_partial_pool = 0
    for slot in by_id.values():
        clean, observed = noise.lower(slot.normalized), noise.lower(slot.original)
        if (slot.boundary or len(clean) < 2 or any(c not in noise.alphabet for c in clean + observed)
                or sum(c != c.translate(noise.fold_map) for c in clean) < 2):
            continue
        original_partial_pool += 1
        partial_pool += noise.lower(noise.fold(slot.normalized)) == noise.fold(clean)
    if (original_partial_pool, partial_pool) != (621, 617):
        raise ValueError("Surface-compatible folding opportunity derivation changed")
    elongation, omission, partial = Counter(), Counter(), Counter()
    features = [json.loads(line) for line in feature_path.read_text().splitlines()]
    ambiguity_scale = math.lcm(*(x["details"]["internal_omission"]["candidate_count"]
                                for x in features if x["families"]["internal_omission"]))
    if ambiguity_scale != 2:
        raise ValueError("This source's ambiguity marginalization scale changed")
    for item in features:
        slot = by_id[item["sentence_id"], item["word_index"]]
        if hashlib.sha256((slot.original + "\0" + slot.normalized).encode()).hexdigest() != item["pair_sha256"]:
            raise ValueError("Source/feature pair hash differs")
        scale = item["corrected_length"] - 1
        for family, detail in item["details"].items():
            if family == "elongation":
                payload = {"positions": [fraction_pair(x["relative_run_end"], scale)
                                         for x in detail["run_details"]],
                           "extras": [x["extra"] for x in detail["run_details"]]}
                elongation[json.dumps(payload, sort_keys=True)] += 1
            elif family == "internal_omission":
                for index in detail["candidate_indices"]:
                    q = Fraction(index, scale)
                    omission[(q.numerator, q.denominator)] += ambiguity_scale // detail["candidate_count"]
            elif family == "partial_folding":
                if noise.lower(noise.fold(slot.normalized)) != noise.fold(noise.lower(slot.normalized)):
                    raise ValueError("A source partial-fold hit is outside the operational casing scope")
                payload = {"available": detail["available"], "folded": detail["folded"],
                           "positions": [fraction_pair(x["relative_position"], scale)
                                         for x in detail["changed_positions"]]}
                partial[json.dumps(payload, sort_keys=True)] += 1
    source_names = [CALIBRATION, str(feature_path.relative_to(ROOT)), str(source.relative_to(ROOT)),
                    "scripts/compile_transfer_profile.py", "src/turkish_punctuation/noise.py",
                    "src/turkish_punctuation/corpus_audit.py", "configs/noise.json"]
    for name in source_names:
        expected = calibration["specification"]["bindings"].get(name)
        if expected is not None and digest(ROOT / name) != expected:
            raise ValueError("A frozen calibration dependency changed")
    profile = {
        "version": "transfer-morphology-profile-v1", "calibration_identity": calibration["identity"],
        "source_bindings": {name: digest(ROOT / name) for name in source_names},
        "rates": {"elongation": {"numerator": 89, "denominator": 4512},
                  "internal_omission": {"numerator": 50, "denominator": 4174},
                  "partial_folding": {"numerator": 43, "denominator": partial_pool}},
        "partial_operational_scope": {"parent_hits": 43, "parent_opportunities": original_partial_pool,
                                      "excluded_casing_incompatible_opportunities": 4,
                                      "excluded_hits": 0, "compatibility": "turkish_lower_and_frozen_fold_commute"},
        "elongation_templates": [dict(json.loads(key), count=count) for key, count in sorted(elongation.items())],
        "omission_positions": [{"position": list(q), "count": count} for q, count in sorted(omission.items())],
        "omission_ambiguity_weight_scale": ambiguity_scale,
        "partial_templates": [dict(json.loads(key), count=count) for key, count in sorted(partial.items())],
        "raw_words_tweets_users_or_source_ids_in_profile": False,
        "population_frequency_claim": False,
    }
    TransferNoise(profile, read(ROOT / "configs/noise.json"))
    return profile


def main():
    profile = compile_profile()
    destination = ROOT / "configs/transfer-noise-profile.json"
    payload = json.dumps(profile, ensure_ascii=False, indent=2) + "\n"
    if destination.exists():
        if destination.read_text() != payload:
            raise FileExistsError("Existing compiled profile is preserved; version changes explicitly")
        print("Matching compiled profile verified; not regenerated.")
        return
    destination.write_text(payload)
    print(json.dumps({"path": str(destination.relative_to(ROOT)), "sha256": digest(destination),
                      "templates": {f: len(profile[f]) for f in
                                    ["elongation_templates", "omission_positions", "partial_templates"]},
                      "model_calls": False}, indent=2))


if __name__ == "__main__":
    main()
