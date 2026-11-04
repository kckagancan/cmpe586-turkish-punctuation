"""Profile family-specific short-word scopes without changing completed analyses."""

import hashlib
import json
import platform
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from analyze_corpus_depth import check_hash, digest, read, write

from turkish_punctuation.corpus_audit import mechanism_shapes, read_multilexnorm
from turkish_punctuation.corpus_depth import (
    FAMILIES,
    family_profile,
    make_features,
    sentence_features,
)
from turkish_punctuation.noise import WordNoise

ROOT = Path(__file__).resolve().parents[1]
VERSION = "informal-corpus-calibration-profile-v1"
DEPTH_ID = "cde287765953c12dc78f4f56d0eca6ec81c017d21dd10626f3f9fffc7d161cca"
LENGTH_ID = "1bd0abe5b1d6542d4a5b138ae5c1d8c3f8b2fc35cf3b835e832a1f9d7229703c"
POLICY = {"elongation": {"minimum_corrected_length": 2},
          "internal_omission": {"minimum_corrected_length": 3},
          "partial_folding": {"minimum_corrected_length": 2,
                              "minimum_foldable_occurrences": 2}}


def main():
    output_root = ROOT / "artifacts/corpus-audit/multilexnorm-depth-v1"
    parents = {"depth": output_root / ("depth-" + DEPTH_ID[:16]),
               "length": output_root / ("length-" + LENGTH_ID[:16])}
    for label, path in parents.items():
        manifest = read(path / "commit.json")
        expected = DEPTH_ID if label == "depth" else LENGTH_ID
        if manifest["identity"] != expected:
            raise ValueError("Parent identity mismatch")
        for name, expected_hash in manifest["files"].items():
            target = (path / name).resolve()
            if not target.is_relative_to(path):
                raise ValueError("Parent path leaves its run")
            check_hash(target, expected_hash)
    depth, length = (read(parents[label] / "summary.json") for label in ["depth", "length"])
    if (platform.python_version() != depth["runtime"]["python"]
            or np.__version__ != depth["runtime"]["numpy"]):
        raise ValueError("Calibration profile must use the parent analysis runtime")
    settings = read(ROOT / "configs/informal-corpus-depth.json")
    for name, expected in depth["specification"]["bindings"].items():
        check_hash(ROOT / name, expected)
    source = ROOT / depth["source"]["path"]
    check_hash(source, depth["source"]["sha256"])
    names = ["scripts/profile_corpus_calibration.py", "scripts/analyze_corpus_depth.py",
             "src/turkish_punctuation/corpus_depth.py", "src/turkish_punctuation/corpus_audit.py",
             "src/turkish_punctuation/noise.py", "src/turkish_punctuation/__init__.py",
             "configs/informal-corpus-depth.json", "configs/informal-corpus-audit.json",
             "configs/noise.json"]
    bindings = {name: digest(ROOT / name) for name in names}
    for path in parents.values():
        bindings[str((path / "summary.json").relative_to(ROOT))] = digest(path / "summary.json")
    spec = {"version": VERSION, "family_policy": POLICY, "bindings": bindings,
            "source_sha256": depth["source"]["sha256"], "runtime": depth["runtime"]}
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    destination = output_root / ("calibration-" + identity[:16])
    if destination.exists():
        raise FileExistsError("Existing calibration profile is preserved")
    rows, _ = read_multilexnorm(source.read_text())
    audit_config = read(ROOT / "configs/informal-corpus-audit.json")
    audit_root = ROOT / settings["audit_directory"]
    trace = [json.loads(line) for line in (audit_root / "slot-trace.jsonl").read_text().splitlines()]
    noise = WordNoise(read(ROOT / audit_config["frozen_noise_config"]))
    cursor = 0
    derived = []
    for row in rows:
        for slot in row:
            item = dict(trace[cursor])
            cursor += 1
            item["case_neutral_shapes"] = [] if slot.boundary else mechanism_shapes(
                noise.lower(slot.normalized), noise.lower(slot.original), noise, 2)
            derived.append(item)
    features = make_features(rows, derived, noise, 2)
    profiles, by_family_sentences = {}, {}
    columns = []
    for family, policy in POLICY.items():
        scoped = [dict(x, reason="below_family_minimum")
                  if x["reason"] == "eligible" and x["corrected_length"] < policy["minimum_corrected_length"]
                  else x for x in features]
        sentences = sentence_features(scoped, len(rows))
        profile = family_profile(family, scoped, sentences, settings["descriptive_quantiles"])
        expected = length["views"][str(policy["minimum_corrected_length"])]["families"][family]
        if (profile["occurrences"] != expected["matches"]
                or profile["opportunities"] != expected["opportunities"]):
            raise ValueError("Family profile differs from completed length sensitivity")
        profiles[family] = profile
        by_family_sentences[family] = sentences
        key = "partial_opportunities" if family == "partial_folding" else "base_opportunities"
        columns.extend([[s["family_hits"][family] for s in sentences], [s[key] for s in sentences]])
    matrix = np.array(columns, dtype=np.int64).T
    bootstrap = settings["bootstrap"]
    generator = np.random.Generator(np.random.PCG64(bootstrap["seed"]))
    batches = []
    for offset in range(0, bootstrap["replicates"], bootstrap["batch_size"]):
        size = min(bootstrap["batch_size"], bootstrap["replicates"] - offset)
        sample = matrix[generator.integers(0, len(rows), size=(size, len(rows)))]
        sums = sample.sum(axis=1)
        rates = np.divide(sums[:, ::2], sums[:, 1::2], out=np.full((size, 3), np.nan),
                          where=sums[:, 1::2] != 0)
        batches.append(np.concatenate([rates, (sample[:, :, ::2] > 0).mean(axis=1)], axis=1))
    replicates = np.concatenate(batches)
    for i, (family, profile) in enumerate(profiles.items()):
        intervals = {}
        for key, index in [("pooled_opportunity_rate", i), ("sentence_presence_rate", i+3)]:
            values = replicates[:, index]
            finite = values[np.isfinite(values)]
            intervals[key] = {"interval": np.quantile(finite, [.025, .975], method="linear").tolist(),
                              "defined_replicates": len(finite),
                              "undefined_replicates": len(values)-len(finite)}
        profile["sentence_cluster_bootstrap_stability"] = intervals
    combinations = {}
    for i in range(len(rows)):
        key = "+".join(f for f in FAMILIES if by_family_sentences[f][i]["family_hits"][f]) or "none"
        combinations[key] = combinations.get(key, 0) + 1
    result = {"identity": identity, "version": VERSION, "specification": spec,
              "source": depth["source"], "parent_depth_identity": DEPTH_ID,
              "parent_length_identity": LENGTH_ID, "families": profiles,
              "sentence_membership_combinations": dict(sorted(combinations.items())),
              "bootstrap_specification": bootstrap,
              "phase": "exploratory_family_specific_scope_after_length_sensitivity_before_model_scores",
              "source_test_read": False, "model_calls": False,
              "corruptions_generated": False, "evaluation_recipe_frozen": False}
    with TemporaryDirectory(prefix=".calibration-", dir=output_root) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        write(staging / "summary.json", result)
        write(staging / "family-sentence-features.json", by_family_sentences)
        with (staging / "affected-slot-features.jsonl").open("w") as stream:
            for item in features:
                if any(item["families"].values()):
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
        np.save(staging / "bootstrap-replicates.npy", replicates, allow_pickle=False)
        for name in names:
            target = staging / "snapshots" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        write(staging / "commit.json", {"identity": identity, "files": {
            str(p.relative_to(staging)): digest(p) for p in staging.rglob("*") if p.is_file()}})
        staging.rename(destination)
    print(json.dumps({"identity": identity, "destination": str(destination.relative_to(ROOT)),
                      "family_counts": {f: p["occurrences"] for f, p in profiles.items()}}, indent=2))


if __name__ == "__main__":
    main()
