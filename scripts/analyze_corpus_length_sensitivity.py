"""Describe short-word scope as a separate exploratory follow-up; no scoring."""

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from turkish_punctuation.corpus_audit import mechanism_shapes, read_multilexnorm
from turkish_punctuation.corpus_depth import FAMILIES
from turkish_punctuation.noise import WordNoise

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/informal-corpus-length-sensitivity.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main():
    config = read(ROOT / CONFIG)
    if (config["version"] != "informal-corpus-length-sensitivity-v1"
            or config["minimum_corrected_lengths"] != [1, 2, 3, 4]
            or config["model_calls"] or config["source_test_read"] or config["change_parent_profile"]):
        raise ValueError("Expected the separate data-only length sensitivity")
    audit_config = ROOT / config["source_audit_config"]
    if digest(audit_config) != config["source_audit_config_sha256"]:
        raise ValueError("Original audit config changed")
    parent = ROOT / config["parent_depth_directory"]
    manifest = read(parent / "commit.json")
    depth = read(parent / "summary.json")
    if manifest["identity"] != config["parent_depth_identity"] or depth["identity"] != manifest["identity"]:
        raise ValueError("Unexpected parent depth identity")
    for name, expected in manifest["files"].items():
        path = (parent / name).resolve()
        if not path.is_relative_to(parent) or digest(path) != expected:
            raise ValueError("Parent depth files changed")
    source_config = read(audit_config)
    source = ROOT / source_config["source"]["path"]
    if source_config["source"]["official_split"] != "train" or digest(source) != source_config["source"]["sha256"]:
        raise ValueError("Only the pinned source train file is permitted")
    names = [CONFIG, "scripts/analyze_corpus_length_sensitivity.py",
             "src/turkish_punctuation/corpus_audit.py", "src/turkish_punctuation/corpus_depth.py",
             "src/turkish_punctuation/noise.py", "src/turkish_punctuation/__init__.py",
             source_config["frozen_noise_config"], config["source_audit_config"]]
    for name in names:
        if name in depth["specification"]["bindings"] and digest(ROOT / name) != depth["specification"]["bindings"][name]:
            raise ValueError("Frozen source/config dependency changed")
    bindings = {name: digest(ROOT / name) for name in names}
    bindings[str((parent / "summary.json").relative_to(ROOT))] = digest(parent / "summary.json")
    identity = hashlib.sha256(json.dumps(bindings, sort_keys=True).encode()).hexdigest()
    output_root = (ROOT / config["output_root"]).resolve()
    if not output_root.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Output must remain in ignored artifacts")
    destination = output_root / ("length-" + identity[:16])
    if destination.exists():
        raise FileExistsError("Existing sensitivity is preserved")
    rows, _ = read_multilexnorm(source.read_text())
    noise = WordNoise(read(ROOT / source_config["frozen_noise_config"]))
    pools = {minimum: [] for minimum in config["minimum_corrected_lengths"]}
    for row in rows:
        for slot in row:
            clean, observed = noise.lower(slot.normalized), noise.lower(slot.original)
            if slot.boundary or any(c not in noise.alphabet for c in clean + observed):
                continue
            for minimum, pool in pools.items():
                if len(clean) < minimum:
                    continue
                shapes = mechanism_shapes(clean, observed, noise, minimum)
                pool.append({"sentence_id": slot.sentence_id, "corrected_length": len(clean),
                             "partial_opportunity": sum(c != c.translate(noise.fold_map) for c in clean) >= 2,
                             "families": {f: shape in shapes for f, shape in FAMILIES.items()}})
    views = {}
    for minimum, pool in pools.items():
        family_results = {}
        for f in FAMILIES:
            hits = [x for x in pool if x["families"][f]]
            denominator = sum(x["partial_opportunity"] for x in pool) if f == "partial_folding" else len(pool)
            family_results[f] = {"matches": len(hits), "opportunities": denominator,
                                 "fraction": len(hits) / denominator,
                                 "affected_sentences": len({x["sentence_id"] for x in hits}),
                                 "corrected_length_histogram": dict(sorted(Counter(x["corrected_length"] for x in hits).items()))}
        views[str(minimum)] = {"common_opportunities": len(pool), "families": family_results}
    if any(views["4"]["families"][f]["matches"] != depth["families"][f]["occurrences"]
           or views["4"]["families"][f]["opportunities"] != depth["families"][f]["opportunities"]
           for f in FAMILIES):
        raise ValueError("Min4 sensitivity does not reproduce parent counters")
    summary = {"version": config["version"], "identity": identity, "source": source_config["source"],
               "parent_depth_identity": depth["identity"], "bindings": bindings, "views": views,
               "phase": config["phase"], "models_or_test_split_used": False,
               "parent_results_changed": False, "raw_text_published": False}
    with TemporaryDirectory(prefix=".length-", dir=output_root) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        write(staging / "summary.json", summary)
        for name in names:
            target = staging / "snapshots" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        write(staging / "commit.json", {"identity": identity, "files": {
            str(p.relative_to(staging)): digest(p) for p in staging.rglob("*") if p.is_file()}})
        staging.rename(destination)
    print(json.dumps({"identity": identity, "destination": str(destination.relative_to(ROOT)),
                      "views": views}, indent=2))


if __name__ == "__main__":
    main()
