"""Run a new data-only depth profile using the preserved MultiLexNorm audit trace."""

import argparse
import hashlib
import json
import platform
import shutil
import unicodedata
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from turkish_punctuation.corpus_audit import read_multilexnorm
from turkish_punctuation.corpus_depth import analyze_depth
from turkish_punctuation.noise import WordNoise

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["scripts/analyze_corpus_depth.py", "src/turkish_punctuation/corpus_depth.py"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def check_hash(path, expected):
    if not path.is_file() or digest(path) != expected:
        raise ValueError(f"Pinned file changed or missing: {path.relative_to(ROOT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/informal-corpus-depth.json"))
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = read(config_path)
    if (config["version"] != "informal-corpus-depth-v1"
            or config["model_calls"] or config["source_test_read"]
            or config["generate_corruptions"] or config["freeze_evaluation_recipe"]
            or config["minimum_corrected_length"] != 4
            or config["character_intensity_bins"] != {
                "threshold_quantiles": [0.25, 0.75], "lower": "x_le_q25",
                "middle": "q25_lt_x_le_q75", "upper": "x_gt_q75",
                "ties_or_empty_bins": "retain_report_do_not_force_three_distinct_levels"}):
        raise ValueError("Expected the supported data-only depth specification")
    audit_root = (ROOT / config["audit_directory"]).resolve()
    for name, key in [("summary.json", "audit_summary_sha256"),
                      ("slot-trace.jsonl", "audit_trace_sha256"),
                      ("commit.json", "audit_commit_sha256")]:
        check_hash(audit_root / name, config[key])
    audit = read(audit_root / "summary.json")
    if audit["audit_identity"] != config["audit_identity"]:
        raise ValueError("Original audit identity mismatch")
    original_manifest = read(audit_root / "commit.json")
    if original_manifest["identity"] != config["audit_identity"]:
        raise ValueError("Original audit manifest identity mismatch")
    for name, expected in original_manifest["files"].items():
        path = (audit_root / name).resolve()
        if not path.is_relative_to(audit_root):
            raise ValueError("Original manifest path leaves its run directory")
        check_hash(path, expected)
    for name, expected in audit["source_bindings"].items():
        check_hash(ROOT / name, expected)
    audit_config_path = ROOT / config["source_audit_config"]
    if str(audit_config_path.relative_to(ROOT)) not in audit["source_bindings"]:
        raise ValueError("Source configuration was not bound by the original audit")
    source_config = read(audit_config_path)
    if source_config["source"]["official_split"] != "train":
        raise ValueError("Only the previously audited train split is permitted")
    source = (ROOT / source_config["source"]["path"]).resolve()
    output_root = (ROOT / config["output_root"]).resolve()
    if not source.is_relative_to(ROOT / "data/raw") or not output_root.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Raw input/output must remain in ignored local storage")
    check_hash(source, source_config["source"]["sha256"])
    if source.stat().st_size != source_config["source"]["bytes"]:
        raise ValueError("Source byte size differs from the audited release")
    runtime = {"python": platform.python_version(), "unicode": unicodedata.unidata_version,
               "numpy": np.__version__}
    if runtime["unicode"] != audit["runtime"]["unicode"]:
        raise ValueError("Unicode runtime changed; do not silently reinterpret the source")
    bindings = dict(audit["source_bindings"])
    for name in [*SOURCES, config["protocol"], str(config_path.relative_to(ROOT))]:
        bindings[name] = digest(ROOT / name)
    for name in ["summary.json", "slot-trace.jsonl", "commit.json"]:
        bindings[str((audit_root / name).relative_to(ROOT))] = digest(audit_root / name)
    specification = {"bindings": bindings, "source_sha256": digest(source), "runtime": runtime}
    identity = hashlib.sha256(json.dumps(
        specification, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    destination = output_root / ("depth-" + identity[:16])
    if destination.exists():
        raise FileExistsError(f"Completed or existing profile is preserved: {destination}")
    rows, schema = read_multilexnorm(source.read_text(encoding="utf-8"))
    if len(rows) != audit["sentences"] or sum(map(len, rows)) != audit["source_slots"]:
        raise ValueError("Source row/slot conservation failed")
    trace = [json.loads(line) for line in (audit_root / "slot-trace.jsonl").read_text().splitlines()]
    noise = WordNoise(read(ROOT / source_config["frozen_noise_config"]))
    summary, slots, sentences, replicates = analyze_depth(rows, trace, noise, config, audit)
    summary.update({"identity": identity, "version": config["version"], "runtime": runtime,
                    "source": source_config["source"], "original_audit_identity": audit["audit_identity"],
                    "specification": specification, "parser_audit": schema,
                    "previous_audit_reexecuted": False})
    output_root.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".depth-", dir=output_root) as temporary:
        staging = Path(temporary) / "run"
        staging.mkdir()
        write(staging / "summary.json", summary)
        for name, records in [("slot-features.jsonl", slots), ("sentence-features.jsonl", sentences)]:
            with (staging / name).open("w", encoding="utf-8") as stream:
                for item in records:
                    stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":"),
                                            allow_nan=False) + "\n")
        np.save(staging / "bootstrap-replicates.npy", replicates, allow_pickle=False)
        for name in bindings:
            if name.startswith("artifacts/"):
                continue  # Reuse bound original traces; do not copy them into another run.
            target = staging / "snapshots" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        write(staging / "commit.json", {"identity": identity,
              "files": {str(p.relative_to(staging)): digest(p)
                        for p in sorted(staging.rglob("*")) if p.is_file()}})
        staging.rename(destination)
    print(json.dumps({"identity": identity, "destination": str(destination.relative_to(ROOT)),
                      "counts": {f: p["occurrences"] for f, p in summary["families"].items()},
                      "models_or_test_split_used": False}, indent=2))


if __name__ == "__main__":
    main()
