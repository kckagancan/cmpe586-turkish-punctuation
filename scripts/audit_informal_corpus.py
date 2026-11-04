"""Audit a pinned local informal corpus without model calls or source rewriting."""

import argparse
import hashlib
import json
import platform
import shutil
import unicodedata
from pathlib import Path

from turkish_punctuation.corpus_audit import audit_rows, read_multilexnorm
from turkish_punctuation.noise import WordNoise

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "scripts/audit_informal_corpus.py",
    "src/turkish_punctuation/corpus_audit.py",
    "src/turkish_punctuation/noise.py",
    "src/turkish_punctuation/__init__.py",
    "docs/informal-corpus-audit-protocol.md",
    "docs/informal-corpus-selection.md",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/informal-corpus-audit.json"))
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    expected_schema = {
        "format": "multilexnorm_two_column_token_per_line", "encoding": "utf-8",
        "fields": ["original", "normalized"], "blank_line": "sentence_boundary",
        "empty_normalized_token": "merge_continuation_mark_anchor_and_continuation",
        "multiword_normalized_token": "split_exclude_from_word_preserving_pairs",
        "token_fields_preserved_without_strip": True,
    }
    expected_invariants = dict.fromkeys([
        "model_calls", "punctuation_gold_inferred", "historical_recipes_changed",
        "raw_text_in_public_summary", "corpus_population_prevalence_claim",
    ], False)
    expected_coverage = {
        "direction": "normalized_corrected_to_observed_original",
        "views": ["exact", "turkish_lowercase_both_sides_for_lexical_diagnostic_only"],
        "exclude_boundary_changing_pairs": True,
        "single_word_reachability": (
            "all_eligible_nonempty_operator_subsets_in_frozen_order_all_legal_swap_positions"),
        "new_mechanism_shape_minimum_corrected_length": 4,
    }
    if (config["version"] != "informal-corpus-audit-v1"
            or config["schema"] != expected_schema
            or config["coverage"] != expected_coverage
            or config["source"]["official_split"] != "train"
            or config["invariants"] != expected_invariants):
        raise ValueError("Unsupported audit version, split, format or invariants")
    source = (ROOT / config["source"]["path"]).resolve()
    output_root = (ROOT / config["outputs"]).resolve()
    if not source.is_relative_to(ROOT / "data/raw"):
        raise ValueError("Source must remain in ignored data/raw storage")
    if not output_root.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Audit receipts must remain in ignored artifacts storage")
    if source.stat().st_size != config["source"]["bytes"]:
        raise ValueError("Pinned source size mismatch")
    if digest(source) != config["source"]["sha256"]:
        raise ValueError("Pinned source SHA-256 mismatch")
    noise_path = ROOT / config["frozen_noise_config"]
    if digest(noise_path) != config["frozen_noise_config_sha256"]:
        raise ValueError("Frozen noise configuration mismatch")
    bindings = {name: digest(ROOT / name) for name in SOURCES}
    bindings[str(config_path.relative_to(ROOT))] = digest(config_path)
    bindings[config["frozen_noise_config"]] = digest(noise_path)
    identity = hashlib.sha256(json.dumps(
        {"bindings": bindings, "source_sha256": digest(source)},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    destination = output_root / ("audit-" + identity[:16])
    if destination.exists():
        raise FileExistsError(f"Existing audit is preserved: {destination}")
    rows, schema = read_multilexnorm(source.read_text(encoding=config["schema"]["encoding"]))
    noise = WordNoise(json.loads(noise_path.read_text()))
    summary, trace = audit_rows(
        rows, noise, config["coverage"]["new_mechanism_shape_minimum_corrected_length"])
    if (summary["sentences"] != config["source"]["expected_sentences"]
            or summary["source_slots"] != config["source"]["expected_source_slots"]):
        raise ValueError("Pinned file's sentence/slot counters differ from expected source metadata")
    summary.update({"audit_identity": identity, "version": config["version"],
                    "source": config["source"], "source_bindings": bindings,
                    "runtime": {"python": platform.python_version(),
                                "unicode": unicodedata.unidata_version},
                    "schema_audit": schema, "model_calls": False,
                    "test_split_read": False, "training_or_inference_run": False})
    destination.mkdir(parents=True)
    write_json(destination / "summary.json", summary)
    with (destination / "slot-trace.jsonl").open("w", encoding="utf-8") as stream:
        for item in trace:
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
    snapshot_root = destination / "snapshots"
    for name in bindings:
        target = snapshot_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    committed_files = {str(path.relative_to(destination)): digest(path)
                       for path in destination.rglob("*") if path.is_file()}
    write_json(destination / "commit.json", {"identity": identity, "files": committed_files})
    print(json.dumps({"audit_identity": identity, "destination": str(destination.relative_to(ROOT)),
                      "sentences": summary["sentences"], "slots": summary["source_slots"],
                      "models_or_test_data_used": False}, indent=2))


if __name__ == "__main__":
    main()
