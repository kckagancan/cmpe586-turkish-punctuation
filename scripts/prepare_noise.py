"""Prepare/audit the validation noise grid locally; never load model weights."""

import argparse
import hashlib
import json
import platform
import re
import shutil
import subprocess
import time
import unicodedata
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import disable_progress_bars, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from turkish_punctuation.data import read_split
from turkish_punctuation.noise import (
    FAMILIES,
    NoisePlans,
    WordNoise,
    apply_noise,
    build_plans,
    stable_digest,
)
from turkish_punctuation.noise_audit import audit_changes, encode_and_audit, validate_source
from turkish_punctuation.preprocessing import partition_records
from turkish_punctuation.tokenization import IGNORE_INDEX, LABEL_TO_ID

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/prepare_noise.py",
    "src/turkish_punctuation/noise.py",
    "src/turkish_punctuation/noise_audit.py",
    "src/turkish_punctuation/data.py",
    "src/turkish_punctuation/tokenization.py",
    "src/turkish_punctuation/preprocessing.py",
    "configs/tokenizer.json",
    "docs/data-manifest.json",
    "docs/noise-protocol.md",
)


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def safe_name(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
        raise ValueError(f"Condition name must be a simple directory name: {value!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/noise.json", type=Path)
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve()
    config = json.loads(config_path.read_text())
    noise = WordNoise(config)
    split = config["development_split"]
    if split != "validation":
        raise ValueError(
            "This development command prepares validation only; test remains held out."
        )
    destination = (ROOT / config["output_directory"]).resolve()
    if not destination.is_relative_to(ROOT / "data/processed"):
        raise ValueError("Noise output must be a new directory under data/processed/.")
    if destination.exists():
        raise FileExistsError(
            f"Output exists; choose a new run directory in the config: {destination}"
        )
    audit_config = config["audit"]
    if (
        type(audit_config["tokenizer_batch_size"]) is not int
        or audit_config["tokenizer_batch_size"] < 1
        or type(audit_config["examples_per_condition"]) is not int
        or audit_config["examples_per_condition"] < 1
    ):
        raise ValueError("Audit batch size and example count must be positive integers.")
    for level in config["levels"]:
        safe_name(level)
    full_conditions = config["full_stress_conditions"]
    full_names = []
    for item in full_conditions:
        full_names.append(safe_name(item["name"]))
        if item["family"] not in FAMILIES[:2] or item["selection"] != "all_eligible_words":
            raise ValueError("Only full-case and full-fold endpoints are implemented.")
    if len(set(full_names)) != len(full_names):
        raise ValueError("Full-stress condition names must be unique.")
    ablations = config["optional_input_ablations"]
    if ablations["enabled_by_default"]:
        if (
            ablations["level"] not in config["levels"]
            or ablations["reuse_remaining_component_plans"] is not True
            or set(ablations["conditions"]) != {f"combined_without_{f}" for f in FAMILIES}
        ):
            raise ValueError("Unsupported input ablation settings.")
    tokenizer_config = json.loads((ROOT / "configs/tokenizer.json").read_text())
    data_manifest = json.loads((ROOT / "docs/data-manifest.json").read_text())
    prepared = ROOT / config["prepared_directory"]
    clean_manifest = json.loads((prepared / "manifest.json").read_text())
    for key in (
        "model_id",
        "revision",
        "max_input_length",
        "label_to_id",
        "ignore_index",
        "alignment",
    ):
        if clean_manifest["tokenizer"][key] != tokenizer_config[key]:
            raise ValueError(f"Clean preparation and current tokenizer disagree: {key}")
    if (
        tokenizer_config["label_to_id"] != LABEL_TO_ID
        or tokenizer_config["ignore_index"] != IGNORE_INDEX
        or tokenizer_config["alignment"] != "first_subword"
    ):
        raise ValueError("Tokenizer settings disagree with implemented alignment.")
    item = next(item for item in data_manifest["files"] if item["split"] == split)
    raw_path = ROOT / data_manifest["local_directory"] / item["filename"]
    if file_digest(raw_path) != item["sha256"] or raw_path.stat().st_size != item["bytes"]:
        raise ValueError("Raw validation differs from the pinned release.")
    if item not in clean_manifest["source"]["files"]:
        raise ValueError("Clean preparation used a different raw validation source.")
    clean, quarantine = partition_records(read_split(raw_path))
    if clean["row_id"].tolist() != clean_manifest["splits"][split]["eligible_row_ids"]:
        raise ValueError("Eligible IDs differ from clean-v1.")
    validate_source(clean)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_config["model_id"],
        revision=tokenizer_config["revision"],
        cache_dir=str(ROOT / tokenizer_config["cache_directory"]),
        local_files_only=True,
        use_fast=True,
        trust_remote_code=False,
        token=False,
    )
    if (
        not isinstance(tokenizer, PreTrainedTokenizerFast)
        or tokenizer.model_max_length != tokenizer_config["max_input_length"]
    ):
        raise ValueError("Expected the pinned fast tokenizer and recorded position limit.")
    started = time.monotonic()
    disable_progress_bars()
    report = {
        "version": config["version"],
        "split": split,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "noise_config": config,
        "tokenizer_config": tokenizer_config,
        "noise_config_sha256": file_digest(config_path),
        "raw_source": item,
        "data_revision": data_manifest["revision"],
        "clean_manifest_sha256": file_digest(prepared / "manifest.json"),
        "quarantined_rows": len(quarantine),
        "packages": {
            name: version(name) for name in ("transformers", "tokenizers", "datasets", "pandas")
        },
        "python": platform.python_version(),
        "unicode": unicodedata.unidata_version,
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip(),
        "git_status_at_start": subprocess.run(
            ["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout,
        "source_sha256": {},
        "conditions": {},
        "model_weights_loaded": False,
        "training_performed": False,
        "model_evaluation_performed": False,
        "test_text_read": False,
        "authored_operator_examples": {
            "case_loss": {word: noise.lower(word) for word in ("IŞIK", "İzmir")},
            "diacritic_loss": {word: noise.fold(word) for word in ("ışık", "Çocuk")},
            "typo_swap": {"gittim": noise.transform("gittim", "typo_swap", swap_index=3)},
        },
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".noise-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "prepared"
        staging.mkdir()
        for source in SOURCE_FILES:
            snapshot = staging / "source_snapshot" / source
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / source, snapshot)
            report["source_sha256"][source] = file_digest(snapshot)
        shutil.copy2(config_path, staging / "source_snapshot/noise-config.json")
        if (
            file_digest(staging / "source_snapshot/noise-config.json")
            != report["noise_config_sha256"]
        ):
            raise ValueError("Noise config changed during startup.")
        write_json(staging / "noise-config.json", config)
        write_json(staging / "tokenizer-config.json", tokenizer_config)
        clean_encoded, clean_audit = encode_and_audit(
            clean,
            tokenizer,
            max_length=tokenizer_config["max_input_length"],
            batch_size=audit_config["tokenizer_batch_size"],
        )
        old_clean = load_from_disk(str(prepared / "dataset"))[split]
        for column in clean_encoded.column_names:
            if clean_encoded[column][:] != old_clean[column][:]:
                raise ValueError(f"Re-encoded clean data differs from clean-v1: {column}")
        report["clean_tokenization"] = clean_audit
        clean_encoded.save_to_disk(str(staging / "clean/dataset"))
        del clean_encoded, old_clean
        print(f"clean: {len(clean)} rows; exact clean-v1 inputs/targets reproduced", flush=True)

        def prepare_condition(
            name: str,
            families: list[str],
            plans: NoisePlans,
            rate: int | None,
            level: str,
            seed: int | None,
        ) -> dict:
            safe_name(name)
            if name in report["conditions"]:
                raise ValueError(f"Duplicate condition name: {name}")
            condition_start = time.monotonic()
            noisy, edits = apply_noise(
                clean, noise, plans, families=families, coverage_percent=rate
            )
            changes, examples = audit_changes(
                clean,
                noisy,
                edits,
                example_seed=seed if seed is not None else 0,
                condition_name=name,
                example_count=audit_config["examples_per_condition"],
            )
            for family in families:
                pool = plans.positions[family]
                expected = len(pool) if rate is None else plans.total_words * rate // 100
                actual = changes["families"][family]
                if actual["selected_words"] != expected:
                    raise ValueError("Selected-word budget differs from the requested rate.")
                if (
                    actual["selected_positions_sha256"]
                    != stable_digest(sorted(pool[:expected])).hex()
                ):
                    raise ValueError("Observed selection differs from the clean-source plan.")
                if len(families) == 1 and changes["changed_words"] != expected:
                    raise ValueError("Single-family realized corruption differs from the budget.")
            encoded, token_audit = encode_and_audit(
                noisy,
                tokenizer,
                max_length=tokenizer_config["max_input_length"],
                batch_size=audit_config["tokenizer_batch_size"],
            )
            output = staging / "conditions" / name
            output.mkdir(parents=True)
            write_jsonl(output / "records.jsonl", noisy.to_dict(orient="records"))
            write_jsonl(output / "edits.jsonl", edits)
            write_json(output / "examples.json", examples)
            encoded.save_to_disk(str(output / "dataset"))
            summary = {
                "families": families,
                "level": level,
                "seed": seed,
                "coverage_percent": rate,
                "eligible_words": {f: len(plans.positions[f]) for f in FAMILIES},
                "changes": changes,
                "tokenization": token_audit,
                "mean_subwords_per_word_delta": token_audit["mean_subwords_per_word"]
                - clean_audit["mean_subwords_per_word"],
                "output_sha256": {
                    file: file_digest(output / file)
                    for file in ("records.jsonl", "edits.jsonl", "examples.json")
                },
                "elapsed_seconds": time.monotonic() - condition_start,
            }
            write_json(output / "audit.json", summary)
            report["conditions"][name] = summary
            print(
                f"{name}: {changes['changed_words']} changed words; "
                f"max {token_audit['length_including_special_tokens']['max']} tokens",
                flush=True,
            )
            return summary

        for seed in config["evaluation_seeds"]:
            plans = build_plans(clean, noise, split=split, seed=seed)
            previous_masks = {family: set() for family in FAMILIES}
            for level, setting in config["levels"].items():
                rate = setting["coverage_percent"]
                for family in (*FAMILIES, "combined"):
                    families = (
                        config["combined"]["apply_order"] if family == "combined" else [family]
                    )
                    name = f"{family}-{level}-seed{seed}"
                    summary = prepare_condition(name, families, plans, rate, level, seed)
                    if family != "combined":
                        mask = set(plans.positions[family][: plans.total_words * rate // 100])
                        if not previous_masks[family].issubset(mask):
                            raise ValueError("Severity selections are not nested.")
                        previous_masks[family] = mask
                    else:
                        for component in FAMILIES:
                            single = report["conditions"][f"{component}-{level}-seed{seed}"]
                            key = "selected_positions_sha256"
                            if (
                                single["changes"]["families"][component][key]
                                != summary["changes"]["families"][component][key]
                            ):
                                raise ValueError(
                                    "Combined selection differs from its single-family plan."
                                )
            if ablations["enabled_by_default"]:
                level = ablations["level"]
                for item_name in ablations["conditions"]:
                    removed = item_name.removeprefix("combined_without_")
                    remaining = [f for f in config["combined"]["apply_order"] if f != removed]
                    prepare_condition(
                        f"{item_name}-{level}-seed{seed}",
                        remaining,
                        plans,
                        config["levels"][level]["coverage_percent"],
                        level,
                        seed,
                    )
        for item in full_conditions:
            prepare_condition(item["name"], [item["family"]], plans, None, "full", None)
        report["nested_masks_and_combined_component_identity_passed"] = True
        report["elapsed_seconds"] = time.monotonic() - started
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(staging / "manifest.json", report)
        # Detect code/config edits during execution before publishing a successful run.
        for source, digest in report["source_sha256"].items():
            if file_digest(ROOT / source) != digest:
                raise ValueError(
                    f"Source changed during preparation; rerun with stable sources: {source}"
                )
        if file_digest(config_path) != report["noise_config_sha256"]:
            raise ValueError("Noise config changed during preparation.")
        staging.rename(destination)
    print(
        f"Saved {len(report['conditions'])} noisy validation conditions: {destination}", flush=True
    )


if __name__ == "__main__":
    main()
