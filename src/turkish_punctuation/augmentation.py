"""Deterministic epoch training views, independently of model and dropout RNGs."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import PreTrainedTokenizerFast

from turkish_punctuation.noise import FAMILIES, WordNoise, apply_noise, build_plans, stable_digest
from turkish_punctuation.noise_audit import audit_changes, encode_and_audit, validate_source
from turkish_punctuation.preprocessing import encode_records
from turkish_punctuation.training_io import file_hashes, sha256, value_digest


def validate_augmentation(config: dict, noise_config: dict) -> None:
    """Reject unsupported protocol changes instead of silently approximating them."""
    expected = {
        "view": "combined",
        "level": "medium",
        "families": list(FAMILIES),
        "noisy_view_row_percent": 50,
        "row_count_rounding": "floor_N_times_percent_divided_by_100",
        "row_selection": "lowest_sha256_rank",
        "row_key_fields": [
            "training_version",
            "training_seed",
            "epoch_1_based",
            "row_id",
            "purpose",
        ],
        "row_key_purpose": "noisy_view",
        "row_key_serialization": "JSON array; ensure_ascii=False; separators=(',', ':'); UTF-8",
        "row_tie_break": "row_id",
        "excluded_row_key_fields": ["labels", "model_id", "arm", "ablated_family"],
        "word_plan_scope": "whole_clean_training_split_before_row_view_selection",
        "refresh": "once_per_epoch",
        "replace_selected_rows_in_place": True,
        "append_extra_records": False,
        "force_actual_edit_in_selected_row": False,
        "audit_realized_coverage_after_row_selection": True,
        "retokenize_and_realign": True,
    }
    if any(config["augmentation"].get(k) != v for k, v in expected.items()):
        raise ValueError(
            "Unsupported training augmentation rules; update protocol and code together."
        )
    if noise_config["levels"]["medium"]["coverage_percent"] != 5:
        raise ValueError("The paired training recipe specifies medium = 5% per family.")
    if (
        config["seeds"]["training_noise_seed_formula"]
        != "700000 + 10 * training_seed + epoch_1_based"
    ):
        raise ValueError("Unsupported epoch noise seed formula.")
    WordNoise(noise_config)


def select_noisy_rows(row_ids: list[int], config: dict, *, seed: int, epoch: int) -> set[int]:
    """Select row views with label-free hashes, independently of word noise/shuffling."""
    if epoch < 1 or len(row_ids) != len(set(row_ids)):
        raise ValueError("Expected a positive epoch and unique source row IDs.")
    count = len(row_ids) * config["augmentation"]["noisy_view_row_percent"] // 100
    ranked = sorted(
        row_ids,
        key=lambda row_id: (
            stable_digest([config["version"], seed, epoch, row_id, "noisy_view"]),
            row_id,
        ),
    )
    return set(ranked[:count])


def encoded_digest(dataset: Dataset) -> str:
    """Hash encoded values independently of Arrow/package serialization."""
    digest = hashlib.sha256()
    for batch in dataset.iter(batch_size=512):
        for i in range(len(batch["row_id"])):
            digest.update(stable_digest([{k: batch[k][i] for k in sorted(batch)}]))
    return digest.hexdigest()


def build_epoch_view(
    clean: pd.DataFrame,
    tokenizer: PreTrainedTokenizerFast,
    config: dict,
    noise_config: dict,
    *,
    seed: int,
    epoch: int,
    without_family: str | None = None,
) -> tuple[pd.DataFrame, Dataset, list[dict], dict, list[dict]]:
    """Plan on all clean words, then replace exactly half the rows in place."""
    validate_augmentation(config, noise_config)
    validate_source(clean)
    if without_family is not None and without_family not in FAMILIES:
        raise ValueError("Unknown omitted family.")
    noise_seed = 700000 + 10 * seed + epoch
    noise = WordNoise(noise_config)
    plans = build_plans(clean, noise, split="train", seed=noise_seed)
    families = [f for f in noise_config["combined"]["apply_order"] if f != without_family]
    full, edits = apply_noise(clean, noise, plans, families=families, coverage_percent=5)
    count = plans.total_words * 5 // 100
    masks = {
        f: {(e["row_id"], e["word_index"]) for e in edits if e["family"] == f} for f in families
    }
    for family in families:
        if masks[family] != set(plans.positions[family][:count]):
            raise ValueError("Full-split training word mask differs from the declared prefix.")
    selected = select_noisy_rows(clean["row_id"].tolist(), config, seed=seed, epoch=epoch)
    rows = []
    for original, noisy in zip(
        clean.to_dict(orient="records"), full.to_dict(orient="records"), strict=True
    ):
        rows.append(noisy if original["row_id"] in selected else original)
    mixed = pd.DataFrame(rows, columns=["row_id", "tokens", "tags"])
    used_edits = [e for e in edits if e["row_id"] in selected]
    changes, examples = audit_changes(
        clean,
        mixed,
        used_edits,
        example_seed=noise_seed,
        condition_name=f"training-epoch-{epoch:02d}",
        example_count=8,
    )
    encoded, tokenization = encode_and_audit(
        mixed, tokenizer, max_length=config["data"]["max_input_length"], batch_size=256
    )
    if changes["words"] != config["data"]["expected_train_words"] or len(encoded) != len(clean):
        raise ValueError("Augmentation changed the training record/word budget.")
    report = {
        "epoch": epoch,
        "training_seed": seed,
        "noise_seed": noise_seed,
        "without_family": without_family,
        "applied_families": families,
        "selected_view_rows": len(selected),
        "guaranteed_clean_view_rows": len(clean) - len(selected),
        "selected_row_ids_sha256": stable_digest(sorted(selected)).hex(),
        "selected_view_rows_without_actual_edits": len(selected) - changes["changed_rows"],
        "full_split_word_plans": {
            f: {"selected_words": count, "positions_sha256": stable_digest(sorted(masks[f])).hex()}
            for f in families
        },
        "changes": changes,
        "tokenization": tokenization,
        "encoded_content_sha256": encoded_digest(encoded),
    }
    return mixed, encoded, used_edits, report, examples


def load_epoch_views(
    directory: Path,
    clean: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    config: dict,
    *,
    seed: int,
    without_family: str | None = None,
) -> tuple[dict[int, Dataset], dict]:
    """Verify all persisted epoch views before creating an optimizer or updating weights."""
    manifest = json.loads((directory / "manifest.json").read_text())
    specification = manifest["specification"]
    if (
        value_digest(specification) != manifest["identity"]
        or specification["training_version"] != config["version"]
        or specification["training_seed"] != seed
        or specification["without_family"] != without_family
        or specification["augmentation"] != config["augmentation"]
        or specification["epochs"] != config["optimization"]["epochs"]
        or specification["clean_encoded_content_sha256"] != encoded_digest(clean)
        or manifest["training_performed"] is not False
        or manifest["test_accessed"] is not False
    ):
        raise ValueError("Prepared augmentation belongs to a different recipe/seed/clean input.")
    noise_path = Path(__file__).resolve().parents[2] / config["augmentation"]["noise_config"]
    if specification["noise_config"] != json.loads(noise_path.read_text()):
        raise ValueError("Prepared augmentation uses a different noise protocol.")
    root = Path(__file__).resolve().parents[2]
    for name, expected in specification["source_sha256"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Augmentation preparation code changed: {name}")
    expected_ids = list(clean["row_id"])
    result = {}
    for epoch in range(1, config["optimization"]["epochs"] + 1):
        path = directory / f"epoch-{epoch:02d}"
        audit = manifest["epochs"][str(epoch)]
        if file_hashes(path) != audit["files"]:
            raise ValueError(f"Epoch {epoch}: augmentation files changed.")
        with (path / "records.jsonl").open() as stream:
            records = pd.DataFrame([json.loads(line) for line in stream])
        dataset = load_from_disk(str(path / "dataset"))
        if not isinstance(dataset, Dataset) or list(dataset["row_id"]) != expected_ids:
            raise ValueError(f"Epoch {epoch}: augmentation row IDs/order differ.")
        if records["row_id"].tolist() != expected_ids:
            raise ValueError(f"Epoch {epoch}: source words have different IDs/order.")
        reproduced = encode_records(
            records, tokenizer, max_length=config["data"]["max_input_length"]
        )
        for start in range(0, len(clean), 512):
            old, new = clean[start : start + 512], dataset[start : start + 512]
            if any(
                [v for v in a if v != -100] != [v for v in b if v != -100]
                for a, b in zip(old["labels"], new["labels"], strict=True)
            ):
                raise ValueError(f"Epoch {epoch}: original word targets changed.")
            if reproduced[start : start + 512] != new:
                raise ValueError(f"Epoch {epoch}: prepared encoding differs from source words.")
        if encoded_digest(dataset) != audit["audit"]["encoded_content_sha256"]:
            raise ValueError(f"Epoch {epoch}: encoded-content digest changed.")
        if (
            not audit["audit"]["changes"]["invariants_passed"]
            or not audit["audit"]["tokenization"]["all_first_subword_targets_preserved"]
        ):
            raise ValueError(f"Epoch {epoch}: augmentation audit did not pass.")
        result[epoch] = dataset
    return result, manifest


def verify_clean_pair(
    directory: Path,
    config: dict,
    *,
    seed: int,
    inputs: dict,
    initialization: dict,
    hardware: dict,
    budget: dict,
) -> dict:
    """Require the completed clean run's data, initialization and numerical recipe."""
    run = json.loads((directory / "run.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    old = run["specification"]
    if (
        value_digest(old) != run["identity"]
        or not run["identity"] == summary["identity"] == config["paired_clean_run"]["identity"]
        or old["mode"] != "train"
        or old["arm"] != "clean"
        or old["seed"] != seed
        or len(summary["history"]) != old["configuration"]["optimization"]["epochs"]
        or summary["global_step"] != budget["total_updates"]
        or run["budget"] != budget
        or old["inputs"] != inputs
        or old["initialization_tensors_sha256"] != initialization["model_tensors_sha256"]
        or old["initialization_identity"] != initialization["identity"]
    ):
        raise ValueError("Augmented arm does not match the declared completed clean run.")
    for section in (
        "model",
        "data",
        "optimization",
        "runtime",
        "seeds",
        "selection",
        "hyperparameter_search",
    ):
        if old["configuration"][section] != config[section]:
            raise ValueError(f"Paired numerical recipe differs: {section}")
    for key in ("python", "packages", "gpu", "precision", "tf32", "deterministic_algorithms"):
        if old["environment"][key] != hardware[key]:
            raise ValueError(f"Paired training environment differs: {key}")
    return {
        "identity": run["identity"],
        "seed": seed,
        "run_json_sha256": sha256(directory / "run.json"),
        "summary_sha256": sha256(directory / "summary.json"),
        "matched_numerical_recipe_and_environment": True,
    }
