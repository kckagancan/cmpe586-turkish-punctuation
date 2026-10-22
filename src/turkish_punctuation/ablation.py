"""Readiness checks for the prespecified case-loss training ablation."""

import json
import math
import sys
from pathlib import Path

from turkish_punctuation.training import training_budget, validate_training_config
from turkish_punctuation.training_io import file_hashes, sha256, value_digest

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "configs/ablation-case-loss.json"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_plan() -> tuple[dict, dict]:
    """Keep the numerical recipe byte-identical to the completed augmented arm."""
    plan = read_json(PLAN)
    path = ROOT / plan["training_config"]
    if sha256(path) != plan["training_config_sha256"]:
        raise ValueError("The fixed training configuration changed.")
    config = read_json(path)
    validate_training_config(config)
    if (
        plan["without_family"] != "case_loss"
        or plan["redistribute_removed_budget"] is not False
        or plan["seed"] != 586
        or plan["formal_training_enabled_by_default"] is not False
    ):
        raise ValueError("This workflow implements the fixed case-loss ablation only.")
    return plan, config


def verify_view_files(directory: Path, omitted: str | None) -> dict:
    """Check the immutable preparation and all saved epoch bytes, without inference."""
    plan, config = load_plan()
    manifest = read_json(directory / "manifest.json")
    spec = manifest["specification"]
    if (
        manifest["identity"] != value_digest(spec)
        or spec["without_family"] != omitted
        or spec["training_seed"] != plan["seed"]
        or spec["augmentation"] != config["augmentation"]
        or spec["epochs"] != config["optimization"]["epochs"]
        or manifest["training_performed"] is not False
        or manifest["test_accessed"] is not False
    ):
        raise ValueError("Epoch views belong to another experiment.")
    for name, expected in spec["source_sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Prepared-view source changed: {name}")
    for epoch in range(1, config["optimization"]["epochs"] + 1):
        if file_hashes(directory / f"epoch-{epoch:02d}") != manifest["epochs"][str(epoch)]["files"]:
            raise ValueError(f"Epoch {epoch}: saved view files changed.")
    return manifest


def audit_view_pair(views: Path, reference: Path) -> dict:
    """Compare row selection and retained word plans to the original augmented views."""
    plan, config = load_plan()
    current = verify_view_files(views, plan["without_family"])
    original = verify_view_files(reference, None)
    for key, value in original["specification"].items():
        if key != "without_family" and current["specification"][key] != value:
            raise ValueError(f"View preparation changed beyond the omitted family: {key}")
    epochs = {}
    for epoch in range(1, config["optimization"]["epochs"] + 1):
        a = current["epochs"][str(epoch)]["audit"]
        b = original["epochs"][str(epoch)]["audit"]
        for key in ("epoch", "training_seed", "noise_seed", "selected_view_rows",
                    "guaranteed_clean_view_rows", "selected_row_ids_sha256"):
            if a[key] != b[key]:
                raise ValueError(f"Epoch {epoch}: row selection/seed differs: {key}")
        if a["applied_families"] != plan["remaining_families"]:
            raise ValueError("Retained operation order differs.")
        for family in plan["remaining_families"]:
            if a["full_split_word_plans"][family] != b["full_split_word_plans"][family]:
                raise ValueError(f"Epoch {epoch}: retained word mask differs: {family}")

        def retained_operations(directory: Path) -> list[tuple]:
            with (directory / f"epoch-{epoch:02d}/edits.jsonl").open() as stream:
                edits = [json.loads(line) for line in stream]
            return [(e["family"], e["row_id"], e["word_index"], e.get("swap_index"))
                    for e in edits if e["family"] != plan["without_family"]]

        if retained_operations(views) != retained_operations(reference):
            raise ValueError(f"Epoch {epoch}: retained operation positions/parameters differ.")
        if (
            a["changes"]["words"] != config["data"]["expected_train_words"]
            or not a["changes"]["invariants_passed"]
            or not a["tokenization"]["all_first_subword_targets_preserved"]
            or a["tokenization"]["length_including_special_tokens"]["max"] > 512
        ):
            raise ValueError("Ablation changed targets or violated a preparation invariant.")
        epochs[str(epoch)] = {
            "selected_view_rows": a["selected_view_rows"],
            "selected_row_ids_sha256": a["selected_row_ids_sha256"],
            "changed_words": a["changes"]["changed_words"],
            "changed_word_percent": a["changes"]["changed_word_percent"],
            "reference_changed_word_percent": b["changes"]["changed_word_percent"],
            "max_subwords": a["tokenization"]["length_including_special_tokens"]["max"],
            "encoded_content_sha256": a["encoded_content_sha256"],
        }
    return {
        "status": "passed", "plan_sha256": sha256(PLAN),
        "prepared_identity": current["identity"], "reference_identity": original["identity"],
        "same_selected_rows": True, "same_retained_word_plans_and_operations": True,
        "redistributed_budget": False, "epochs": epochs,
        "training_performed": False, "test_accessed": False,
    }


def training_command(storage: Path, *, mode: str, resume: bool = False) -> list[str]:
    """Build the existing runner's command; callers explicitly decide whether to run it."""
    plan, _ = load_plan()
    if mode not in ("calibrate", "train") or (resume and mode != "train"):
        raise ValueError("Unsupported ablation execution mode.")
    command = [sys.executable, "scripts/train.py", "--config", str(ROOT / plan["training_config"]),
               "--mode", mode, "--arm", "augmented", "--seed", str(plan["seed"]),
               "--without-family", plan["without_family"],
               "--augmentation-directory", str(storage / plan["storage_views"]),
               "--paired-clean-run", str(storage / plan["reference_runs"]["clean"]["storage_directory"]),
               "--initialization-directory", str(storage / plan["initialization"]["storage_directory"]),
               "--output-directory", str(storage / plan["storage_calibration" if mode == "calibrate" else "storage_training"]),
               "--download-model"]
    if mode == "calibrate":
        command += ["--calibration-steps", str(plan["calibration_steps"])]
    if resume:
        command.append("--resume")
    return command


def verify_ready(storage: Path, *, allow_started: bool = False) -> dict:
    """Bind the separate calibration, reference runs, input hashes and current GPU runtime."""
    # This configures deterministic CUDA; it does not create a model or optimizer.
    from importlib.util import module_from_spec, spec_from_file_location

    plan, config = load_plan()
    audit = audit_view_pair(storage / plan["storage_views"], storage / plan["storage_reference_views"])
    views = read_json(storage / plan["storage_views"] / "manifest.json")
    budget = training_budget(config["data"]["expected_train_records"], config)
    for arm, reference in plan["reference_runs"].items():
        directory = storage / reference["storage_directory"]
        if sha256(directory / "run.json") != reference["run_json_sha256"] or sha256(directory / "summary.json") != reference["summary_sha256"]:
            raise ValueError(f"Completed {arm} reference metadata differs.")
        run, summary = read_json(directory / "run.json"), read_json(directory / "summary.json")
        spec = run["specification"]
        if (value_digest(spec) != run["identity"] or run["identity"] != reference["identity"]
            or summary["global_step"] != budget["total_updates"] or run["budget"] != budget
            or spec["initialization_tensors_sha256"] != plan["initialization"]["model_tensors_sha256"]):
            raise ValueError("Reference identity, initialization or update budget differs.")
        for key in ("model", "data", "optimization", "runtime", "seeds", "selection", "hyperparameter_search"):
            if spec["configuration"][key] != config[key]:
                raise ValueError(f"Matched numerical recipe differs: {arm}/{key}")
    initialization = storage / plan["initialization"]["storage_directory"] / f"seed-{plan['seed']}"
    initial = read_json(initialization / "initialization.json")
    if (initial["identity"] != plan["initialization"]["identity"]
        or initial["model_tensors_sha256"] != plan["initialization"]["model_tensors_sha256"]
        or file_hashes(initialization, exclude=("initialization.json",)) != initial["files"]):
        raise ValueError("The shared initial model changed.")
    directory = storage / plan["storage_calibration"]
    run, result = read_json(directory / "run.json"), read_json(directory / "calibration.json")
    spec = run["specification"]
    progress = read_json(directory / "progress.json")
    if (
        value_digest(spec) != run["identity"] or result["identity"] != run["identity"]
        or spec["mode"] != "calibrate" or spec["arm"] != plan["arm"]
        or spec["without_family"] != plan["without_family"] or spec["seed"] != plan["seed"]
        or spec["configuration"] != config or spec["inputs"]["augmentation"]["identity"] != views["identity"]
        or spec["inputs"]["augmentation"]["manifest_sha256"] != sha256(storage / plan["storage_views"] / "manifest.json")
        or spec["initialization_identity"] != plan["initialization"]["identity"]
        or spec["initialization_tensors_sha256"] != plan["initialization"]["model_tensors_sha256"]
        or result["optimizer_updates"] != plan["calibration_steps"]
        or not result["longest_prepared_batch_included"] or result["weights_retained"]
        or progress["status"] != "complete" or (directory / "models").exists()
    ):
        raise ValueError("A matching completed, discarded-weight calibration is required.")
    paths = {"training-config.json": ROOT / plan["training_config"],
             "tokenizer-config.json": ROOT / config["model"]["source_config"],
             "noise-config.json": ROOT / config["augmentation"]["noise_config"]}
    for name, expected in spec["source_hashes"].items():
        if sha256(paths.get(name, ROOT / name)) != expected:
            raise ValueError(f"Training source changed after calibration: {name}")
    module_spec = spec_from_file_location("ablation_training_environment", ROOT / "scripts/train.py")
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("Training environment reader unavailable.")
    module = module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    current_environment = module.environment()
    for key in ("python", "packages", "gpu", "precision", "tf32", "deterministic_algorithms"):
        if current_environment[key] != spec["environment"][key]:
            raise ValueError(f"Runtime changed after calibration: {key}")
    with (directory / "events.jsonl").open() as stream:
        events = [json.loads(line) for line in stream]
    if ([e["global_step"] for e in events] != list(range(1, plan["calibration_steps"] + 1))
        or any(not math.isfinite(e["window_loss"]) for e in events)):
        raise ValueError("Calibration updates/losses are incomplete or non-finite.")
    formal = storage / plan["storage_training"]
    if not allow_started and formal.exists():
        raise FileExistsError("Formal ablation already has an output directory; do not overwrite it.")
    return {"status": "ready", "formal_training_started": formal.exists(),
            "formal_training_enabled": False, "plan_sha256": sha256(PLAN),
            "calibration_identity": run["identity"], "calibration_optimizer_updates": len(events),
            "calibration_weights_reused": False, "view_preparation_optimizer_updates": 0,
            "view_audit": audit, "budget": budget, "environment": current_environment,
            "test_accessed": False, "formal_command": training_command(storage, mode="train")}
