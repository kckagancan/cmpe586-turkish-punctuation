"""Separate GPU execution gates for immutable CPU-prepared transfer inputs."""

import json

import pandas as pd
from datasets import load_from_disk

from turkish_punctuation.heldout import execution_sources, load_plan, load_test, verify_commit
from turkish_punctuation.noise_audit import encode_and_audit, validate_source
from turkish_punctuation.training_io import file_hashes, sha256, value_digest
from turkish_punctuation.transfer_audit import audit_transfer_changes
from turkish_punctuation.transfer_noise import (
    CONTROL,
    TransferNoise,
    build_transfer_plans,
    position_key,
)
from turkish_punctuation.transfer_preparation import bindings, read, transfer_tokenizer

EXECUTION_SOURCES = (
    "configs/transfer-execution.json", "src/turkish_punctuation/transfer_execution.py",
    "scripts/ready_transfer_evaluation.py", "scripts/evaluate_transfer.py",
    "scripts/package_transfer_execution.py", "scripts/restore_transfer_colab.py",
)


def load_execution(root, input_root):
    """Keep preparation's CPU package gate intact; enforce GPU numerics separately."""
    execution = read(root / "configs/transfer-execution.json")
    config = read(root / execution["protocol_config"])
    original, _, old_freeze, cohorts = load_plan(root, input_root)
    if (execution["version"] != "transfer-execution-v1" or execution["scoring_enabled_by_default"]
            or execution["protocol_config_sha256"] != sha256(root / execution["protocol_config"])
            or execution["model_order"] != config["model_order"]
            or config["model_order"] != list(original["models"])
            or execution["conditions"] != config["run_matrix"]["conditions"]
            or execution["planned_forwards"] != 52 or execution["planned_cohort_scores"] != 104
            or execution["output_directory"] != config["paths"]["output_directory"]
            or config["inference"] != original["inference"]):
        raise ValueError("Execution differs from the fixed supplemental matrix/inference protocol")
    directory = input_root / config["paths"]["freeze_directory"]
    marker = verify_commit(directory)
    freeze = read(directory / "freeze.json")
    spec = freeze["specification"]
    if (freeze["identity"] != marker["identity"] or freeze["identity"] != execution["freeze_identity"]
            or freeze["identity"] != value_digest(spec) or spec["source_sha256"] != bindings(root)
            or spec["original_freeze_identity"] != old_freeze["identity"]
            or spec["cohorts_sha256"] != value_digest(cohorts)):
        raise ValueError("Transfer freeze/source/original cohort identity differs")
    for name, expected in spec["source_sha256"].items():
        if sha256(directory / "source_snapshot" / name) != expected:
            raise ValueError("Frozen transfer source snapshot changed")
    prepared = input_root / config["paths"]["prepared_directory"]
    marker = verify_commit(prepared)
    manifest = read(prepared / "manifest.json")
    if (sha256(prepared / "manifest.json") != execution["input_manifest_sha256"]
            or sha256(prepared / "complete.json") != execution["input_commit_sha256"]
            or manifest["identity"] != marker["identity"] or marker["identity"] != execution["input_identity"]
            or marker["identity"] != value_digest(manifest["specification"])
            or manifest["freeze_identity"] != freeze["identity"]
            or manifest["specification"]["source_sha256"] != bindings(root)
            or list(manifest["conditions"]) != execution["conditions"]
            or any(manifest[k] for k in ["model_weights_loaded", "model_evaluation_performed", "training_performed"])):
        raise ValueError("Prepared input identity or condition set changed")
    reference = config["clean_encoding_reference"]
    if (file_hashes(input_root / reference["directory"]) != reference["files"]
            or sha256(input_root / reference["source_manifest"]) != reference["source_manifest_sha256"]):
        raise ValueError("Reused clean encoding reference changed")
    # Compare the historical preparation runtime to its historical reference,
    # not the deliberately different GPU runtime running inference here.
    clean_manifest = read(input_root / reference["source_manifest"])
    for key in ("packages", "python"):
        if manifest["specification"][key] != clean_manifest["specification"][key]:
            raise ValueError("Recorded CPU preparation runtime drifted before input freeze")
    profile = config["profile"]
    if sha256(root / profile["path"]) != profile["sha256"]:
        raise ValueError("Frozen morphology profile changed")
    noise = TransferNoise(read(root / profile["path"]), read(root / original["noise"]["recipe"]))
    clean, source = load_test(root, input_root, original, cohorts)
    validate_source(clean)
    if (manifest["specification"]["raw_source_sha256"] != source["sha256"]
            or manifest["eligible_row_ids"] != clean["row_id"].tolist()):
        raise ValueError("Prepared original source IDs/bytes changed")
    return config, execution, original, freeze, cohorts, manifest, clean, noise


def view_path(input_root, config, name):
    return (input_root / config["clean_encoding_reference"]["directory"] if name == "clean-anchor"
            else input_root / config["paths"]["prepared_directory"] / "conditions" / name / "dataset")


def records_for(input_root, config, name, clean):
    if name == "clean-anchor":
        return clean
    path = input_root / config["paths"]["prepared_directory"] / "conditions" / name / "records.jsonl"
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines()])


def verify_inputs(root, input_root, *, retokenize=False):
    context = load_execution(root, input_root)
    config, execution, _, _, cohorts, manifest, clean, noise = context
    tokenizer = transfer_tokenizer(root, config)
    sensitive_ids = set(cohorts["sensitivity_row_ids"])
    sensitive_clean = clean[clean["row_id"].isin(sensitive_ids)]
    prepared = input_root / config["paths"]["prepared_directory"]
    plans = {s: build_transfer_plans(clean, noise, split="test", seed=s) for s in config["noise"]["seeds"]}
    for seed, plan in plans.items():
        if read(prepared / "plans" / f"seed{seed}.json") != json.loads(json.dumps(plan)):
            raise ValueError("Saved masks differ from independent deterministic word ranking")
    for name in execution["conditions"]:
        audit = manifest["conditions"][name]
        noisy = records_for(input_root, config, name, clean)
        if name == "clean-anchor":
            edits, plan = [], None
        else:
            path = prepared / "conditions" / name
            if read(path / "audit.json") != audit:
                raise ValueError("Condition audit differs from immutable manifest")
            edits = [json.loads(line) for line in (path / "edits.jsonl").read_text().splitlines()]
            plan = plans[audit["seed"]]
            family = audit["family"]
            selector = "partial_folding" if family == CONTROL else family
            for edit in edits:
                word = edit["before"]
                expected = ({"positions": noise.foldable(word), "target_foldable": len(noise.foldable(word))}
                            if family == CONTROL else noise.parameters(word, family,
                                position_key(audit["seed"], "test", edit["row_id"], edit["word_index"], selector)))
                if edit["parameters"] != expected:
                    raise ValueError("Saved morphology parameters differ from pinned profile/key")
        changes = audit_transfer_changes(clean, noisy, edits, noise, audit["family"], plan)
        sensitivity = audit_transfer_changes(sensitive_clean, noisy[noisy["row_id"].isin(sensitive_ids)],
            [e for e in edits if e["row_id"] in sensitive_ids], noise, audit["family"])
        if (json.loads(json.dumps(changes)) != audit["changes"]
                or json.loads(json.dumps(sensitivity)) != audit["sensitivity_changes"]):
            raise ValueError("Source/target/exposure/shape audit differs")
        saved = load_from_disk(str(view_path(input_root, config, name)))
        if saved["row_id"][:] != clean["row_id"].tolist():
            raise ValueError("Original encoded row IDs/order changed")
        if retokenize:
            fresh, token_audit = encode_and_audit(noisy, tokenizer, max_length=512, batch_size=256)
            if (set(fresh.column_names) != set(saved.column_names)
                    or any(fresh[key][:] != saved[key][:] for key in fresh.column_names)
                    or json.loads(json.dumps(token_audit)) != audit["tokenization"]):
                raise ValueError("Fresh GPU-runtime tokenization differs from saved inputs/labels")
        print(f"TRANSFER_INPUT_OK {name}; fresh_tokenization={retokenize}", flush=True)
    return context


def source_hashes(root, original_freeze):
    return {**execution_sources(root, original_freeze), **bindings(root),
            **{name: sha256(root / name) for name in EXECUTION_SOURCES}}


def runtime_specification(root, input_root, context, runtime):
    config, execution, original, freeze, _, _, _, _ = context
    _, _, old_freeze, _ = load_plan(root, input_root)
    return {"version": "transfer-runtime-v1", "freeze_identity": freeze["identity"],
        "execution_config_sha256": sha256(root / "configs/transfer-execution.json"),
        "input_identity": execution["input_identity"], "input_commit_sha256": execution["input_commit_sha256"],
        "source_sha256": source_hashes(root, old_freeze), "runtime": runtime,
        "inference": config["inference"], "models": original["models"],
        "model_order": execution["model_order"], "conditions": execution["conditions"]}


def verify_readiness(root, input_root, path, context, runtime):
    marker = verify_commit(path)
    receipt = read(path / "readiness.json")
    expected = runtime_specification(root, input_root, context, runtime)
    if (receipt["status"] != "ready_for_transfer_scoring" or receipt["specification"] != expected
            or receipt["identity"] != value_digest(expected) or marker["identity"] != receipt["identity"]
            or receipt["optimizer_updates"] != 0 or receipt["metrics_computed"]
            or receipt["test_predictions_computed"] or receipt["probe_data_split"] != "validation"):
        raise ValueError("Readiness differs from actual inputs/models/runtime/sources")
    for name, digest in expected["source_sha256"].items():
        if sha256(path / "source_snapshot" / name) != digest:
            raise ValueError("Readiness snapshot changed")
    return receipt


def ordered_inputs(input_root, config, clean):
    dataset = load_from_disk(str(view_path(input_root, config, "clean-anchor")))
    lengths, ids = [len(x) for x in dataset["input_ids"]], dataset["row_id"][:]
    order = sorted(range(len(dataset)), key=lambda i: (lengths[i], ids[i]))
    rows = {row["row_id"]: row for row in clean.to_dict(orient="records")}
    return order, {ids[i]: rows[ids[i]] for i in order}
