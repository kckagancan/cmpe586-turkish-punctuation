"""Read-only bindings and data preparation gates for the supplemental transfer study."""

import json
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from datasets import disable_progress_bars, load_from_disk
from transformers import AutoTokenizer

from turkish_punctuation.heldout import load_plan, load_test, verify_commit
from turkish_punctuation.noise_audit import encode_and_audit, validate_source
from turkish_punctuation.tokenization import LABEL_TO_ID
from turkish_punctuation.training_io import file_hashes, sha256, value_digest
from turkish_punctuation.transfer_audit import audit_transfer_changes
from turkish_punctuation.transfer_noise import (
    CONTROL,
    FAMILIES,
    VERSION,
    TransferNoise,
    build_transfer_plans,
    position_key,
)

CONFIG_PATH = "configs/transfer-evaluation.json"
SOURCES = (
    CONFIG_PATH, "configs/transfer-noise-profile.json", "docs/transfer-evaluation-protocol.md",
    "src/turkish_punctuation/transfer_noise.py", "src/turkish_punctuation/transfer_audit.py",
    "src/turkish_punctuation/transfer_preparation.py", "scripts/compile_transfer_profile.py",
    "scripts/freeze_transfer_protocol.py", "scripts/prepare_transfer_inputs.py",
    "src/turkish_punctuation/heldout.py", "src/turkish_punctuation/data.py",
    "src/turkish_punctuation/preprocessing.py", "src/turkish_punctuation/tokenization.py",
    "src/turkish_punctuation/noise.py", "src/turkish_punctuation/noise_audit.py",
    "src/turkish_punctuation/training_io.py",
)


def read(path):
    return json.loads(Path(path).read_text())


def bindings(root):
    return {name: sha256(root / name) for name in SOURCES}


def load_draft(root):
    config = read(root / CONFIG_PATH)
    expected_conditions = ["clean-anchor", *[f"{f}-corpus-seed{s}" for s in [586, 587, 588]
                                           for f in (*FAMILIES, CONTROL)]]
    if (config["version"] != "transfer-evaluation-v1" or config["noise"]["version"] != VERSION
            or config["noise"]["seeds"] != [586, 587, 588]
            or config["noise"]["families"] != list(FAMILIES) or config["noise"]["control"] != CONTROL
            or config["run_matrix"] != {"conditions": expected_conditions, "input_views": 13,
                                        "model_condition_forwards": 52, "cohort_score_sets": 104}
            or config["execution"] != {"training_enabled": False, "scoring_enabled_by_default": False,
                "new_model_scores_this_step": False, "require_separate_gpu_readiness_and_runner": True,
                "condition_specific_drops_truncation_or_budget_relaxation": False}):
        raise ValueError("Unsupported preparation-only transfer recipe")
    old_execution = config["original_execution"]
    if sha256(root / old_execution["path"]) != old_execution["sha256"]:
        raise ValueError("Original held-out execution binding changed")
    original, execution, freeze, cohorts = load_plan(root)
    if (old_execution["protocol_sha256"] != execution["protocol_config_sha256"]
            or old_execution["freeze_identity"] != freeze["identity"]
            or config["model_order"] != list(original["models"])
            or config["inference"] != original["inference"]):
        raise ValueError("Original model/cohort/inference identity changed")
    for key in ["primary", "punctuation_labels", "all_labels", "label_to_id", "alignment",
                "ignore_index", "zero_division", "retain_O_for_false_positive_and_false_negative_counts"]:
        if config["metrics"][key] != original["metrics"][key]:
            raise ValueError("Transfer target/metric convention changed")
    profile_settings = config["profile"]
    if sha256(root / profile_settings["path"]) != profile_settings["sha256"]:
        raise ValueError("Compiled corpus morphology profile changed")
    profile = read(root / profile_settings["path"])
    noise = TransferNoise(profile, read(root / original["noise"]["recipe"]))
    return config, noise, original, freeze, cohorts


def verify_clean_reference(root, config):
    reference = config["clean_encoding_reference"]
    directory = root / reference["directory"]
    if (sha256(root / reference["source_manifest"]) != reference["source_manifest_sha256"]
            or file_hashes(directory) != reference["files"]):
        raise ValueError("Read-only original clean encoding/reference changed")
    manifest = read(root / reference["source_manifest"])
    actual = {name: version(name) for name in ("transformers", "tokenizers", "datasets", "pandas")}
    if (manifest["specification"]["packages"] != actual
            or manifest["specification"]["python"] != platform.python_version()):
        raise ValueError("Local tokenizer preparation runtime differs from the completed original inputs")
    return manifest["conditions"]["clean-anchor"]["tokenization"]


def transfer_tokenizer(root, config):
    settings = config["tokenizer"]
    if sha256(root / settings["config"]) != settings["config_sha256"]:
        raise ValueError("Tokenizer configuration changed")
    snapshot = root / settings["local_snapshot"]
    for name, expected in settings["files"].items():
        if sha256(snapshot / name) != expected:
            raise ValueError(f"Cached tokenizer file changed: {name}")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, use_fast=True,
                                             trust_remote_code=False, token=False)
    if not tokenizer.is_fast or config["metrics"]["label_to_id"] != LABEL_TO_ID:
        raise ValueError("Fast-tokenizer or original labels differ")
    return tokenizer


def load_frozen(root):
    config, noise, original, old_freeze, cohorts = load_draft(root)
    directory = root / config["paths"]["freeze_directory"]
    committed = verify_commit(directory)
    freeze = read(directory / "freeze.json")
    if (freeze["identity"] != committed["identity"]
            or freeze["identity"] != value_digest(freeze["specification"])
            or freeze["specification"]["source_sha256"] != bindings(root)
            or freeze["specification"]["original_freeze_identity"] != old_freeze["identity"]
            or freeze["specification"]["cohorts_sha256"] != value_digest(cohorts)):
        raise ValueError("Frozen transfer protocol/source/cohort binding differs")
    if (root / config["paths"]["output_directory"]).exists():
        raise FileExistsError("Scoring namespace exists; preparation must not run after scores")
    verify_clean_reference(root, config)
    return config, noise, original, freeze, cohorts


def verify_prepared(root, *, retokenize=False):
    """Reconcile saved text/edits/encodings with pinned raw targets and fixed masks."""
    config, noise, original, freeze, cohorts = load_frozen(root)
    prepared = root / config["paths"]["prepared_directory"]
    marker = verify_commit(prepared)
    manifest = read(prepared / "manifest.json")
    spec = manifest["specification"]
    if (manifest["freeze_identity"] != freeze["identity"] or spec["source_sha256"] != bindings(root)
            or marker["identity"] != manifest["identity"] or marker["identity"] != value_digest(spec)
            or list(manifest["conditions"]) != config["run_matrix"]["conditions"]
            or any(manifest[k] for k in ["model_weights_loaded", "model_evaluation_performed", "training_performed"])):
        raise ValueError("Completed transfer input identity differs")
    clean, source = load_test(root, root, original, cohorts)
    validate_source(clean)
    if (spec["raw_source_sha256"] != source["sha256"]
            or spec["source_row_ids_sha256"] != value_digest(clean["row_id"].tolist())
            or spec["cohorts_sha256"] != value_digest(cohorts)):
        raise ValueError("Saved inputs bind different source rows/cohorts")
    sensitivity_ids = set(cohorts["sensitivity_row_ids"])
    sensitive_clean = clean[clean["row_id"].isin(sensitivity_ids)]
    plans = {seed: build_transfer_plans(clean, noise, split="test", seed=seed) for seed in config["noise"]["seeds"]}
    for seed, plan in plans.items():
        if read(prepared / "plans" / f"seed{seed}.json") != json.loads(json.dumps(plan)):
            raise ValueError("Saved selection masks differ from independently ranked original words")
    tokenizer = transfer_tokenizer(root, config) if retokenize else None
    disable_progress_bars()
    audited = {}
    for name, audit in manifest["conditions"].items():
        family = audit["family"]
        if family is None:
            noisy, edits, plan = clean, [], None
            if audit["encoding_reference"] != config["clean_encoding_reference"]:
                raise ValueError("Clean anchor does not reuse its exact read-only reference")
            encoded_path = root / config["clean_encoding_reference"]["directory"]
        else:
            directory = prepared / "conditions" / name
            if read(directory / "audit.json") != audit:
                raise ValueError("Saved condition audit differs from manifest")
            noisy = pd.DataFrame([json.loads(line) for line in (directory / "records.jsonl").read_text().splitlines()])
            edits = [json.loads(line) for line in (directory / "edits.jsonl").read_text().splitlines()]
            plan = plans[audit["seed"]]
            selector = "partial_folding" if family == CONTROL else family
            for edit in edits:
                before = edit["before"]
                expected = ({"positions": noise.foldable(before), "target_foldable": len(noise.foldable(before))}
                            if family == CONTROL else noise.parameters(before, family,
                                position_key(audit["seed"], "test", edit["row_id"], edit["word_index"], selector)))
                if edit["parameters"] != expected:
                    raise ValueError("Saved amplitude/template/placement differs from deterministic source recipe")
            encoded_path = directory / "dataset"
        changes = audit_transfer_changes(clean, noisy, edits, noise, family, plan)
        sensitivity = audit_transfer_changes(sensitive_clean, noisy[noisy["row_id"].isin(sensitivity_ids)],
            [e for e in edits if e["row_id"] in sensitivity_ids], noise, family)
        if (json.loads(json.dumps(changes)) != audit["changes"]
                or json.loads(json.dumps(sensitivity)) != audit["sensitivity_changes"]):
            raise ValueError("Recomputed saved-text exposure/shape/target audit differs")
        saved = load_from_disk(str(encoded_path))
        if saved["row_id"][:] != clean["row_id"].tolist():
            raise ValueError("Saved encoding changed original row IDs/order")
        for index, tags in enumerate(clean["tags"]):
            if [x for x in saved[index]["labels"] if x != -100] != [LABEL_TO_ID[t] for t in tags]:
                raise ValueError("Saved encoding changed original first-subword targets")
        if retokenize:
            fresh, token_audit = encode_and_audit(noisy, tokenizer, max_length=512, batch_size=256)
            if (set(fresh.column_names) != set(saved.column_names)
                    or any(fresh[n][:] != saved[n][:] for n in fresh.column_names)
                    or json.loads(json.dumps(token_audit)) != audit["tokenization"]):
                raise ValueError("Saved encoding differs from fresh tokenizer/alignment audit")
            del fresh
        audited[name] = {"changed_words": changes["changed_words"],
                         "max_input_length": audit["tokenization"]["length_including_special_tokens"]["max"]}
        print(f"Reconciled saved input {name}: targets, morphology, masks" +
              (", fresh tokenization" if retokenize else ""), flush=True)
        del saved
    return {"status": "saved_inputs_reconciled_not_gpu_ready", "identity": marker["identity"],
            "freeze_identity": freeze["identity"], "conditions": audited,
            "retokenization_performed": retokenize, "model_evaluation_performed": False,
            "training_performed": False}
