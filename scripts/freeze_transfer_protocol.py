"""Freeze the supplemental transfer recipe after a data-only validation probe."""

import shutil
import uuid
from pathlib import Path

from turkish_punctuation.heldout import verify_commit
from turkish_punctuation.training_io import (
    directory_lock,
    file_hashes,
    sha256,
    utc_now,
    value_digest,
    write_json,
)
from turkish_punctuation.transfer_preparation import (
    bindings,
    load_draft,
    read,
    verify_clean_reference,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    config, noise, original, old_freeze, cohorts = load_draft(ROOT)
    verify_clean_reference(ROOT, config)
    source_hashes = bindings(ROOT)
    for name, expected in noise.profile["source_bindings"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Compiled source morphology binding changed: {name}")
    models = {}
    for arm, model in original["models"].items():
        for field in ("reference_config", "reference_summary"):
            if sha256(ROOT / model[field]) != model[field + "_sha256"]:
                raise ValueError(f"Original model reference changed: {arm}/{field}")
        metadata_hashes = {}
        if model["kind"] == "fine_tuned_export":
            run = ROOT / model["local_run_directory"]
            exported = ROOT / model["local_model_directory"]
            for path, key in [(run / "run.json", "run_sha256"), (run / "summary.json", "summary_sha256"),
                              (exported / "export.json", "export_sha256")]:
                if sha256(path) != model[key]:
                    raise ValueError(f"Original selected checkpoint metadata changed: {arm}/{key}")
                metadata_hashes[str(path.relative_to(ROOT))] = model[key]
            if any(not (exported / name).is_file() for name in model["export_file_sha256"]):
                raise FileNotFoundError(f"Retained model export incomplete: {arm}")
        models[arm] = {"metadata_verified": True, "weights_rehashed": False,
                       "model_tensors_sha256": model["model_tensors_sha256"], "metadata_sha256": metadata_hashes}
    preparation_root = ROOT / config["paths"]["preparation_root"]
    matches = []
    for path in sorted(preparation_root.glob("probe-*")):
        marker = verify_commit(path)
        probe = read(path / "manifest.json")
        spec = probe["specification"]
        if (spec["source_sha256"] == source_hashes and spec["mode"] == "validation_data_only_probe"
                and spec["tokenizer"] == config["tokenizer"] and probe["split"] == "validation"
                and probe["source_prefix_rows"] == config["technical_probe"]["source_prefix_rows"]
                and marker["identity"] == probe["identity"] == value_digest(spec)
                and list(probe["conditions"]) == config["run_matrix"]["conditions"]
                and not any(probe[k] for k in ["model_weights_loaded", "model_evaluation_performed", "training_performed"])):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"Expected one matching completed validation data-only probe, found {len(matches)}")
    probe = matches[0]
    specification = {"source_sha256": source_hashes, "original_freeze_identity": old_freeze["identity"],
        "cohorts_sha256": value_digest(cohorts), "models": models,
        "probe": {"path": str(probe.relative_to(ROOT)), "complete_sha256": sha256(probe / "complete.json"),
                  "identity": read(probe / "manifest.json")["identity"]},
        "clean_encoding_reference": config["clean_encoding_reference"], "tokenizer": config["tokenizer"]}
    identity = value_digest(specification)
    destination = ROOT / config["paths"]["freeze_directory"]
    with directory_lock(preparation_root / "protocol-lock"):
        if destination.exists():
            marker = verify_commit(destination)
            if marker["identity"] != identity or read(destination / "freeze.json")["specification"] != specification:
                raise ValueError("Existing freeze differs; preserve it and version changes")
            print(f"Matching protocol freeze verified: {identity}", flush=True)
            return
        if any((ROOT / config["paths"][key]).exists() for key in ("prepared_directory", "output_directory")):
            raise FileExistsError("Input/scoring namespace exists; freezing must precede formal test-view generation")
        staging = preparation_root / f"freeze-attempt-{uuid.uuid4().hex}"
        staging.mkdir(parents=True)
        for name, expected in source_hashes.items():
            target = staging / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
            if sha256(target) != expected:
                raise ValueError(f"Snapshot readback changed: {name}")
        write_json(staging / "cohorts.json", cohorts)
        write_json(staging / "freeze.json", {"identity": identity, "specification": specification,
            "status": "protocol_frozen_inputs_pending_not_runtime_ready", "created_at_utc": utc_now(),
            "new_test_views_generated": False, "models_loaded": False, "weights_rehashed": False,
            "test_scored": False, "training_performed": False})
        if source_hashes != bindings(ROOT):
            raise ValueError("Source changed while freezing")
        write_json(staging / "complete.json", {"identity": identity, "files": file_hashes(staging)})
        staging.rename(destination)
    print(f"TRANSFER_PROTOCOL_FROZEN: {identity}; not GPU/model ready", flush=True)


if __name__ == "__main__":
    main()
