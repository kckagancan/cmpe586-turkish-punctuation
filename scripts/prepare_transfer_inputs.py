"""Prepare corpus-anchored transfer inputs; never load prediction models or score."""

import argparse
import json
import platform
import shutil
import time
import uuid
from importlib.metadata import version
from pathlib import Path

from datasets import disable_progress_bars, load_from_disk

from turkish_punctuation.data import read_split
from turkish_punctuation.heldout import load_test, verify_commit
from turkish_punctuation.noise_audit import encode_and_audit, validate_source
from turkish_punctuation.preprocessing import partition_records
from turkish_punctuation.training_io import (
    directory_lock,
    file_hashes,
    sha256,
    utc_now,
    value_digest,
    write_json,
)
from turkish_punctuation.transfer_audit import audit_transfer_changes
from turkish_punctuation.transfer_noise import (
    CONTROL,
    FAMILIES,
    apply_transfer,
    build_transfer_plans,
)
from turkish_punctuation.transfer_preparation import (
    bindings,
    load_draft,
    load_frozen,
    read,
    transfer_tokenizer,
    verify_clean_reference,
    verify_prepared,
)

ROOT = Path(__file__).resolve().parents[1]


def jsonl(path, rows):
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-only", action="store_true", help="Use validation prefix, no test views")
    parser.add_argument("--verify-only", action="store_true", help="Read and reconcile completed inputs")
    parser.add_argument("--retokenize", action="store_true", help="Also compare fresh tokenizer output")
    args = parser.parse_args()
    if args.probe_only and args.verify_only or args.retokenize and not args.verify_only:
        parser.error("Use --retokenize with --verify-only; probe and verify are separate operations")
    if args.verify_only:
        print(json.dumps(verify_prepared(ROOT, retokenize=args.retokenize), indent=2), flush=True)
        return
    config, noise, original, freeze, cohorts = (load_draft(ROOT) if args.probe_only else load_frozen(ROOT))
    original_clean_audit = verify_clean_reference(ROOT, config)
    split = "validation" if args.probe_only else "test"
    release = read(ROOT / original["data_manifest"]["path"])
    source = next(item for item in release["files"] if item["split"] == split)
    if args.probe_only:
        path = ROOT / release["local_directory"] / source["filename"]
        if sha256(path) != source["sha256"] or path.stat().st_size != source["bytes"]:
            raise ValueError("Pinned validation source changed")
        clean, quarantined = partition_records(read_split(path, nrows=config["technical_probe"]["source_prefix_rows"]))
        excluded_ids = quarantined["row_id"].tolist()
    else:
        clean, source = load_test(ROOT, ROOT, original, cohorts)
        excluded_ids = cohorts["quarantined_row_ids"]
    validate_source(clean)
    specification = {
        "mode": "validation_data_only_probe" if args.probe_only else "frozen_test_input_preparation",
        "freeze_identity": None if args.probe_only else freeze["identity"],
        "source_sha256": bindings(ROOT), "raw_source_sha256": source["sha256"],
        "source_row_ids_sha256": value_digest(clean["row_id"].tolist()),
        "cohorts_sha256": value_digest(cohorts), "tokenizer": config["tokenizer"],
        "packages": {name: version(name) for name in ("transformers", "tokenizers", "datasets", "pandas")},
        "python": platform.python_version(),
    }
    identity = value_digest(specification)
    preparation_root = ROOT / config["paths"]["preparation_root"]
    destination = (preparation_root / f"probe-{identity[:16]}" if args.probe_only
                   else ROOT / config["paths"]["prepared_directory"])
    with directory_lock(preparation_root / "input-lock"):
        if destination.exists():
            marker = verify_commit(destination)
            manifest = read(destination / "manifest.json")
            if marker["identity"] != identity or manifest["specification"] != specification:
                raise ValueError("Completed destination belongs to different sources/settings; preserve it")
            print(f"Matching completed inputs verified, not regenerated: {destination}", flush=True)
            return
        staging = preparation_root / f"attempt-{split}-{uuid.uuid4().hex}"
        staging.mkdir(parents=True)
        started = time.monotonic()
        manifest = {
            "version": "transfer-inputs-v1", "specification": specification,
            "identity": identity, "freeze_identity": specification["freeze_identity"], "split": split,
            "started_at_utc": utc_now(), "raw_source": source,
            "eligible_row_ids": clean["row_id"].tolist(), "excluded_source_row_ids": excluded_ids,
            "source_prefix_rows": config["technical_probe"]["source_prefix_rows"] if args.probe_only else None,
            "model_weights_loaded": False, "model_evaluation_performed": False,
            "training_performed": False, "gpu_readiness_checked": False, "conditions": {},
        }
        try:
            tokenizer = transfer_tokenizer(ROOT, config)
            disable_progress_bars()
            for name in specification["source_sha256"]:
                target = staging / "source_snapshot" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, target)
            sensitivity_ids = set(cohorts["sensitivity_row_ids"]) if split == "test" else set(clean["row_id"])
            sensitive_clean = clean[clean["row_id"].isin(sensitivity_ids)]

            def audit_view(noisy, edits, family=None, plan=None):
                changes = audit_transfer_changes(clean, noisy, edits, noise, family, plan)
                sensitivity = audit_transfer_changes(sensitive_clean, noisy[noisy["row_id"].isin(sensitivity_ids)],
                    [e for e in edits if e["row_id"] in sensitivity_ids], noise, family)
                encoded, token_audit = encode_and_audit(noisy, tokenizer, max_length=512, batch_size=256)
                return encoded, {"family": family, "changes": changes,
                                 "sensitivity_changes": sensitivity, "tokenization": token_audit}

            encoded, audit = audit_view(clean, [])
            if split == "test":
                saved = load_from_disk(str(ROOT / config["clean_encoding_reference"]["directory"]))
                if (set(encoded.column_names) != set(saved.column_names)
                        or any(encoded[n][:] != saved[n][:] for n in encoded.column_names)
                        or json.loads(json.dumps(audit["tokenization"])) != original_clean_audit):
                    raise ValueError("Fresh clean tokenization/targets differ from the audited read-only reference")
                audit["encoding_reference"] = config["clean_encoding_reference"]
                del saved
            manifest["conditions"]["clean-anchor"] = audit
            del encoded
            print(f"Audited {split} clean anchor: rows={len(clean)}, no duplicate encoding saved", flush=True)
            for seed in config["noise"]["seeds"]:
                plans = build_transfer_plans(clean, noise, split=split, seed=seed)
                write_json(staging / "plans" / f"seed{seed}.json", plans)
                for family in (*FAMILIES, CONTROL):
                    name = f"{family}-corpus-seed{seed}"
                    noisy, edits = apply_transfer(clean, noise, plans, family)
                    encoded, audit = audit_view(noisy, edits, family, plans)
                    audit["seed"] = seed
                    output = staging / "conditions" / name
                    output.mkdir(parents=True)
                    if not args.probe_only:
                        jsonl(output / "records.jsonl", noisy.to_dict(orient="records"))
                        jsonl(output / "edits.jsonl", edits)
                        encoded.save_to_disk(str(output / "dataset"))
                    write_json(output / "audit.json", audit)
                    manifest["conditions"][name] = audit
                    if family == CONTROL:
                        partial = manifest["conditions"][f"partial_folding-corpus-seed{seed}"]["changes"]
                        full = audit["changes"]
                        if (partial["selected_positions_sha256"] != full["selected_positions_sha256"]
                                or partial["changed_words"] != full["changed_words"]
                                or partial["declared_character_edit_count"] >= full["declared_character_edit_count"]):
                            raise ValueError("Partial/full control did not preserve word dose and proper-subset character dose")
                    print(f"Audited {split}/{name}: changed={audit['changes']['changed_words']}; "
                          f"max_subwords={audit['tokenization']['length_including_special_tokens']['max']}", flush=True)
                    del encoded, noisy, edits
            if list(manifest["conditions"]) != config["run_matrix"]["conditions"]:
                raise ValueError("Prepared condition order/membership differs from fixed matrix")
            if bindings(ROOT) != specification["source_sha256"]:
                raise ValueError("Source/configuration changed during preparation")
            if not args.probe_only:
                load_frozen(ROOT)
            manifest["elapsed_seconds"] = time.monotonic() - started
            manifest["completed_at_utc"] = utc_now()
            manifest["max_input_length"] = max(c["tokenization"]["length_including_special_tokens"]["max"]
                                                for c in manifest["conditions"].values())
            write_json(staging / "manifest.json", manifest)
            write_json(staging / "complete.json", {"identity": identity, "files": file_hashes(staging)})
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError("Destination appeared during preparation; preserve both attempts")
            staging.rename(destination)
        except Exception as error:
            write_json(staging / "failure.json", {"identity": identity, "specification": specification,
                "failed_at_utc": utc_now(), "error": repr(error), "completed_condition_names": list(manifest["conditions"]),
                "model_evaluation_performed": False, "training_performed": False})
            raise
    print(f"TRANSFER_INPUTS_PREPARED: {split}, 13 views, no weights/scores; {destination}", flush=True)


if __name__ == "__main__":
    main()
