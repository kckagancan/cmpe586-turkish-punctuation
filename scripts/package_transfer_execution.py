"""Package audited supplemental inputs and exact source; exclude model weights."""

import argparse
from pathlib import Path

from package_heldout_execution import build, inspect_bundle, restore

from turkish_punctuation.heldout import load_plan, verify_commit
from turkish_punctuation.training_io import sha256, write_json
from turkish_punctuation.transfer_execution import load_execution, read, source_hashes

ROOT = Path(__file__).resolve().parents[1]


def pack(output):
    config, execution, original, freeze, _, _, _, _ = load_execution(ROOT, ROOT)
    _, _, old_freeze, _ = load_plan(ROOT)
    files = {name: ROOT / name for name in source_hashes(ROOT, old_freeze)}
    for name in ("scripts/package_heldout_execution.py", "src/turkish_punctuation/__init__.py",
                 "requirements/colab.constraints.txt"):
        files[name] = ROOT / name
    for name in config["tokenizer"]["files"]:
        relative = config["tokenizer"]["local_snapshot"] + "/" + name
        files[relative] = ROOT / relative
    for relative in (config["paths"]["prepared_directory"], config["paths"]["freeze_directory"],
                     original["paths"]["freeze_directory"], config["clean_encoding_reference"]["directory"]):
        for path in (ROOT / relative).rglob("*"):
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = path
    reference = config["clean_encoding_reference"]["source_manifest"]
    files[reference] = ROOT / reference
    release = read(ROOT / original["data_manifest"]["path"])
    source = next(x for x in release["files"] if x["split"] == "test")
    name = release["local_directory"] + "/" + source["filename"]
    files[name] = ROOT / name
    name = original["inference"]["reference_runtime_receipt"]
    files[name] = ROOT / name
    if any(path.suffix in {".bin", ".safetensors", ".pt"} for path in files.values()):
        raise ValueError("Model tensors must not enter the transfer package")
    output.mkdir(parents=True, exist_ok=True)
    bundle = build(output / "transfer-execution-v1.zip", "supplemental-transfer-execution", files, freeze["identity"])
    verify_commit(ROOT / config["paths"]["prepared_directory"])
    if inspect_bundle(output / bundle["filename"])["freeze_identity"] != execution["freeze_identity"]:
        raise ValueError("Bundle freeze identity differs")
    write_json(output / "transfer.json", {"bundle": bundle, "freeze_identity": freeze["identity"],
        "input_identity": execution["input_identity"], "contains_weights_or_predictions": False,
        "validation_archive_reused_not_duplicated": execution["validation"],
        "restorer_sha256": sha256(ROOT / "scripts/restore_transfer_colab.py")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    packing = sub.add_parser("pack")
    packing.add_argument("--output", type=Path, required=True)
    restoring = sub.add_parser("restore")
    restoring.add_argument("--archive", type=Path, required=True)
    restoring.add_argument("--sha256", required=True)
    restoring.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pack":
        pack(args.output)
    else:
        restore(args.archive, args.sha256, args.destination)


if __name__ == "__main__":
    main()
