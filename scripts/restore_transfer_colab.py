"""Restore a SHA-pinned bundle and recorded Colab packages; no model forwards."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def installed(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.archive) != args.sha256:
        raise ValueError("Full execution archive SHA differs")
    import torch
    with zipfile.ZipFile(args.archive) as archive:
        reference = json.loads(archive.read("artifacts/evaluation-preparation/robustness-grid-gpu-v1/calibration.json"))
        runtime = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0))}
        if any(runtime[k] != reference["gpu_runtime"][k] for k in runtime):
            raise ValueError(f"Base GPU runtime differs; preserve evidence, no fallback: {runtime}")
        with tempfile.TemporaryDirectory(prefix="transfer-bootstrap-") as temporary:
            helper = Path(temporary) / "package_heldout_execution.py"
            helper.write_bytes(archive.read("scripts/package_heldout_execution.py"))
            subprocess.run([sys.executable, str(helper), "restore", "--archive", str(args.archive),
                "--sha256", args.sha256, "--destination", str(args.project)], check=True)
    constraints = args.project / "requirements/colab.constraints.txt"
    pins = [line.strip() for line in constraints.read_text().splitlines() if line.strip() and not line.startswith("#")]
    expected = dict(line.split("==") for line in pins)
    before = installed(expected)
    if before != expected:
        print("RESTORING_RECORDED_PACKAGES", json.dumps({"before": before, "expected": expected}), flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-c", str(constraints), *pins], check=True)
    if installed(expected) != expected:
        raise ValueError("Recorded package pins not restored exactly")
    os.environ["PYTHONPATH"] = str(args.project / "src")
    sys.path.insert(0, str(args.project / "src"))
    importlib.invalidate_caches()
    execution = json.loads((args.project / "configs/transfer-execution.json").read_text())
    original = json.loads((args.project / "configs/held-out-test.json").read_text())
    release = json.loads((args.project / original["data_manifest"]["path"]).read_text())
    source = next(x for x in release["files"] if x["split"] == "validation")
    raw_name = release["local_directory"] + "/" + source["filename"]
    raw = args.project / raw_name
    if not raw.exists():
        old_archive = args.storage / execution["validation"]["archive"]
        if digest(old_archive) != execution["validation"]["archive_sha256"]:
            raise ValueError("Existing validation archive changed")
        with zipfile.ZipFile(old_archive) as archive:
            payload = archive.read(raw_name)
        if len(payload) != source["bytes"] or hashlib.sha256(payload).hexdigest() != source["sha256"]:
            raise ValueError("Pinned validation archive member changed")
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(payload)
    if digest(raw) != source["sha256"] or raw.stat().st_size != source["bytes"]:
        raise ValueError("Restored validation CSV differs")
    from huggingface_hub import hf_hub_download
    token = json.loads((args.project / "configs/tokenizer.json").read_text())
    downloaded = {}
    for name, expected_hash in execution["published_checkpoint_files"].items():
        path = hf_hub_download(repo_id=token["model_id"], revision=token["revision"], filename=name,
            cache_dir=str(args.project / token["cache_directory"]), token=False)
        if digest(path) != expected_hash:
            raise ValueError("Pinned published checkpoint bytes differ")
        downloaded[name] = expected_hash
    from turkish_punctuation.training_io import utc_now, write_json
    result = {"status": "restored_not_readiness", "base_runtime": runtime, "archive_sha256": args.sha256,
        "packages_before": before, "packages_after": installed(expected), "validation_raw_sha256": source["sha256"],
        "published_checkpoint_file_sha256": downloaded, "created_at_utc": utc_now(),
        "training_updates": 0, "model_forward_passes": 0, "test_scores_computed": False}
    if args.receipt.exists():
        previous = json.loads(args.receipt.read_text())
        if previous["archive_sha256"] != args.sha256 or previous["base_runtime"] != runtime:
            raise ValueError("Existing restoration receipt differs; preserve/version it")
        print("Matching restoration receipt reused", flush=True)
    else:
        write_json(args.receipt, result)
    print("TRANSFER_COLAB_RESTORED: actual readiness still required", flush=True)


if __name__ == "__main__":
    main()
