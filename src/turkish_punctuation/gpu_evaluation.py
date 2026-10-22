"""Explicit GPU numerics and verified condition publication for the validation grid."""

import json
import os
import platform
import shutil
from pathlib import Path

import torch

from turkish_punctuation.training_io import sha256


def configure_runtime(config: dict) -> dict | None:
    """Keep legacy CPU pilots strict; GPU runs must request a fresh clean anchor."""
    if config.get("runtime_policy") != "fresh_gpu_anchor":
        if config["device"] != "cpu":
            raise ValueError("A GPU run requires the explicit fresh_gpu_anchor policy.")
        return None
    if config["device"] != "cuda" or not config["recompute_clean_anchor"]:
        raise ValueError("GPU conditions require their own recomputed GPU clean control.")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; do not silently substitute a CPU run.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "precision": "float32", "tf32": False, "sdpa_backend": "math",
        "attention_implementation": "sdpa", "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def write_in_place(path: Path, value: object) -> None:
    """Overwrite small status files without Drive rename/Trash rotation.

    Progress is advisory; verified condition commits are the resume authority.
    """
    with path.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def publish_condition(staging: Path, destination: Path, identity: str) -> None:
    """Copy staged files, verify Drive readback, then write the commit marker last."""
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("predictions.jsonl", "error_examples.json", "metrics.json"):
        source, target = staging / name, destination / name
        # copyfile overwrites an interrupted, uncommitted file in place.
        shutil.copyfile(source, target)
        if sha256(source) != sha256(target):
            raise OSError(f"Published file readback differs: {target}")
        hashes[name] = sha256(target)
    write_in_place(destination / "complete.json", {"identity": identity, "files": hashes})
    if not condition_committed(destination, identity):
        raise OSError("Condition commit could not be read back.")


def condition_committed(directory: Path, identity: str) -> bool:
    """Incomplete writes are retryable; a committed but modified result is an error."""
    marker = directory / "complete.json"
    if not marker.exists():
        return False
    try:
        commit = json.loads(marker.read_text())
    except json.JSONDecodeError:
        return False
    if commit["identity"] != identity or set(commit["files"]) != {
        "predictions.jsonl", "error_examples.json", "metrics.json"
    }:
        raise ValueError(f"Condition commit identity changed: {directory}")
    for name, digest in commit["files"].items():
        if not (directory / name).is_file() or sha256(directory / name) != digest:
            raise ValueError(f"Committed output changed: {directory / name}")
    return True
