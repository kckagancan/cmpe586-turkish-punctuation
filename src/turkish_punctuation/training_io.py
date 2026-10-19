"""Atomic run artifacts and restricted-loader checkpoints for local/Colab training."""

import hashlib
import json
import os
import pickle
import random
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir

import numpy as np
import torch


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def value_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hashes(directory: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, str]:
    return {
        str(path.relative_to(directory)): sha256(path)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and str(path.relative_to(directory)) not in exclude
    }


def write_json(path: Path, value: object) -> None:
    """Publish complete JSON without exposing a partially written destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def directory_lock(directory: Path):
    """Coordinate processes within one host, even when artifacts live on Drive FUSE."""
    import fcntl

    directory.mkdir(parents=True, exist_ok=True)
    lock_directory = Path(gettempdir()) / "cmpe586-training-locks"
    lock_directory.mkdir(exist_ok=True)
    lock_path = lock_directory / f"{value_digest(str(directory.resolve()))}.lock"
    with lock_path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process holds the run lock: {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng() -> dict:
    """Use tensors/primitives so torch.load(weights_only=True) can read all state."""
    numpy_state = np.random.get_state(legacy=True)
    if not isinstance(numpy_state, tuple):
        raise TypeError("Expected the legacy NumPy MT19937 state tuple.")
    return {
        "python": random.getstate(),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.tensor(numpy_state[1].astype(np.int64)),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            numpy_state["keys"].numpy().astype(np.uint32),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count differs from the saved run.")
        torch.cuda.set_rng_state_all(state["cuda"])


def model_digest(model: torch.nn.Module) -> str:
    """Hash actual parameter/buffer tensors, independently of serialization metadata."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_checkpoint(
    directory: Path, payload: dict, *, identity: str, serial: int, keep: int
) -> Path:
    """Commit only complete checkpoints; retain the previous checkpoint during writes."""
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"checkpoint-{serial:06d}"
    if destination.exists():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    with TemporaryDirectory(prefix=".checkpoint-", dir=directory) as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        torch.save(payload, staging / "state.pt")
        write_json(
            staging / "manifest.json",
            {
                "identity": identity,
                "serial": serial,
                "state_sha256": sha256(staging / "state.pt"),
                "created_at_utc": utc_now(),
            },
        )
        staging.rename(destination)
    committed = sorted(directory.glob("checkpoint-[0-9]*"))
    for old in committed[:-keep]:
        shutil.rmtree(old)
    return destination


def load_checkpoint(directory: Path, *, identity: str) -> tuple[dict, Path]:
    """Try newest committed state first; report any damaged checkpoint explicitly."""
    errors = []
    candidates = [*directory.glob("checkpoint-[0-9]*"), *directory.glob("slot-0[12]")]

    def serial(path: Path) -> int:
        try:
            return int(json.loads((path / "manifest.json").read_text())["serial"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{path.name}: unreadable manifest: {error}")
            return -1

    for checkpoint in sorted(candidates, key=serial, reverse=True):
        try:
            manifest = json.loads((checkpoint / "manifest.json").read_text())
            if manifest.get("status", "complete") != "complete":
                raise ValueError("Checkpoint slot has an incomplete write.")
            if manifest["identity"] != identity:
                raise ValueError("Checkpoint belongs to a different run identity.")
            if sha256(checkpoint / "state.pt") != manifest["state_sha256"]:
                raise ValueError("Checkpoint hash differs.")
            payload = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
            if payload["identity"] != identity or payload["serial"] != manifest["serial"]:
                raise ValueError("Checkpoint payload identity/serial differs.")
            if errors:
                print(
                    f"Resuming from {checkpoint.name}; skipped damaged states: {errors}", flush=True
                )
            return payload, checkpoint
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            EOFError,
            pickle.UnpicklingError,
        ) as error:
            errors.append(f"{checkpoint.name}: {error}")
    raise ValueError(f"No verified resumable checkpoint found: {errors}")


def _rewrite_file(path: Path, content: bytes) -> None:
    """Preserve a Drive file's pathname; do not unlink or rename an old version."""
    if path.is_symlink():
        raise ValueError("Checkpoint slots must not contain symlinks.")
    with path.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def save_rotating_checkpoint(
    directory: Path,
    payload: dict,
    *,
    identity: str,
    serial: int,
    protected: Path | None,
) -> Path:
    """Reuse two stable slots while retaining the last verified slot during a write.

    Stage on the VM first. Invalidate the other slot's manifest, overwrite its
    state in place, check the readback hash, and publish its complete manifest
    last. No Drive directory/file is deleted. Cloud upload durability and the
    provider's version history remain separate from this local protocol.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or any(directory.glob("checkpoint-[0-9]*")):
        raise ValueError("Use a fresh two-slot directory, not legacy checkpoints or symlinks.")
    slots = [directory / "slot-01", directory / "slot-02"]
    if protected is not None and protected not in slots:
        raise ValueError("Protected checkpoint is outside the declared two slots.")
    if protected is None and any(path.exists() for path in slots):
        raise ValueError("Existing slots require a verified resume checkpoint before rewriting.")
    destination = slots[1] if protected == slots[0] else slots[0]
    if destination.is_symlink():
        raise ValueError("Checkpoint slot must not be a symlink.")
    if payload["identity"] != identity or payload["serial"] != serial:
        raise ValueError("Checkpoint payload identity/serial differs.")
    destination.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="cmpe586-checkpoint-") as temporary:
        staging = Path(temporary) / "state.pt"
        torch.save(payload, staging)
        expected = sha256(staging)
        writing = {"identity": identity, "serial": serial, "status": "writing"}
        _rewrite_file(destination / "manifest.json", json.dumps(writing).encode())
        state_path = destination / "state.pt"
        if state_path.is_symlink():
            raise ValueError("Checkpoint state must not be a symlink.")
        with staging.open("rb") as source, state_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if sha256(state_path) != expected:
            raise ValueError("Checkpoint slot readback hash differs; previous slot retained.")
        complete = {
            **writing,
            "status": "complete",
            "state_sha256": expected,
            "created_at_utc": utc_now(),
        }
        _rewrite_file(destination / "manifest.json", json.dumps(complete).encode())
    return destination
