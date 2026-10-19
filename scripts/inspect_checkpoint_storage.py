"""Inspect the two-slot storage protocol with tiny payloads and zero training updates."""

import argparse
import json
from pathlib import Path

import torch

from turkish_punctuation.training_io import (
    capture_rng,
    load_checkpoint,
    save_rotating_checkpoint,
    sha256,
    utc_now,
    value_digest,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output_directory.resolve()
    if output.is_relative_to(Path("/content/drive")) and not output.is_relative_to(
        Path("/content/drive/MyDrive/cmpe586-turkish-punctuation/preparation/checkpoint-layout-v1")
    ):
        raise ValueError(
            "Drive inspections must use the project's checkpoint preparation directory."
        )
    if not 1 <= args.cycles <= 8:
        raise ValueError("Choose 1–8 tiny inspection writes.")
    identity = value_digest(
        [
            "checkpoint-storage-inspection-v1",
            str(output),
            sha256(ROOT / "src/turkish_punctuation/training_io.py"),
        ]
    )
    checkpoints = output / "checkpoints"
    report_path = output / "inspection.json"
    if args.resume:
        report = json.loads(report_path.read_text())
        if report["identity"] != identity:
            raise ValueError("Inspection source/location differs from the existing inspection.")
        loaded, protected = load_checkpoint(checkpoints, identity=identity)
        serial = loaded["serial"]
    else:
        if output.exists():
            raise FileExistsError("Use a new, empty inspection directory or --resume.")
        report = {"identity": identity, "writes": [], "created_at_utc": utc_now()}
        protected, serial = None, 0
    for _ in range(args.cycles):
        serial += 1
        payload = {
            "identity": identity,
            "serial": serial,
            "model": {"inspection_tensor": torch.arange(8, dtype=torch.float32) + serial},
            "optimizer": {},
            "scheduler": {},
            "progress": {"global_step": 0},
            "permutation": torch.arange(4),
            "rng": capture_rng(),
        }
        protected = save_rotating_checkpoint(
            checkpoints, payload, identity=identity, serial=serial, protected=protected
        )
        loaded, selected = load_checkpoint(checkpoints, identity=identity)
        if (
            selected != protected
            or loaded["serial"] != serial
            or not torch.equal(
                loaded["model"]["inspection_tensor"], payload["model"]["inspection_tensor"]
            )
        ):
            raise ValueError("Checkpoint readback differs from the tiny written payload.")
        file = protected / "state.pt"
        report["writes"].append(
            {
                "serial": serial,
                "slot": protected.name,
                "bytes": file.stat().st_size,
                "inode": file.stat().st_ino,
                "state_sha256": sha256(file),
            }
        )
    for slot in ("slot-01", "slot-02"):
        versions = [item for item in report["writes"] if item["slot"] == slot]
        if len({item["inode"] for item in versions}) > 1:
            raise ValueError("The filesystem replaced a slot's inode during an in-place overwrite.")
    report.update(
        {
            "status": "passed",
            "updated_at_utc": utc_now(),
            "latest_serial": serial,
            "slot_state_files": len(list(checkpoints.glob("slot-0[12]/state.pt"))),
            "restricted_load_and_readback_passed": True,
            "slot_inodes_preserved": True,
            "training_io_sha256": sha256(ROOT / "src/turkish_punctuation/training_io.py"),
            "training_performed": False,
            "optimizer_updates": 0,
            "real_model_weights_loaded": False,
            "permanent_deletion_performed": False,
            "cloud_durability_verified": False,
            "scope": "Tiny storage payload only; not an actual optimizer-resume or cloud durability guarantee.",
        }
    )
    write_json(report_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
