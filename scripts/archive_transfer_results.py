"""Archive the completed transfer run without model loading or repeated inference."""

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

RUN = "artifacts/evaluations/transfer-noise-v1"
READY = "preparation/transfer-noise-v1/readiness-v1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def tree(path):
    files = {}
    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"Symlink in evidence: {item}")
        if item.is_file():
            files[item.relative_to(path).as_posix()] = sha(item)
    return files


def committed(path):
    marker = read(path / "complete.json")
    files = tree(path)
    del files["complete.json"]
    if marker["files"] != files:
        raise ValueError(f"Committed evidence changed: {path}")
    return marker


def inspect(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        manifest = json.loads(archive.read("bundle-manifest.json"))
        if set(names) != set(manifest["files"]) | {"bundle-manifest.json"}:
            raise ValueError("Archive inventory differs")
        for name, expected in manifest["files"].items():
            member = archive.getinfo(name)
            relative = PurePosixPath(name)
            if (relative.is_absolute() or ".." in relative.parts or "\\" in name
                    or relative.as_posix() != name or member.is_dir()
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError(f"Unsafe member: {name}")
            with archive.open(name) as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != expected:
                raise ValueError(f"Archive readback differs: {name}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory = args.storage_root / RUN
    run = read(directory / "provenance/run.json")
    identity = run["identity"]
    progress = read(directory / "progress.json")
    if (identity != digest(run["specification"])
            or committed(directory / "provenance")["identity"] != identity
            or progress["identity"] != identity or progress["status"] != "inference_complete"
            or progress["completed_conditions"] != 52 or progress["planned_conditions"] != 52):
        raise ValueError("Completed 52-condition run is required")
    binding = run["specification"]["runtime_binding"]
    for name, expected in binding["source_sha256"].items():
        if sha(directory / "provenance/source_snapshot" / name) != expected:
            raise ValueError(f"Execution source differs: {name}")
    execution = read(directory / "provenance/source_snapshot/configs/transfer-execution.json")
    expected_conditions = set()
    for arm in execution["model_order"]:
        for name in execution["conditions"]:
            condition = directory / "models" / arm / "conditions" / name
            marker = committed(condition)
            if (marker["identity"] != identity or set(marker["files"]) != {
                    "predictions.jsonl", "metrics.json", "error_examples.json"}):
                raise ValueError(f"Condition differs: {arm}/{name}")
            expected_conditions.add(f"{arm}/conditions/{name}")
    actual = {p.parent.relative_to(directory / "models").as_posix()
              for p in (directory / "models").glob("*/conditions/*/complete.json")}
    if actual != expected_conditions or len(actual) != 52:
        raise ValueError("Condition matrix differs")
    ready = args.storage_root / READY
    readiness = read(ready / "readiness.json")
    if (committed(ready)["identity"] != readiness["identity"]
            or readiness["identity"] != digest(binding)
            or readiness["specification"] != binding
            or sha(ready / "complete.json") != run["specification"]["readiness_commit_sha256"]):
        raise ValueError("Original readiness binding differs")
    files = {}
    for base in (directory, ready):
        for name in tree(base):
            path = base / name
            files[path.relative_to(args.storage_root).as_posix()] = path
    manifest = {"version": "held-out-transfer-v1", "kind": "transfer-completed-raw-results",
                "freeze_identity": execution["freeze_identity"], "run_identity": identity,
                "files": {n: sha(p) for n, p in sorted(files.items())}}
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / "transfer-noise-v1-raw-results.zip"
    if not target.exists():
        with TemporaryDirectory(prefix=".transfer-archive-", dir=args.output) as temporary:
            staged = Path(temporary) / target.name
            with zipfile.ZipFile(staged, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for name, path in sorted(files.items()):
                    archive.write(path, name)
                archive.writestr("bundle-manifest.json", json.dumps(manifest, indent=2) + "\n")
            if inspect(staged) != manifest:
                raise ValueError("New archive readback differs")
            staged.rename(target)
    if inspect(target) != manifest:
        raise ValueError("Existing archive differs; preserve and version it")
    receipt = {"status": "raw_result_archive_verified", "run_identity": identity,
               "filename": target.name, "sha256": sha(target), "bytes": target.stat().st_size,
               "files": len(files), "forward_conditions": 52, "cohort_score_sets": 104,
               "analysis_pending": True, "inference_repeated": False, "weights_included": False,
               "drive_deleted": False, "archive_source_sha256": sha(Path(__file__))}
    receipt_path = args.output / "raw-archive.json"
    if receipt_path.exists() and read(receipt_path) != receipt:
        raise ValueError("Existing archive receipt differs")
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print("TRANSFER_RAW_ARCHIVE_COMPLETE " + json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
