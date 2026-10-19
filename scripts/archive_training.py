"""Archive a completed run's models/results; never delete source or Drive files."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def files(directory: Path, *, exclude: tuple[str, ...] = ()) -> dict[str, Path]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Archive sources must not contain symlinks: {path}")
        relative = path.relative_to(directory)
        if relative.parts[0] in exclude or not path.is_file():
            continue
        result[relative.as_posix()] = path
    return result


def validate_run(run_root: Path, initialization: Path) -> tuple[dict, dict]:
    run = read_json(run_root / "run.json")
    summary = read_json(run_root / "summary.json")
    progress = read_json(run_root / "progress.json")
    specification = run["specification"]
    identity = hashlib.sha256(
        json.dumps(specification, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    if run["identity"] != identity or summary["identity"] != identity:
        raise ValueError("Run identity differs from its specification/summary.")
    epochs = specification["configuration"]["optimization"]["epochs"]
    if (
        progress.get("status") != "complete"
        or progress.get("global_step") != summary["global_step"]
        or len(summary["history"]) != epochs
        or [item["epoch"] for item in summary["history"]] != list(range(1, epochs + 1))
        or summary["global_step"] != run["budget"]["total_updates"]
        or summary["best_model"] != f"models/epoch-{summary['best_epoch']:02d}"
    ):
        raise ValueError("The run has not completed its declared training/selection.")
    selected = min(
        summary["history"],
        key=lambda item: (-item["validation"]["metrics"]["punctuation_macro"]["f1"], item["epoch"]),
    )
    if selected["epoch"] != summary["best_epoch"]:
        raise ValueError("Best epoch does not match the declared validation selection rule.")
    for epoch in summary["history"]:
        number = epoch["epoch"]
        model = run_root / "models" / f"epoch-{number:02d}"
        export = read_json(model / "export.json")
        actual = {name: sha256(path) for name, path in files(model).items() if name != "export.json"}
        if export["identity"] != identity or export != epoch["model_export"] or actual != export["files"]:
            raise ValueError(f"Epoch {number} model export differs from its manifest.")
        validation = run_root / "validation" / f"epoch-{number:02d}"
        if (
            sha256(validation / "metrics.json") != epoch["validation_report_sha256"]
            or sha256(validation / "predictions.jsonl") != epoch["validation"]["predictions_sha256"]
        ):
            raise ValueError(f"Epoch {number} validation outputs differ from their hashes.")
    initial = read_json(initialization / "initialization.json")
    actual_initial = {
        name: sha256(path) for name, path in files(initialization).items()
        if name != "initialization.json"
    }
    if (
        initial["identity"] != specification["initialization_identity"]
        or initial["model_tensors_sha256"] != specification["initialization_tensors_sha256"]
        or actual_initial != initial["files"]
    ):
        raise ValueError("Shared initialization differs from the recorded run.")
    return run, summary


def create_archive(run_root: Path, initialization: Path, output: Path) -> dict:
    run, summary = validate_run(run_root, initialization)
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}")
    members = {f"run/{name}": path for name, path in files(run_root, exclude=("checkpoints",)).items()}
    members.update({f"initialization/{name}": path for name, path in files(initialization).items()})
    manifest = {
        "schema_version": 1,
        "run_identity": run["identity"],
        "completed_epochs": len(summary["history"]),
        "optimizer_updates": summary["global_step"],
        "best_model": f"run/{summary['best_model']}",
        "source_run_directory": str(run_root),
        "source_initialization_directory": str(initialization),
        "checkpoint_states_included": False,
        "checkpoint_policy": "Completed run: models/results retained; optimizer resume state omitted.",
        "files": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".archive-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for name, source in members.items():
                digest, size = hashlib.sha256(), 0
                with source.open("rb") as reader, archive.open(name, "w", force_zip64=True) as writer:
                    for chunk in iter(lambda: reader.read(4 * 1024 * 1024), b""):
                        writer.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                manifest["files"][name] = {"sha256": digest.hexdigest(), "bytes": size}
            archive.writestr("archive-manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        staging.rename(output)
    return {
        "archive": str(output), "bytes": output.stat().st_size, "sha256": sha256(output),
        "members": len(members), "run_identity": run["identity"],
        "checkpoint_states_included": False,
    }


def verify_archive(archive_path: Path, expected_sha256: str, extract_to: Path | None) -> dict:
    archive_hash = sha256(archive_path)
    if archive_hash != expected_sha256:
        raise ValueError("Downloaded archive SHA-256 differs from its transfer receipt.")
    if extract_to is not None and extract_to.exists():
        raise FileExistsError(f"Extraction destination already exists: {extract_to}")
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("archive-manifest.json"))
        expected = {*manifest["files"], "archive-manifest.json"}
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("Archive entries differ from the manifest or contain duplicates.")
        for name in names:
            relative = PurePosixPath(name)
            info = archive.getinfo(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Unsafe archive member: {name}")
        for name, expected_file in manifest["files"].items():
            with archive.open(name) as stream:
                actual_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual_hash != expected_file["sha256"] or archive.getinfo(name).file_size != expected_file["bytes"]:
                raise ValueError(f"Archive member failed hash/size verification: {name}")
        if extract_to is not None:
            extract_to.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(prefix=".extract-", dir=extract_to.parent) as temporary:
                staging = Path(temporary) / "complete"
                staging.mkdir()
                archive.extractall(staging)
                for name, expected_file in manifest["files"].items():
                    if sha256(staging / name) != expected_file["sha256"]:
                        raise ValueError(f"Extracted file failed verification: {name}")
                run, summary = validate_run(staging / "run", staging / "initialization")
                if (
                    run["identity"] != manifest["run_identity"]
                    or f"run/{summary['best_model']}" != manifest["best_model"]
                ):
                    raise ValueError("Extracted run differs from the archive identity/selection.")
                shutil.move(str(staging), str(extract_to))
    return {
        "status": "verified", "archive": str(archive_path), "sha256": archive_hash,
        "run_identity": manifest["run_identity"], "members": len(manifest["files"]),
        "checkpoint_states_included": manifest["checkpoint_states_included"],
        "extracted_to": str(extract_to) if extract_to is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--initialization-directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        if not args.expected_sha256:
            parser.error("--verify requires --expected-sha256 from the source receipt")
        result = verify_archive(args.verify, args.expected_sha256, args.extract_to)
    else:
        if any(value is None for value in (args.run_directory, args.initialization_directory, args.output)):
            parser.error("Creation requires --run-directory, --initialization-directory and --output")
        result = create_archive(args.run_directory, args.initialization_directory, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
