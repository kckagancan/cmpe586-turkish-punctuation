"""Pack already audited validation inputs; verify safely before extracting on Colab."""

import argparse
import json
import stat
import zipfile
from pathlib import Path, PurePosixPath

from turkish_punctuation.training_io import sha256

ROOT = Path(__file__).resolve().parents[1]


def verify(path: Path, expected: str, destination: Path | None) -> dict:
    if sha256(path) != expected:
        raise ValueError("Transfer ZIP SHA-256 differs.")
    if destination is not None and destination.exists():
        raise FileExistsError("Choose a new extraction directory.")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("transfer-manifest.json"))
        entries = archive.infolist()
        if len(entries) != len({entry.filename for entry in entries}):
            raise ValueError("Duplicate ZIP paths.")
        if set(archive.namelist()) != set(manifest["files"]) | {"transfer-manifest.json"}:
            raise ValueError("ZIP member set differs from its manifest.")
        for entry in entries:
            name = PurePosixPath(entry.filename)
            if name.is_absolute() or ".." in name.parts or "\\" in entry.filename:
                raise ValueError("Unsafe ZIP path.")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise ValueError("ZIP symlinks are not allowed.")
            content = archive.read(entry)
            if entry.filename != "transfer-manifest.json":
                import hashlib
                item = manifest["files"][entry.filename]
                if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
                    raise ValueError(f"ZIP member differs: {entry.filename}")
            if destination is not None:
                target = destination / entry.filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
                if entry.filename != "transfer-manifest.json" and sha256(target) != item["sha256"]:
                    raise OSError(f"Extracted member differs: {entry.filename}")
    return {"status": "verified", "sha256": expected, "files": len(manifest["files"]),
            "test_text_included": False, "noise_regenerated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()
    if args.verify:
        if not args.sha256:
            parser.error("--verify requires --sha256")
        print(json.dumps(verify(args.verify, args.sha256, args.extract_to), indent=2))
        return
    if args.output is None:
        parser.error("Choose --output or --verify.")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    noise = ROOT / "data/processed/noise-v1"
    prepared = json.loads((noise / "manifest.json").read_text())
    for name, audit in prepared["conditions"].items():
        for file, digest in audit["output_sha256"].items():
            if sha256(noise / "conditions" / name / file) != digest:
                raise ValueError(f"Existing noise input changed: {name}/{file}")
    source = json.loads((ROOT / "docs/data-manifest.json").read_text())
    validation = next(item for item in source["files"] if item["split"] == "validation")
    raw = ROOT / source["local_directory"] / validation["filename"]
    if sha256(raw) != validation["sha256"]:
        raise ValueError("Raw validation differs.")
    reference = ROOT / "artifacts/evaluations/published-bert-clean-validation-v1"
    files = [path for path in noise.rglob("*") if path.is_file()]
    files += [raw, reference / "metrics.json", reference / "predictions.jsonl"]
    manifest = {"version": "robustness-validation-transfer-v1", "test_text_included": False,
                "noise_regenerated": False, "files": {
                    str(path.relative_to(ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
                    for path in sorted(files)}}
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files):
            archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr("transfer-manifest.json", json.dumps(manifest, indent=2))
    digest = sha256(output)
    result = verify(output, digest, None)
    result["bytes"] = output.stat().st_size
    result["archive"] = str(output)
    output.with_suffix(".receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
