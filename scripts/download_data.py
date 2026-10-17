"""Download the pinned upstream CSV files and verify their recorded SHA-256 hashes."""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / "docs/data-manifest.json").read_text())
    directory = ROOT / manifest["local_directory"]
    directory.mkdir(parents=True, exist_ok=True)
    print("Release license remains unresolved; see docs/data-audit.md.")
    for item in manifest["files"]:
        target = directory / item["filename"]
        if target.exists():
            if target.stat().st_size != item["bytes"] or sha256(target) != item["sha256"]:
                raise ValueError(f"Existing file differs from pinned release: {target}")
            print(f"Verified existing file: {item['filename']}")
            continue
        temporary = target.with_suffix(".csv.part")
        try:
            with urlopen(item["source_url"], timeout=60) as response, temporary.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            if temporary.stat().st_size != item["bytes"]:
                raise ValueError(f"Unexpected file size: {item['filename']}")
            if sha256(temporary) != item["sha256"]:
                raise ValueError(f"Checksum mismatch: {item['filename']}")
            temporary.rename(target)
            print(f"Downloaded and verified: {item['filename']}")
        finally:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
