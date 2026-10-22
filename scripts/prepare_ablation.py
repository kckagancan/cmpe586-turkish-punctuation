"""Audit, transfer and check readiness of existing case-loss ablation views."""

import argparse
import json
import tempfile
import zipfile
from pathlib import Path

from package_evaluation_inputs import verify
from prepare_training import copy_verified_tree

from turkish_punctuation.ablation import ROOT, audit_view_pair, load_plan, verify_ready
from turkish_punctuation.training_io import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "package", "restore", "ready"))
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan, _ = load_plan()
    views, reference = ROOT / plan["local_views"], ROOT / plan["local_reference_views"]
    if args.storage_root:
        views = args.storage_root / plan["storage_views"]
        reference = args.storage_root / plan["storage_reference_views"]
    if args.action == "audit":
        result = audit_view_pair(views, reference)
    elif args.action == "package":
        if args.output is None or args.output.exists():
            parser.error("package requires a new --output ZIP path")
        audit = audit_view_pair(views, reference)
        files = {"views/" + path.relative_to(views).as_posix(): path
                 for path in sorted(views.rglob("*")) if path.is_file()}
        if any(path.is_symlink() for path in views.rglob("*")):
            raise ValueError("View symlinks cannot be transferred.")
        manifest = {"version": plan["version"], "prepared_identity": audit["prepared_identity"],
                    "test_text_included": False, "noise_regenerated": False,
                    "files": {name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                              for name, path in files.items()}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "x", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, path in files.items():
                archive.write(path, name)
            archive.writestr("transfer-manifest.json", json.dumps(manifest, indent=2))
        result = verify(args.output, sha256(args.output), None)
        result.update({"version": plan["version"], "bytes": args.output.stat().st_size,
                       "prepared_identity": audit["prepared_identity"], "epoch_audit": audit["epochs"]})
        args.output.with_suffix(".receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    elif args.action == "restore":
        if not args.storage_root or not args.archive or not args.sha256:
            parser.error("restore requires --storage-root, --archive and --sha256")
        with tempfile.TemporaryDirectory(prefix="ablation-restore-") as temporary:
            staging = Path(temporary) / "verified"
            result = verify(args.archive, args.sha256, staging)
            metadata = json.loads((staging / "transfer-manifest.json").read_text())
            if metadata["version"] != plan["version"] or any(not name.startswith("views/") for name in metadata["files"]):
                raise ValueError("ZIP is not the declared ablation view bundle.")
            copy_verified_tree(staging / "views", views)
        result["pair_audit"] = audit_view_pair(views, reference)
        result["noise_regenerated_on_colab"] = False
    else:
        if not args.storage_root:
            parser.error("ready requires --storage-root")
        result = verify_ready(args.storage_root)
    if args.action != "package" and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
