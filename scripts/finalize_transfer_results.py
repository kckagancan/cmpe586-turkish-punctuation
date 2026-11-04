"""Package verified transfer results/report editions; preserve separate original evidence."""

import argparse
import json
from pathlib import Path

from package_heldout_execution import build, inspect_bundle

from turkish_punctuation.heldout import verify_commit
from turkish_punctuation.training_io import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/evaluations/transfer-noise-v1"
COMPLETION = ROOT / "artifacts/evaluation-preparation/transfer-noise-v1/completion-2026-11-04"
ORIGINAL = ROOT / "artifacts/publication/original-study-v1"


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=COMPLETION)
    args = parser.parse_args()
    analysis = read(RUN / "analysis-v1/analysis.json")
    identity = verify_commit(RUN / "analysis-v1")["identity"]
    if (analysis["identity"] != identity or analysis["verified_forward_conditions"] != 52
            or analysis["verified_cohort_score_sets"] != 104 or analysis["model_inference_performed"]):
        raise ValueError("Complete original-target analysis required")
    run = read(RUN / "provenance/run.json")
    if (analysis["specification"]["run_identity"] != run["identity"]
            or verify_commit(RUN / "provenance")["identity"] != run["identity"]):
        raise ValueError("Run/analysis identity differs")
    for key, expected in analysis["specification"]["condition_commits_sha256"].items():
        arm, name = key.split("/")
        directory = RUN / "models" / arm / "conditions" / name
        if sha256(directory / "complete.json") != expected or verify_commit(directory)["identity"] != run["identity"]:
            raise ValueError(f"Completed prediction evidence changed: {key}")
    preservation = read(ORIGINAL / "preservation.json")
    old_zip = Path(preservation["archive"])
    if sha256(old_zip) != preservation["archive_sha256"]:
        raise ValueError("Original report/source snapshot changed")
    for name, expected in preservation["preserved_pdfs"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Original edition changed: {name}")
    old_results = ROOT / "artifacts/evaluation-preparation/held-out-test-v1/completion-2026-10-23/held-out-test-v1-results.zip"
    if sha256(old_results) != "2463a23ed22dc9543341393bedfb902f693448330da860374c708748d4830925":
        raise ValueError("Original experiment results archive changed")
    pdf = read(COMPLETION / "pdf-verification.json")
    if not pdf["passed"] or pdf["public_pages"] != pdf["private_pages"]:
        raise ValueError("Verified public/private final PDFs required")
    for edition, name in [("public", "output/pdf/turkish-punctuation-report.pdf"),
                          ("private", "output/pdf/course-submission-report.pdf")]:
        if sha256(ROOT / name) != pdf[f"{edition}_pdf_sha256"]:
            raise ValueError("Final PDF changed since content/layout verification")
    files = {}

    def add(path, name=None):
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing/unsafe archive input: {path}")
        files[name or path.relative_to(ROOT).as_posix()] = path

    def add_tree(path):
        for item in path.rglob("*"):
            if item.is_file():
                add(item)

    add_tree(RUN)
    ready = COMPLETION / "downloaded/preparation/transfer-noise-v1/readiness-v1"
    verify_commit(ready)
    add_tree(ready)
    for config_path in ("configs/transfer-evaluation.json", "configs/held-out-test.json"):
        config = read(ROOT / config_path)
        frozen = ROOT / config["paths"]["freeze_directory"]
        verify_commit(frozen)
        add_tree(frozen)
    for name in ("manifest.json", "complete.json"):
        add(ROOT / "data/processed/transfer-noise-v1" / name)
    for name in ("raw-archive.json", "restoration.json", "local-analysis.log", "pdf-verification.json"):
        add(COMPLETION / name)
    add(ORIGINAL / "preservation.json")
    add(old_zip, "original-study-v1/original-report-source-and-evidence.zip")
    add(old_results, "original-study-v1/held-out-test-v1-results.zip")
    for item in (ROOT / "report").rglob("*"):
        if (item.is_file() and "proposal" not in item.relative_to(ROOT / "report").parts
                and item.suffix in {".tex", ".sty", ".json", ".md", ".png", ".pdf"}):
            add(item)
    for name in ("turkish-punctuation-report.pdf", "course-submission-report.pdf",
                 "original-study-v1-public.pdf", "original-study-v1-course-submission.pdf"):
        add(ROOT / "output/pdf" / name)
    for item in (ROOT / "docs").iterdir():
        if item.is_file() and (item.name.startswith("transfer-") and item.name != "transfer-evaluation-completion.md"
                              or item.name == "report-versions.md"):
            add(item)
    for name in ("scripts/analyze_transfer.py", "scripts/archive_transfer_results.py",
                 "scripts/report_transfer_results.py", "scripts/build_transfer_report.py",
                 "scripts/finalize_transfer_results.py", "scripts/package_heldout_execution.py"):
        add(ROOT / name)
    for item in (ROOT / "configs").glob("transfer-*.json"):
        add(item)
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "transfer-noise-v1-results-and-report-v2.zip"
    result = build(archive, "verified-transfer-results-and-report-v2", files,
                   analysis["specification"]["freeze_identity"])
    manifest = inspect_bundle(archive)
    if manifest["files"] != {name: sha256(path) for name, path in sorted(files.items())}:
        raise ValueError("Final complete archive readback differs")
    receipt = {"status": "verified_complete", "run_identity": run["identity"], "analysis_identity": identity,
               "archive": str(archive.resolve()), **result, "original_report_edition_preserved": True,
               "original_result_archive_sha256": sha256(old_results),
               "verified_forward_conditions": 52, "verified_cohort_score_sets": 104,
               "member_sha256_readback_passed": True, "contains_private_course_metadata": True,
               "training_performed": False, "inference_repeated": False,
               "test_suite_run": False, "weights_included": False, "drive_deleted": False}
    write_json(args.output / "final-archive.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
