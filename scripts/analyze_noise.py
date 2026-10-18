"""Summarize a completed noise run and distinguish direct/contextual new errors."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/noise-evaluation.json").read_text())
    run = (ROOT / (args.run_directory or config["output_directory"])).resolve()
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    destination = (ROOT / (args.output_directory or run / "analysis")).resolve()
    if destination.exists():
        raise FileExistsError("Analysis already exists; choose a new --output-directory.")
    prepared = ROOT / summary["run"]["prepared_directory"]
    anchor_path = run / "conditions/clean-anchor/predictions.jsonl"
    if sha256(anchor_path) != summary["clean_anchor"]["predictions_sha256"]:
        raise ValueError("Clean-anchor prediction hash differs.")
    anchor = {row["row_id"]: row for row in read_jsonl(anchor_path)}
    analysis = {
        "summary_sha256": sha256(summary_path),
        "clean_anchor_predictions_sha256": sha256(anchor_path),
        "analysis_source_sha256": sha256(Path(__file__)),
        "context_definition": "A word is directly changed iff its final text differs from its first pre-edit text; context-only means the word is unchanged but another word in its record changed.",
        "conditions": {},
    }
    table = [
        "# Noise pilot comparison",
        "",
        "F1 values are percentages; degradation is positive clean-minus-noisy percentage points.",
        "",
        "| Condition | Changed words (%) | Macro F1 | Drop (pp) | Period F1 | Comma F1 | Question F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    clean = summary["clean_reference_metrics"]
    table.append(
        f"| Clean control | 0.00 | {100 * clean['punctuation_macro']['f1']:.2f} | 0.00 | "
        + " | ".join(
            f"{100 * clean['per_class'][label]['f1']:.2f}"
            for label in ("PERIOD", "COMMA", "QUESTION_MARK")
        )
        + " |"
    )
    for name, report in summary["conditions"].items():
        prediction_path = run / "conditions" / name / "predictions.jsonl"
        if sha256(prediction_path) != report["predictions_sha256"]:
            raise ValueError(f"{name}: predictions changed.")
        source = prepared / "conditions" / name
        for filename in ("records.jsonl", "edits.jsonl"):
            if (
                sha256(source / filename)
                != summary["identity"]["condition_input_sha256"][name][filename]
            ):
                raise ValueError(f"{name}/{filename}: prepared data changed.")
        records = {row["row_id"]: row for row in read_jsonl(source / "records.jsonl")}
        first_text = {}
        for edit in read_jsonl(source / "edits.jsonl"):
            first_text.setdefault((edit["row_id"], edit["word_index"]), edit["before"])
        changed = {row_id: set() for row_id in records}
        for (row_id, index), original in first_text.items():
            if records[row_id]["tokens"][index] != original:
                changed[row_id].add(index)
        if sum(map(len, changed.values())) != report["input_audit"]["changes"]["changed_words"]:
            raise ValueError(f"{name}: reconstructed final changes disagree with the data audit.")
        context, per_class = (
            Counter(),
            {label: Counter() for label in ("O", "PERIOD", "COMMA", "QUESTION_MARK")},
        )
        new_period_omissions = next_word_changed = next_changed_error_word_unchanged = 0
        predictions = read_jsonl(prediction_path)
        if len(predictions) != len(anchor) or {row["row_id"] for row in predictions} != set(anchor):
            raise ValueError(f"{name}: paired prediction IDs differ.")
        for row in predictions:
            row_id = row["row_id"]
            baseline = anchor[row_id]
            if row["reference_ids"] != baseline["reference_ids"]:
                raise ValueError(f"{name}: paired targets differ.")
            for index, (true, old, new) in enumerate(
                zip(
                    row["reference_ids"],
                    baseline["prediction_ids"],
                    row["prediction_ids"],
                    strict=True,
                )
            ):
                if old != true or new == true:
                    continue
                category = (
                    "edited_word"
                    if index in changed[row_id]
                    else "unedited_word_edited_context"
                    if changed[row_id]
                    else "unedited_record"
                )
                context[category] += 1
                label = ("O", "PERIOD", "COMMA", "QUESTION_MARK")[true]
                per_class[label][category] += 1
                if true == 1 and new == 0:
                    new_period_omissions += 1
                    next_word_changed += index + 1 in changed[row_id]
                    next_changed_error_word_unchanged += (
                        index not in changed[row_id] and index + 1 in changed[row_id]
                    )
        if sum(context.values()) != report["paired_analysis"]["outcomes"]["new_error"]:
            raise ValueError(f"{name}: new-error population does not reproduce paired analysis.")
        analysis["conditions"][name] = {
            "new_errors_by_input_location": dict(context),
            "new_errors_by_reference_class_and_location": {
                label: dict(value) for label, value in per_class.items()
            },
            "new_period_omissions": new_period_omissions,
            "new_period_omissions_with_following_word_changed": next_word_changed,
            "new_period_omissions_at_unedited_word_with_following_word_changed": next_changed_error_word_unchanged,
            "input_changed_words": sum(map(len, changed.values())),
        }
        metrics = report["metrics"]
        table.append(
            f"| {name} | {report['input_audit']['changes']['changed_word_percent']:.2f} | "
            f"{100 * metrics['punctuation_macro']['f1']:.2f} | "
            f"{report['degradation_pp']['punctuation_macro']['f1']:.2f} | "
            + " | ".join(
                f"{100 * metrics['per_class'][label]['f1']:.2f}"
                for label in ("PERIOD", "COMMA", "QUESTION_MARK")
            )
            + " |"
        )
    table.extend(
        [
            "",
            "## Paired new errors",
            "",
            "| Condition | Edited word | Unedited word, edited context | Completely unedited record |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for name, result in analysis["conditions"].items():
        counts = result["new_errors_by_input_location"]
        table.append(
            f"| {name} | {counts.get('edited_word', 0)} | "
            f"{counts.get('unedited_word_edited_context', 0)} | {counts.get('unedited_record', 0)} |"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".analysis-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "result"
        staging.mkdir()
        (staging / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")
        (staging / "comparison.md").write_text("\n".join(table) + "\n")
        shutil.copy2(Path(__file__), staging / "analyze_noise.py")
        staging.rename(destination)
    print(f"Saved verified paired analysis: {destination}")


if __name__ == "__main__":
    main()
