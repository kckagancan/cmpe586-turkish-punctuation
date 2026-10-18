"""Paired word-level robustness measurements using the shared inference pipeline."""

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from datasets import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerFast

from turkish_punctuation.data import LABELS
from turkish_punctuation.inference import iter_predictions
from turkish_punctuation.metrics import compute_metrics
from turkish_punctuation.tokenization import LABEL_TO_ID

OUTCOMES = ("both_correct", "new_error", "recovered_error", "changed_error", "unchanged_error")


def read_predictions(path: Path, originals: dict[int, dict]) -> tuple[dict[int, dict], dict]:
    """Validate every saved clean target/order, then recompute its metric reference."""
    expected_ids = list(originals)
    predictions, reference, predicted = {}, [], []
    with path.open(encoding="utf-8") as stream:
        for i, line in enumerate(stream):
            row = json.loads(line)
            row_id = row["row_id"]
            if i >= len(expected_ids) or row_id != expected_ids[i]:
                raise ValueError("Clean prediction IDs/order differ from eligible validation.")
            targets = [LABEL_TO_ID[tag] for tag in originals[row_id]["tags"]]
            if row["reference_ids"] != targets or len(row["prediction_ids"]) != len(targets):
                raise ValueError(f"Row {row_id}: clean reference targets/count differ.")
            predictions[row_id] = row
            reference.extend(targets)
            predicted.extend(row["prediction_ids"])
    if len(predictions) != len(expected_ids):
        raise ValueError("Clean prediction file is incomplete.")
    return predictions, compute_metrics(reference, predicted)


def metric_degradation(clean: dict, noisy: dict) -> dict:
    """Positive values mean lower noisy performance; report percentage points."""
    result = {}
    for average in ("punctuation_macro", "punctuation_micro"):
        result[average] = {
            key: 100 * (clean[average][key] - noisy[average][key])
            for key in ("precision", "recall", "f1")
        }
    result["per_class"] = {
        label: {
            key: 100 * (clean["per_class"][label][key] - noisy["per_class"][label][key])
            for key in ("precision", "recall", "f1")
        }
        for label in LABELS
    }
    return result


def evaluate_variant(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    *,
    clean_records: dict[int, dict],
    noisy_records: dict[int, dict],
    clean_predictions: dict[int, dict],
    config: dict,
    max_length: int,
    prediction_path: Path,
    progress: Callable[[int, float], None],
) -> dict:
    """Stream predictions and count clean-to-noisy outcomes at the same word positions.

    Inspection examples are first occurrences by source order, not representative
    samples. Unedited-row prediction differences are diagnostic because batch
    shapes can also change floating-point execution.
    """
    expected_ids = list(clean_records)
    if list(dataset["row_id"]) != expected_ids or list(noisy_records) != expected_ids:
        raise ValueError("Paired condition has different rows/order from clean validation.")
    reference, prediction = [], []
    outcomes = Counter()
    per_class = {label: Counter() for label in LABELS}
    example_groups: dict[tuple[str, str], dict[int, dict]] = {}
    changed_predictions = changed_prediction_rows = untouched_row_changes = 0
    completed = 0
    started = perf_counter()
    with prediction_path.open("x", encoding="utf-8") as stream:
        for row in iter_predictions(
            dataset,
            tokenizer,
            model,
            max_length=max_length,
            batch_size=config["batch_size"],
            device=config["device"],
        ):
            row_id = row["row_id"]
            if row_id != expected_ids[completed]:
                raise RuntimeError("Noise prediction order changed.")
            clean = clean_records[row_id]
            noisy = noisy_records[row_id]
            targets = [LABEL_TO_ID[tag] for tag in clean["tags"]]
            if (
                row["reference_ids"] != targets
                or noisy["tags"] != clean["tags"]
                or len(row["prediction_ids"]) != len(clean["tokens"])
            ):
                raise RuntimeError(f"Row {row_id}: noise targets/prediction count changed.")
            baseline = clean_predictions[row_id]["prediction_ids"]
            noisy_ids = row["prediction_ids"]
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            reference.extend(targets)
            prediction.extend(noisy_ids)
            input_changes = [
                i
                for i, (a, b) in enumerate(zip(clean["tokens"], noisy["tokens"], strict=True))
                if a != b
            ]
            row_changed = baseline != noisy_ids
            changed_prediction_rows += row_changed
            untouched_row_changes += not input_changes and row_changed
            for position, (true, old, new) in enumerate(
                zip(targets, baseline, noisy_ids, strict=True)
            ):
                if old == true and new == true:
                    outcome = "both_correct"
                elif old == true:
                    outcome = "new_error"
                elif new == true:
                    outcome = "recovered_error"
                elif old != new:
                    outcome = "changed_error"
                else:
                    outcome = "unchanged_error"
                label = LABELS[true]
                outcomes[outcome] += 1
                per_class[label][outcome] += 1
                changed_predictions += old != new
                key = (outcome, label)
                if outcome in ("new_error", "recovered_error", "changed_error"):
                    # Keep earliest original source rows even when batches are length-sorted.
                    group = example_groups.setdefault(key, {})
                    limit = config["error_examples_per_category_and_class"]
                    if row_id not in group and (len(group) < limit or row_id < max(group)):
                        group[row_id] = {
                            "category": outcome,
                            "row_id": row_id,
                            "word_index": position,
                            "reference_class": label,
                            "clean_words": clean["tokens"],
                            "noisy_words": noisy["tokens"],
                            "reference_tags": clean["tags"],
                            "clean_prediction_ids": baseline,
                            "noisy_prediction_ids": noisy_ids,
                            "input_changed_word_indices": input_changes,
                            "error_word_itself_changed": position in input_changes,
                        }
                        if len(group) > limit:
                            del group[max(group)]
            completed += 1
            if completed % config["progress_every_records"] == 0 or completed == len(dataset):
                progress(completed, perf_counter() - started)
    if completed != len(dataset):
        raise RuntimeError("Not all noisy validation records were evaluated.")
    metrics = compute_metrics(reference, prediction)
    examples = sorted(
        [item for group in example_groups.values() for item in group.values()],
        key=lambda item: (item["row_id"], item["word_index"], item["category"]),
    )
    expected_support = Counter(tag for row in clean_records.values() for tag in row["tags"])
    if {label: data["support"] for label, data in metrics["per_class"].items()} != expected_support:
        raise RuntimeError("Noisy class supports differ from clean validation.")
    return {
        "metrics": metrics,
        "evaluated_records": completed,
        "prediction_loop_seconds_including_io": perf_counter() - started,
        "paired_analysis": {
            "outcomes": {key: outcomes[key] for key in OUTCOMES},
            "outcomes_per_reference_class": {
                label: {key: counts[key] for key in OUTCOMES} for label, counts in per_class.items()
            },
            "changed_word_predictions": changed_predictions,
            "rows_with_changed_predictions": changed_prediction_rows,
            "unedited_rows_with_changed_predictions": untouched_row_changes,
        },
        "error_examples": examples,
        "error_sampling": "First source-order occurrences, up to configured count per category/reference class; distinct rows within each group.",
    }
