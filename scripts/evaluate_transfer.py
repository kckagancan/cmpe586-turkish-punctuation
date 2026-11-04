"""Explicit resumable 52-forward supplemental evaluation on fixed checkpoints."""

import argparse
import gc
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch
from datasets import disable_progress_bars, load_from_disk

from turkish_punctuation.gpu_evaluation import (
    condition_committed,
    publish_condition,
    write_in_place,
)
from turkish_punctuation.heldout import load_model, verify_commit
from turkish_punctuation.heldout_runtime import check_model, runtime_binding, score_cohorts
from turkish_punctuation.inference import iter_predictions
from turkish_punctuation.robustness import metric_degradation
from turkish_punctuation.tokenization import LABEL_TO_ID
from turkish_punctuation.training_io import (
    directory_lock,
    file_hashes,
    sha256,
    utc_now,
    value_digest,
    write_json,
)
from turkish_punctuation.transfer_execution import (
    ordered_inputs,
    read,
    runtime_specification,
    verify_inputs,
    verify_readiness,
    view_path,
)
from turkish_punctuation.transfer_preparation import transfer_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--allow-test-scoring", action="store_true")
    args = parser.parse_args()
    if not args.allow_test_scoring:
        raise ValueError("Transfer scoring is off; require explicit --allow-test-scoring")
    disable_progress_bars()
    context = verify_inputs(ROOT, args.input_root)
    config, execution, original, _, cohorts, manifest, clean, _ = context
    runtime = runtime_binding(ROOT, args.input_root, original)
    readiness = verify_readiness(ROOT, args.input_root, args.readiness, context, runtime)
    order, originals = ordered_inputs(args.input_root, config, clean)
    batch = config["inference"]["batch_size"]
    batching = {"source_row_ids": list(originals), "batch_size": batch, "padded_tokens": {}}
    for name in execution["conditions"]:
        dataset = load_from_disk(str(view_path(args.input_root, config, name))).select(order)
        lengths = [len(row) for row in dataset["input_ids"]]
        batching["padded_tokens"][name] = sum(max(lengths[i:i+batch])*len(lengths[i:i+batch])
                                               for i in range(0, len(lengths), batch))
    specification = {"version": "transfer-evaluation-run-v1", "readiness_identity": readiness["identity"],
        "readiness_commit_sha256": sha256(args.readiness / "complete.json"),
        "runtime_binding": runtime_specification(ROOT, args.input_root, context, runtime),
        "batching_sha256": value_digest(batching)}
    identity = value_digest(specification)
    output = args.storage_root / execution["output_directory"]
    tokenizer = transfer_tokenizer(ROOT, config)
    completed = 0
    with directory_lock(output.parent / "transfer-noise-v1-writer-lock"):
        output.mkdir(parents=True, exist_ok=True)
        provenance = output / "provenance"
        if provenance.exists():
            marker = verify_commit(provenance)
            run = read(provenance / "run.json")
            if marker["identity"] != identity or run["specification"] != specification or run["identity"] != identity:
                raise ValueError("Existing run differs; preserve it and version changes")
        else:
            with TemporaryDirectory(prefix="transfer-provenance-") as temporary:
                staging = Path(temporary)
                write_json(staging / "run.json", {"identity": identity, "specification": specification,
                    "started_at_utc": utc_now(), "training_performed": False})
                write_json(staging / "batching.json", batching)
                for name in specification["runtime_binding"]["source_sha256"]:
                    target = staging / "source_snapshot" / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, target)
                write_json(staging / "complete.json", {"identity": identity, "files": file_hashes(staging)})
                provenance.mkdir()
                for path in staging.rglob("*"):
                    if path.is_file() and path.name != "complete.json":
                        target = provenance / path.relative_to(staging)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(path, target)
                shutil.copyfile(staging / "complete.json", provenance / "complete.json")
            verify_commit(provenance)
        try:
            for arm in execution["model_order"]:
                model, _ = load_model(ROOT, original, arm, args.storage_root, tokenizer)
                check_model(model, original["models"][arm])
                del model
                gc.collect()
                print(f"TRANSFER_CHECKPOINT_REVERIFIED {arm}", flush=True)
            for arm in execution["model_order"]:
                model, clean_metrics = None, None
                try:
                    for name in execution["conditions"]:
                        destination = output / "models" / arm / "conditions" / name
                        if (destination / "complete.json").exists():
                            read(destination / "complete.json")
                        if condition_committed(destination, identity):
                            _, metrics = score_cohorts(destination / "predictions.jsonl", originals, cohorts)
                            saved = read(destination / "metrics.json")
                            if (saved["cohorts"] != metrics or saved["identity"] != identity
                                    or saved["arm"] != arm or saved["condition"] != name):
                                raise ValueError("Committed condition targets/metrics/identity differ")
                            print(f"TRANSFER_REUSED {arm} {name}: both cohorts verified", flush=True)
                        else:
                            if model is None:
                                model, _ = load_model(ROOT, original, arm, args.storage_root, tokenizer)
                                check_model(model, original["models"][arm])
                            dataset = load_from_disk(str(view_path(args.input_root, config, name))).select(order)
                            write_in_place(output / "progress.json", {"identity": identity, "status": "running",
                                "arm": arm, "condition": name, "completed_conditions": completed,
                                "planned_conditions": 52, "updated_at_utc": utc_now()})
                            with TemporaryDirectory(prefix="transfer-condition-") as temporary:
                                staging = Path(temporary)
                                started = perf_counter()
                                with (staging / "predictions.jsonl").open("x") as stream:
                                    count = 0
                                    for row, expected in zip(iter_predictions(dataset, tokenizer, model,
                                        max_length=512, batch_size=batch, device="cuda"), originals, strict=True):
                                        targets = [LABEL_TO_ID[tag] for tag in originals[expected]["tags"]]
                                        if row["row_id"] != expected or row["reference_ids"] != targets:
                                            raise ValueError("Inference changed original IDs/first-subword targets")
                                        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
                                        count += 1
                                        if count % execution["progress_every_records"] == 0 or count == len(originals):
                                            print(f"TRANSFER_PROGRESS {arm} {name}: {count}/{len(originals)}", flush=True)
                                _, metrics = score_cohorts(staging / "predictions.jsonl", originals, cohorts)
                                anchor = metrics if name == "clean-anchor" else clean_metrics
                                report = {"identity": identity, "arm": arm, "condition": name, "split": "test",
                                    "cohorts": metrics, "own_clean_to_noisy_drop_pp": {
                                        cohort: metric_degradation(anchor[cohort], value) for cohort, value in metrics.items()},
                                    "realized_input_changes": {"primary": manifest["conditions"][name]["changes"],
                                        "sensitivity": manifest["conditions"][name]["sensitivity_changes"]},
                                    "model_selection_performed": False, "training_performed": False,
                                    "inference_loop_seconds_including_io": perf_counter()-started,
                                    "predictions_sha256": sha256(staging / "predictions.jsonl")}
                                write_json(staging / "metrics.json", report)
                                write_json(staging / "error_examples.json", {"examples": [],
                                    "selection": "no_qualitative_sample_specified_for_supplemental_runner"})
                                publish_condition(staging, destination, identity)
                            print(f"TRANSFER_COMMITTED {arm} {name}", flush=True)
                        if name == "clean-anchor":
                            clean_metrics = metrics
                        completed += 1
                        write_in_place(output / "progress.json", {"identity": identity, "status": "running",
                            "completed_conditions": completed, "planned_conditions": 52,
                            "last_committed": [arm, name], "updated_at_utc": utc_now()})
                finally:
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
            if completed != 52 or runtime_specification(ROOT, args.input_root, context, runtime) != specification["runtime_binding"]:
                raise ValueError("Matrix incomplete or execution binding changed")
            write_in_place(output / "progress.json", {"identity": identity, "status": "inference_complete",
                "completed_conditions": completed, "planned_conditions": 52, "updated_at_utc": utc_now(),
                "next_action": "Audit saved raw-target predictions and aggregate all specified contrasts; no new inference"})
        except BaseException as error:
            write_in_place(output / "progress.json", {"identity": identity, "status": "interrupted_or_failed",
                "completed_conditions": completed, "planned_conditions": 52, "error": repr(error), "updated_at_utc": utc_now()})
            raise
    print("TRANSFER_INFERENCE_COMPLETE: 52 conditions /104 cohort score sets; saved-prediction audit next", flush=True)


if __name__ == "__main__":
    main()
