"""Audit all 52 saved transfer predictions and predefined paired contrasts, on CPU."""

import argparse
import csv
import json
import shutil
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import load_from_disk

from turkish_punctuation.gpu_evaluation import condition_committed
from turkish_punctuation.heldout import verify_commit
from turkish_punctuation.heldout_runtime import score_cohorts
from turkish_punctuation.robustness import metric_degradation
from turkish_punctuation.training_io import file_hashes, sha256, utc_now, value_digest, write_json
from turkish_punctuation.transfer_execution import (
    load_execution,
    ordered_inputs,
    read,
    runtime_specification,
    verify_readiness,
    view_path,
)

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("elongation", "internal_omission", "partial_folding", "full_folding_matched")


def flatten(metrics):
    result = {f"{average}.{key}": metrics[average][key]
              for average in ("punctuation_macro", "punctuation_micro")
              for key in ("precision", "recall", "f1")}
    result.update({f"per_class.{label}.{key}": values[key]
                   for label, values in metrics["per_class"].items()
                   for key in ("precision", "recall", "f1")})
    return result


def aggregate(cohort, arm, group, metric, values, unit, seeds):
    return {"cohort": cohort, "arm_or_contrast": arm, "condition_group": group,
            "metric": metric, "unit": unit, "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) == 3 else None,
            "n_corruption_seeds": len(seeds), "seeds": seeds, "values": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        marker = verify_commit(args.output)
        previous = read(args.output / "analysis.json")
        if (previous["identity"] != marker["identity"]
                or previous["specification"]["analysis_source_sha256"] != sha256(Path(__file__))):
            raise ValueError("Completed analysis differs; preserve and version it")
        print("TRANSFER_ANALYSIS_REUSED: exact complete saved analysis; no recomputation", flush=True)
        return
    context = load_execution(ROOT, args.input_root)
    config, execution, original, freeze, cohorts, manifest, clean, _ = context
    run = read(args.run_directory / "provenance/run.json")
    identity, spec = run["identity"], run["specification"]
    runtime = spec["runtime_binding"]["runtime"]
    ready = verify_readiness(ROOT, args.input_root, args.readiness, context, runtime)
    reference = read(args.input_root / original["inference"]["reference_runtime_receipt"])
    if (runtime["gpu_runtime"] != reference["gpu_runtime"] or runtime["packages"] != reference["packages"]
            or runtime["reference_runtime_receipt_sha256"] != sha256(
                args.input_root / original["inference"]["reference_runtime_receipt"])):
        raise ValueError("Recorded inference runtime differs from the pinned original runtime")
    if (value_digest(spec) != identity
            or verify_commit(args.run_directory / "provenance")["identity"] != identity
            or spec["runtime_binding"] != runtime_specification(ROOT, args.input_root, context, runtime)
            or spec["readiness_identity"] != ready["identity"]
            or spec["readiness_commit_sha256"] != sha256(args.readiness / "complete.json")
            or run["training_performed"]):
        raise ValueError("Run provenance differs from frozen inputs/models/runtime")
    progress = read(args.run_directory / "progress.json")
    if (progress["identity"] != identity or progress["status"] != "inference_complete"
            or progress["completed_conditions"] != 52 or progress["planned_conditions"] != 52):
        raise ValueError("Incomplete inference matrix")
    for name, expected in spec["runtime_binding"]["source_sha256"].items():
        if sha256(args.run_directory / "provenance/source_snapshot" / name) != expected:
            raise ValueError(f"Inference source snapshot differs: {name}")
    order, originals = ordered_inputs(args.input_root, config, clean)
    batching = read(args.run_directory / "provenance/batching.json")
    batch = config["inference"]["batch_size"]
    if (value_digest(batching) != spec["batching_sha256"]
            or batching["source_row_ids"] != list(originals) or batching["batch_size"] != batch
            or set(batching["padded_tokens"]) != set(execution["conditions"])):
        raise ValueError("Shared batching identity/order differs")
    for name in execution["conditions"]:
        dataset = load_from_disk(str(view_path(args.input_root, config, name))).select(order)
        lengths = [len(row) for row in dataset["input_ids"]]
        padded = sum(max(lengths[i:i+batch])*len(lengths[i:i+batch]) for i in range(0, len(lengths), batch))
        if padded != batching["padded_tokens"][name]:
            raise ValueError(f"Saved padded-token work differs: {name}")
    reports, commits, anchor_matches, seconds = {}, {}, {}, 0.0
    for arm in execution["model_order"]:
        reports[arm] = {}
        anchor = None
        for name in execution["conditions"]:
            directory = args.run_directory / "models" / arm / "conditions" / name
            marker = verify_commit(directory)
            if (not condition_committed(directory, identity) or marker["identity"] != identity
                    or set(marker["files"]) != {"predictions.jsonl", "metrics.json", "error_examples.json"}):
                raise ValueError(f"Incomplete condition: {arm}/{name}")
            predictions, metrics = score_cohorts(directory / "predictions.jsonl", originals, cohorts)
            if list(predictions) != list(originals):
                raise ValueError("Saved prediction order differs from shared original order")
            if anchor is None:
                anchor = metrics
                old = read(args.input_root / "artifacts/evaluations/held-out-test-v1/models" / arm
                           / "conditions/clean-anchor/metrics.json")
                if metrics != old["cohorts"]:
                    raise ValueError(f"Fresh clean anchor differs from original: {arm}")
                anchor_matches[arm] = {"both_cohorts_equal": True,
                                      "original_metrics_sha256": sha256(args.input_root
                                          / "artifacts/evaluations/held-out-test-v1/models" / arm
                                          / "conditions/clean-anchor/metrics.json")}
            saved = read(directory / "metrics.json")
            drops = {c: metric_degradation(anchor[c], value) for c, value in metrics.items()}
            exposure = {"primary": manifest["conditions"][name]["changes"],
                        "sensitivity": manifest["conditions"][name]["sensitivity_changes"]}
            if (saved["identity"] != identity or saved["arm"] != arm or saved["condition"] != name
                    or saved["split"] != "test" or saved["cohorts"] != metrics
                    or saved["own_clean_to_noisy_drop_pp"] != drops
                    or saved["realized_input_changes"] != exposure
                    or saved["predictions_sha256"] != sha256(directory / "predictions.jsonl")
                    or saved["model_selection_performed"] or saved["training_performed"]
                    or read(directory / "error_examples.json") != {"examples": [], "selection":
                        "no_qualitative_sample_specified_for_supplemental_runner"}):
                raise ValueError(f"Saved scores/exposure/drop identity differs: {arm}/{name}")
            reports[arm][name] = {"cohorts": metrics, "own_clean_to_noisy_drop_pp": drops,
                                 "realized_input_changes": exposure}
            commits[f"{arm}/{name}"] = sha256(directory / "complete.json")
            seconds += saved["inference_loop_seconds_including_io"]
            print(f"TRANSFER_SAVED_TARGETS_OK {arm} {name}: both cohorts", flush=True)
    actual = {p.parent.relative_to(args.run_directory / "models").as_posix()
              for p in (args.run_directory / "models").glob("*/conditions/*/complete.json")}
    expected = {f"{arm}/conditions/{name}" for arm in execution["model_order"] for name in execution["conditions"]}
    if actual != expected or len(commits) != 52:
        raise ValueError("Unexpected condition inventory")
    groups = {"clean-anchor": (["clean-anchor"], [])}
    seeds = config["noise"]["seeds"]
    groups.update({family: ([f"{family}-corpus-seed{s}" for s in seeds], seeds) for family in FAMILIES})
    aggregates = []
    for cohort in ("primary", "sensitivity"):
        for group, (names, group_seeds) in groups.items():
            for arm in execution["model_order"]:
                flat = [flatten(reports[arm][name]["cohorts"][cohort]) for name in names]
                clean_flat = flatten(reports[arm]["clean-anchor"]["cohorts"][cohort])
                for metric in flat[0]:
                    values = [f[metric] for f in flat]
                    aggregates.append(aggregate(cohort, arm, group, metric, values, "fraction", group_seeds))
                    aggregates.append(aggregate(cohort, arm + "_own_clean_drop", group, metric,
                                                [100*(clean_flat[metric]-v) for v in values], "pp", group_seeds))
            for left, right in [("augmented", "clean"), ("augmented", "without_case_loss")]:
                paired = [(flatten(reports[left][n]["cohorts"][cohort]),
                           flatten(reports[right][n]["cohorts"][cohort])) for n in names]
                for metric in paired[0][0]:
                    aggregates.append(aggregate(cohort, left + "_minus_" + right, group, metric,
                                                [100*(a[metric]-b[metric]) for a, b in paired], "pp", group_seeds))
        for arm in execution["model_order"]:
            paired = [(flatten(reports[arm][f"partial_folding-corpus-seed{s}"]["cohorts"][cohort]),
                       flatten(reports[arm][f"full_folding_matched-corpus-seed{s}"]["cohorts"][cohort]))
                      for s in seeds]
            for metric in paired[0][0]:
                aggregates.append(aggregate(cohort, arm, "partial_minus_matched_full", metric,
                                            [100*(a[metric]-b[metric]) for a, b in paired], "pp", seeds))
    specification = {"version": "transfer-saved-prediction-analysis-v1", "run_identity": identity,
                     "run_commit_sha256": sha256(args.run_directory / "provenance/complete.json"),
                     "condition_commits_sha256": commits, "analysis_source_sha256": sha256(Path(__file__)),
                     "freeze_identity": freeze["identity"], "input_identity": manifest["identity"],
                     "readiness_commit_sha256": sha256(args.readiness / "complete.json"),
                     "clean_anchor_reference_matches": anchor_matches,
                     "corruption_seeds": seeds, "sample_sd_ddof": 1,
                     "contrasts": ["augmented_minus_clean", "augmented_minus_without_case_loss",
                                   "partial_minus_matched_full", "own_clean_drop"],
                     "training_seed_uncertainty_estimated": False}
    analysis_id = value_digest(specification)
    analysis = {"identity": analysis_id, "specification": specification,
                "created_at_utc": utc_now(), "verified_forward_conditions": 52,
                "verified_cohort_score_sets": 104, "model_inference_performed": False,
                "test_suite_run": False, "recorded_inference_loop_seconds_sum": seconds,
                "conditions": reports, "aggregates": aggregates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".transfer-analysis-", dir=args.output.parent) as temporary:
        staged = Path(temporary)
        write_json(staged / "analysis.json", analysis)
        shutil.copyfile(__file__, staged / "analysis_source.py")
        with (staged / "aggregates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
            writer.writeheader()
            for row in aggregates:
                writer.writerow({**row, "seeds": json.dumps(row["seeds"]), "values": json.dumps(row["values"])})
        write_json(staged / "complete.json", {"identity": analysis_id, "files": file_hashes(staged)})
        verify_commit(staged)
        staged.rename(args.output)
    print(f"TRANSFER_ANALYSIS_COMPLETE {analysis_id}: 52 forwards /104 cohort scores; no new inference", flush=True)


if __name__ == "__main__":
    main()
