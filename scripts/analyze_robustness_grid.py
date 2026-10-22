"""Verify saved grid predictions, aggregate corruption seeds, and plot severity curves."""

import argparse
import json
import shutil
import statistics
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_noise import compare_runs
from turkish_punctuation.data import read_split
from turkish_punctuation.gpu_evaluation import write_in_place
from turkish_punctuation.preprocessing import partition_records
from turkish_punctuation.robustness import read_predictions
from turkish_punctuation.training_io import sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    args = parser.parse_args()
    matrix = json.loads((ROOT / "configs/robustness-grid.json").read_text())
    directories, summaries = {}, {}
    for arm, name in zip(("published", "clean", "augmented"), matrix["configuration_order"], strict=True):
        config = json.loads((ROOT / name).read_text())
        directories[arm] = args.storage_root / config["output_directory"]
        summaries[arm] = json.loads((directories[arm] / "summary.json").read_text())
    published = args.storage_root / matrix["output_directory"] / "analysis"
    if (published / "complete.json").exists():
        marker = json.loads((published / "complete.json").read_text())
        old = json.loads((published / "analysis.json").read_text())
        if old["summary_sha256"] != {a:sha256(d/"summary.json") for a,d in directories.items()} or old["analysis_source_sha256"] != sha256(Path(__file__)):
            raise ValueError("Completed analysis belongs to different inputs/source; use a new version.")
        for name,digest in marker["files"].items():
            if sha256(published/name) != digest:
                raise ValueError("Committed analysis changed.")
        print("Verified already completed grid analysis.",flush=True)
        return
    temporary = TemporaryDirectory(prefix="robustness-analysis-")
    destination = Path(temporary.name) / "analysis"
    source = json.loads((ROOT / "docs/data-manifest.json").read_text())
    item = next(item for item in source["files"] if item["split"] == "validation")
    raw = args.input_root / source["local_directory"] / item["filename"]
    if sha256(raw) != item["sha256"]:
        raise ValueError("Raw validation hash differs.")
    eligible, _ = partition_records(read_split(raw))
    records = {row["row_id"]: row for row in eligible.to_dict(orient="records")}
    with (directories["published"] / "conditions/clean-anchor/predictions.jsonl").open() as stream:
        ordered = [json.loads(line)["row_id"] for line in stream]
    if len(ordered) != len(set(ordered)) or set(ordered) != set(records):
        raise ValueError("Grid does not cover each eligible validation row exactly once.")
    records = {key: records[key] for key in ordered}
    expected_conditions = [f"{family}-{level}-seed{seed}" for seed in matrix["noise_seeds"]
                           for level in matrix["levels"] for family in matrix["families"]]
    expected_conditions += matrix["stress_conditions"]
    for arm, summary in summaries.items():
        if list(summary["conditions"]) != expected_conditions or summary["run"]["device"] != "cuda":
            raise ValueError("Grid membership/device differs.")
        for name, report in {"clean-anchor":summary["clean_anchor"], **summary["conditions"]}.items():
            predictions = directories[arm] / "conditions" / name / "predictions.jsonl"
            if sha256(predictions) != report["predictions_sha256"]:
                raise ValueError(f"Prediction hash differs: {arm}/{name}")
            _, metrics = read_predictions(predictions, records)
            if metrics != report["metrics"] or metrics["evaluated_words"] != 179164:
                raise ValueError(f"Saved metrics do not reproduce: {arm}/{name}")
        print(f"Verified 39 conditions against raw targets: {arm}", flush=True)
    comparisons = {}
    for current, reference in (("clean", "published"), ("augmented", "published"), ("augmented", "clean")):
        comparisons[f"{current}-vs-{reference}"] = compare_runs(
            directories[current], summaries[current], directories[reference])
    destination.mkdir(parents=True)
    for name, (data, markdown) in comparisons.items():
        write_in_place(destination / f"{name}.json", data)
        (destination / f"{name}.md").write_text(markdown)
    aggregates = {}
    def mean_sd(values):
        return {"mean":statistics.mean(values), "sample_sd":statistics.stdev(values),
                "values_by_corruption_seed":dict(zip(map(str,matrix["noise_seeds"]),values,strict=True))}
    for family in matrix["families"]:
        aggregates[family] = {}
        for level in matrix["levels"]:
            entry = {}
            names = [f"{family}-{level}-seed{seed}" for seed in matrix["noise_seeds"]]
            for arm, summary in summaries.items():
                conditions = [summary["conditions"][name] for name in names]
                entry[arm] = {
                    "macro_f1_percent":mean_sd([100*r["metrics"]["punctuation_macro"]["f1"] for r in conditions]),
                    "drop_pp":mean_sd([r["degradation_pp"]["punctuation_macro"]["f1"] for r in conditions]),
                    "per_class_f1_percent":{label:mean_sd([100*r["metrics"]["per_class"][label]["f1"] for r in conditions])
                                            for label in ("PERIOD","COMMA","QUESTION_MARK")},
                    "realized_changed_word_percent": [r["input_audit"]["changes"]["changed_word_percent"] for r in conditions],
                }
            entry["paired_augmented_minus_clean_pp"] = mean_sd([
                100*(summaries["augmented"]["conditions"][name]["metrics"]["punctuation_macro"]["f1"]
                     -summaries["clean"]["conditions"][name]["metrics"]["punctuation_macro"]["f1"])
                for name in names])
            aggregates[family][level] = entry
    endpoints = {name:{arm:{"metrics":s["conditions"][name]["metrics"],
                     "degradation_pp":s["conditions"][name]["degradation_pp"],
                     "coverage_percent":s["conditions"][name]["input_audit"]["changes"]["changed_word_percent"]}
                     for arm,s in summaries.items()} for name in matrix["stress_conditions"]}
    result = {"status":"verified_complete", "summary_sha256":{a:sha256(d/"summary.json") for a,d in directories.items()},
              "analysis_source_sha256":sha256(Path(__file__)), "gpu_runtime":summaries["published"]["identity"]["gpu_runtime"],
              "stochastic_grid":aggregates, "deterministic_endpoints":endpoints,
              "variation":"sample SD across three corruption seeds; one fine-tuning seed, not training-seed SD",
              "evaluations_verified":117, "test_evaluated":False}
    write_in_place(destination / "analysis.json", result)
    table = ["# GPU validation robustness grid", "",
             "Macro-F1: mean ± sample SD over corruption seeds 586/587/588. One training seed; no held-out test.", "",
             "| Family | Severity | Published | Clean | Augmented | Paired augmented − clean (pp) |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    fmt = lambda value:f"{value['mean']:.4f} ± {value['sample_sd']:.4f}"
    for family,levels in aggregates.items():
        for level,entry in levels.items():
            table.append(f"| {family} | {level} | " + " | ".join(fmt(entry[a]["macro_f1_percent"]) for a in summaries)
                         + f" | {fmt(entry['paired_augmented_minus_clean_pp'])} |")
    table += ["", "Full-loss endpoints are separate deterministic stresses, not another matched heavy dose.",
              "Combined has three component budgets; report its realized union coverage.",
              "Historical CPU controls are diagnostic only; all curves and drops here use fresh GPU controls."]
    (destination / "results.md").write_text("\n".join(table)+"\n")
    for key,ylabel,filename in (("macro_f1_percent","Punctuation macro-F1 (%)","absolute-f1"),
                                ("drop_pp","Clean-to-noisy F1 drop (pp)","degradation")):
        figure,axes = plt.subplots(2,2,figsize=(10,7),layout="constrained",sharex=True)
        for axis,family in zip(axes.flat,matrix["families"],strict=True):
            for arm,color in (("published","#666666"),("clean","#0072B2"),("augmented","#D55E00")):
                points=[aggregates[family][level][arm][key] for level in matrix["levels"]]
                axis.errorbar([2,5,10],[p["mean"] for p in points],yerr=[p["sample_sd"] for p in points],
                              marker="o",capsize=3,label=arm,color=color)
            axis.set_title(family.replace("_"," ")); axis.set_ylabel(ylabel)
            axis.set_xlabel("Component word budget (%)"); axis.set_xticks([2,5,10]); axis.grid(alpha=.2)
        axes.flat[0].legend()
        figure.suptitle("Validation; SD across 3 corruption seeds; 1 training seed")
        figure.savefig(destination/f"{filename}.png",dpi=240)
        figure.savefig(destination/f"{filename}.pdf")
        plt.close(figure)
    shutil.copy2(Path(__file__),destination/Path(__file__).name)
    published.mkdir(parents=True,exist_ok=True)
    hashes = {}
    for path in destination.iterdir():
        shutil.copyfile(path,published/path.name)
        hashes[path.name] = sha256(path)
        if sha256(published/path.name) != hashes[path.name]:
            raise OSError("Analysis publication readback differs.")
    write_in_place(published/"complete.json",{"files":hashes})
    temporary.cleanup()
    print(f"Verified severity/seed tables and figures: {published}",flush=True)


if __name__ == "__main__":
    main()
