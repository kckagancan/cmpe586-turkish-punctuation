"""Build supplemental paper tables/bindings from public aggregates; optionally draw a figure."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("docs/transfer-evaluation-aggregates.csv", "docs/transfer-evaluation-condition-summary.json",
           "docs/transfer-evaluation-per-class.csv", "docs/transfer-evaluation-input-coverage.csv",
           "configs/transfer-evaluation.json", "configs/transfer-noise-profile.json",
           "configs/transfer-execution.json")
ARMS = ("published", "clean", "augmented", "without_case_loss")
NAMES = ("Published", "Clean", "Augmented", "Without case")
GROUPS = ("clean-anchor", "elongation", "internal_omission", "partial_folding", "full_folding_matched")
LABELS = ("Clean anchor", "Elongation", "Internal omission", "Partial folding", "Matched full fold")


def contents():
    rows = list(csv.DictReader((ROOT / SOURCES[0]).open()))
    index = {(r["cohort"], r["arm_or_contrast"], r["condition_group"], r["metric"]): r for r in rows}
    summary = json.loads((ROOT / SOURCES[1]).read_text())
    if summary["verified_forward_conditions"] != 52 or summary["verified_cohort_score_sets"] != 104:
        raise ValueError("Complete supplemental scores required")

    def metric(arm, group, cohort="primary", key="punctuation_macro.f1"):
        r = index[cohort, arm, group, key]
        factor = 100 if r["unit"] == "fraction" else 1
        values = json.loads(r["values"])
        if len(values) not in (1, 3):
            raise ValueError("Unexpected seed count")
        return f"{factor*float(r['mean']):.4f}" + (
            r" $\pm$ " + f"{factor*float(r['sample_sd']):.4f}" if r["sample_sd"] else "")

    def table(command, headers, body, columns):
        return ("\\newcommand{\\" + command + "}{%\n\\begin{tabular}{" + columns + "}\n\\toprule\n"
                + " & ".join(headers) + " \\\\\n\\midrule\n"
                + "".join(" & ".join(row) + " \\\\\n" for row in body)
                + "\\bottomrule\n\\end{tabular}%\n}\n\n")

    text = "% Generated from verified supplemental aggregate evidence; do not hand-edit.\n"
    for cohort, command in [("primary", "TransferPrimary"), ("sensitivity", "TransferSensitivity")]:
        body = [[label, *[metric(arm, group, cohort) for arm in ARMS]]
                for group, label in zip(GROUPS, LABELS, strict=True)]
        text += table(command, ["Input", *NAMES], body, "lrrrr")
    paired = [[label, metric("augmented_minus_clean", group),
               metric("augmented_minus_clean", group, "sensitivity"),
               metric("augmented_minus_without_case_loss", group)]
              for group, label in zip(GROUPS, LABELS, strict=True)]
    text += table("TransferPaired", ["Input", r"Aug$-$Clean (P)", r"Aug$-$Clean (S)",
                                    r"Aug$-$Without (P)"], paired, "lrrr")
    controls = [[name, metric(arm, "partial_minus_matched_full"),
                 metric(arm, "partial_minus_matched_full", "sensitivity")]
                for arm, name in zip(ARMS, NAMES, strict=True)]
    text += table("TransferControl", ["Model", "Primary (pp)", "Sensitivity (pp)"], controls, "lrr")
    classes = [[label, name, *[metric(arm, group, key=f"per_class.{mark}.f1")
                              for mark in ("PERIOD", "COMMA", "QUESTION_MARK")]]
               for group, label in zip(GROUPS, LABELS, strict=True)
               for arm, name in zip(ARMS, NAMES, strict=True)]
    text += table("TransferClasses", ["Input", "Model", "PERIOD", "COMMA", "QUESTION"], classes, "llrrr")
    drops = [[label, *[metric(arm + "_own_clean_drop", group) for arm in ARMS]]
             for group, label in zip(GROUPS[1:], LABELS[1:], strict=True)]
    text += table("TransferDrops", ["Input", *NAMES], drops, "lrrrr")
    text = text.rstrip() + "\n"
    binding = {"version": "transfer-paper-tables-v1", "run_identity": summary["run_identity"],
               "analysis_identity": summary["analysis_identity"], "training_or_inference_performed": False,
               "sources": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES},
               "generated_tables_sha256": hashlib.sha256(text.encode()).hexdigest()}
    return text, json.dumps(binding, indent=2) + "\n", index


def figure(index):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 2.5))
    groups = GROUPS[1:]
    labels = ["Elongation", "Omission", "Partial fold", "Matched full"]
    for ax, cohort, title in zip(axes, ("primary", "sensitivity"), ("Primary cohort", "Overlap sensitivity"), strict=True):
        for contrast, label, offset, color in [("augmented_minus_clean", "Augmented − Clean", -.09, "#0072B2"),
                ("augmented_minus_without_case_loss", "Augmented − Without case", .09, "#D55E00")]:
            rs = [index[cohort, contrast, group, "punctuation_macro.f1"] for group in groups]
            ax.errorbar([i + offset for i in range(4)], [float(r["mean"]) for r in rs],
                        yerr=[float(r["sample_sd"]) for r in rs], marker="o", linestyle="none",
                        capsize=3, markersize=4, color=color, label=label)
        ax.axhline(0, color="#555555", linewidth=.8)
        ax.set(title=title, xticks=range(4), xticklabels=labels, ylim=(-.06, .38))
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Paired macro-F1 difference (pp)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .9), w_pad=1)
    for extension in ("pdf", "png"):
        fig.savefig(ROOT / f"report/figures/transfer.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    if args.check and args.figures:
        raise ValueError("Read-only check cannot redraw figures")
    tables, binding, index = contents()
    for name, value in [("transfer-tables.tex", tables), ("transfer-results-binding.json", binding)]:
        path = ROOT / "report" / name
        if args.check:
            if path.read_text() != value:
                raise ValueError(f"Generated supplemental artifact differs: {path}")
        else:
            path.write_text(value)
    if args.figures:
        figure(index)
    print("Supplemental paper tables and bindings match." if args.check else "Built supplemental paper artifacts from public aggregates.")


if __name__ == "__main__":
    main()
