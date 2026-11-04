"""Publish only aggregate evidence from a completed saved-prediction transfer audit."""

import csv
import json
import shutil
from pathlib import Path

from turkish_punctuation.heldout import verify_commit
from turkish_punctuation.training_io import write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts/evaluations/transfer-noise-v1"
ARMS = ("published", "clean", "augmented", "without_case_loss")
NAMES = ("Published", "Clean-trained", "Augmented", "Without case loss")
GROUPS = ("clean-anchor", "elongation", "internal_omission", "partial_folding", "full_folding_matched")
LABELS = ("Clean anchor", "Elongation", "Internal omission", "Partial folding", "Matched full folding")


def main():
    source = RUN / "analysis-v1"
    analysis = json.loads((source / "analysis.json").read_text())
    if (verify_commit(source)["identity"] != analysis["identity"]
            or analysis["verified_forward_conditions"] != 52
            or analysis["verified_cohort_score_sets"] != 104 or analysis["model_inference_performed"]):
        raise ValueError("A complete original-target audit is required")
    docs = ROOT / "docs"
    shutil.copyfile(source / "aggregates.csv", docs / "transfer-evaluation-aggregates.csv")
    summary = {"version": "transfer-public-condition-summary-v1", "run_identity":
               analysis["specification"]["run_identity"], "analysis_identity": analysis["identity"],
               "verified_forward_conditions": 52, "verified_cohort_score_sets": 104,
               "model_inference_performed_during_analysis": False,
               "conditions": analysis["conditions"]}
    write_json(docs / "transfer-evaluation-condition-summary.json", summary)
    class_rows = []
    for arm in ARMS:
        for condition, report in analysis["conditions"][arm].items():
            for cohort, metrics in report["cohorts"].items():
                for label, values in metrics["per_class"].items():
                    class_rows.append({"cohort": cohort, "model": arm, "condition": condition,
                                       "label": label, **values})
    with (docs / "transfer-evaluation-per-class.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)
    index = {(r["cohort"], r["arm_or_contrast"], r["condition_group"], r["metric"]): r
             for r in analysis["aggregates"]}

    def fmt(arm, group, cohort="primary", metric="punctuation_macro.f1"):
        r = index[cohort, arm, group, metric]
        f = 100 if r["unit"] == "fraction" else 1
        return f"{f*r['mean']:.4f}" + (f" ± {f*r['sample_sd']:.4f}" if r["sample_sd"] is not None else "")

    lines = ["# Corpus-anchored supplemental transfer: verified results", "",
        "Completed **2026-11-04**: **52/52 model-condition forwards and104 cohort score sets**.",
        "All saved file hashes, original row IDs/order/targets, both-cohort metrics, own-anchor",
        "drops, input exposure and shared batch work were independently reconciled locally.",
        "No training, new checkpoint selection or repeated inference was used in analysis.", "",
        "## What this adds", "",
        "The existing selected models are evaluated on elongation, one internal omission and",
        "partial folding, outside the explicit training-v1 augmentation operators. These are",
        "new controlled distortions of already scored project test texts, **not a new natural",
        "corpus test**. MultiLexNorm Turkish train-only morphology anchors the fixed recipe;",
        "its test file remains unread. Sources, condition set and doses were frozen before",
        "the new scores. The extension follows instructor feedback and the original test,",
        "so it is a supplemental follow-up, not blind preregistration.", "",
        "## Complete macro-F1 matrix", "",
        "Values are punctuation macro-F1 (%) over PERIOD, COMMA and QUESTION_MARK.",
        "Noisy cells are mean ± sample SD over seeds586/587/588; clean anchors have one value.",
        "There is one training seed: these SDs measure corruption variation only.", ""]
    for cohort, label in [("primary", "Primary:14,600 rows /181,485 targets"),
                           ("sensitivity", "Sensitivity:14,445 rows /181,116 targets")]:
        lines += [f"### {label}", "", "| Input | " + " | ".join(NAMES) + " |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for group, label in zip(GROUPS, LABELS, strict=True):
            lines.append("| " + " | ".join([label, *[fmt(arm, group, cohort) for arm in ARMS]]) + " |")
        lines += [""]
    lines += ["## Predefined paired contrasts", "",
        "Differences are computed within each corruption seed before mean/SD aggregation (pp).",
        "The primary contrast is augmented minus clean for each new mechanism; the matched",
        "full-folding condition is a seen-mechanism control. All outcomes are retained.", "",
        "| Input | Augmented − clean: primary | Augmented − clean: sensitivity | Augmented − without case loss: primary | Augmented − without case loss: sensitivity |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for group, label in zip(GROUPS, LABELS, strict=True):
        lines.append("| " + " | ".join([label, *[fmt(arm, group, c) for arm, c in [
            ("augmented_minus_clean", "primary"), ("augmented_minus_clean", "sensitivity"),
            ("augmented_minus_without_case_loss", "primary"),
            ("augmented_minus_without_case_loss", "sensitivity")]]]) + " |")
    lines += ["", "| Model | Partial − matched full: primary (pp) | Sensitivity (pp) |",
              "| --- | ---: | ---: |"]
    for arm, label in zip(ARMS, NAMES, strict=True):
        lines.append("| " + " | ".join([label, fmt(arm, "partial_minus_matched_full"),
                                       fmt(arm, "partial_minus_matched_full", "sensitivity")]) + " |")
    lines += ["", "The augmented model has a positive mean advantage over clean training for all",
        "three new mechanisms, but the gains are small at these low anchors. Its clean-anchor",
        "advantage is already0.1387pp: absolute noisy-score gains must be distinguished from",
        "reduced clean-to-noisy degradation. The aggregate CSV retains own-anchor drops for",
        "every metric/model/cohort. This is descriptive transfer under the specified recipe,",
        "not evidence of significance, convergence or general natural-text robustness.", "",
        "Partial-minus-full differences have mixed signs across models. Matching selected words",
        "does **not** match character dose: partial changes2,171/2,188/2,140 characters versus",
        "4,762/4,772/4,777 for full folding. These differences cannot isolate mechanism novelty.", "",
        "## Class support and F1", "",
        "Supports are invariant across models/conditions: primary O152,628; PERIOD15,465;",
        "COMMA12,164; QUESTION_MARK1,228. Sensitivity O152,428; PERIOD15,325; COMMA12,152;",
        "QUESTION_MARK1,211. Each JSON score set retains all four classes and its4×4 confusion.", "",
        "| Input | Model | PERIOD F1 (%) | COMMA F1 (%) | QUESTION_MARK F1 (%) |",
        "| --- | --- | ---: | ---: | ---: |"]
    for group, label in zip(GROUPS, LABELS, strict=True):
        for arm, name in zip(ARMS, NAMES, strict=True):
            lines.append("| " + " | ".join([label, name, *[fmt(arm, group, metric=f"per_class.{mark}.f1")
                for mark in ("PERIOD", "COMMA", "QUESTION_MARK")]]) + " |")
    lines += ["", "## Dose, controls and audit scope", "",
        "Eligible-opportunity rates are89/4,512 elongation,50/4,174 omission and43/617",
        "operational partial folding. Exact primary edited words per seed are3,355,1,826 and",
        "1,803, respectively; the full-folding control edits the same1,803 selected words.",
        "These finite-corpus conditional anchors are transported to a different domain, not",
        "population prevalence. Uniform whole-split selection simplifies natural tweet clustering.",
        "The earlier case-neutral43/621 profile remains unchanged; four casing-incompatible",
        "opportunities and zero hits were excluded for the operational partial-folding map.", "",
        "All four fresh clean anchors exactly match their original held-out metrics in both",
        "cohorts. Maximum encoding length154; no truncation or condition-specific row drop.",
        "Sensitivity filters the same primary inputs/predictions without resampling. No unit",
        "suite was added or run. Numerical checks used saved predictions and original targets;",
        "the actual T4/FP32/SDPA-math runtime and selected tensors were verified before inference.", "",
        "## Files and identities", "",
        "- [All aggregate metrics, own-anchor drops and per-seed paired values](transfer-evaluation-aggregates.csv).",
        "- [Every condition's P/R/F1, supports, confusions and exposure](transfer-evaluation-condition-summary.json).",
        "- [Class P/R/F1/support in flat CSV](transfer-evaluation-per-class.csv).",
        "- [Input coverage](transfer-evaluation-input-coverage.csv), [protocol](transfer-evaluation-protocol.md),",
        "  [execution](transfer-evaluation-execution.md), [completion/archive](transfer-evaluation-completion.md).",
        "- [Original study and version retention](report-versions.md); original metrics are unchanged.", "",
        f"Run: `{analysis['specification']['run_identity']}`.",
        f"Analysis: `{analysis['identity']}`.", "",
        "Dataset terms, exact split provenance/document independence and prior published-model",
        "exposure remain unresolved. No p-values, equivalence claim, pooled robustness score or",
        "new training-seed uncertainty estimate is introduced.", ""]
    (docs / "transfer-evaluation-results.md").write_text("\n".join(lines))
    print("Published52-condition aggregate tables/classes and predefined contrasts; no raw text or inference.")


if __name__ == "__main__":
    main()
