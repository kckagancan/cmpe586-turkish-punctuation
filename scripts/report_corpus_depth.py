"""Publish aggregate depth-profile tables and a figure without repeating analysis."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEPTH_IDENTITY = "cde287765953c12dc78f4f56d0eca6ec81c017d21dd10626f3f9fffc7d161cca"
LENGTH_IDENTITY = "1bd0abe5b1d6542d4a5b138ae5c1d8c3f8b2fc35cf3b835e832a1f9d7229703c"
CALIBRATION_IDENTITY = "f8b903a57451c895712c7a7e71bb7061a02dbeeff123cf3efe4b58605f15039f"
NAMES = {"elongation": "Elongation", "internal_omission": "Internal omission",
         "partial_folding": "Partial folding"}
COLORS = ["#126782", "#c65d21", "#6350a2"]


def pct(value):
    return f"{value * 100:.4f}%"


def stat_table(profiles):
    lines = ["| Shape | Occurrences | /3,598 common eligible pairs | /Own opportunities | Affected tweets /570 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for f, p in profiles.items():
        lines.append(f'| {NAMES[f]} | {p["occurrences"]} | {pct(p["occurrences"] / 3598)} | '
                     f'{p["occurrences"]}/{p["opportunities"]} = {pct(p["opportunity_fraction"])} | '
                     f'{p["affected_sentences"]} = {pct(p["fraction_all_sentences"])} |')
    return "\n".join(lines)


def report(summary, length, calibration):
    profiles = summary["families"]
    lines = ["# MultiLexNorm Turkish: density and severity of three unseen shapes", "",
             "Completed **2026-11-04**. This extends the completed data-only audit; it does",
             "not repeat old reachability, run a model, read source test data, or change the original report.",
             "[Specification](informal-corpus-depth-protocol.md) · "
             "[Config](../configs/informal-corpus-depth.json) · "
             "[Complete aggregate](informal-corpus-depth-summary.json) · "
             "[Original audit](informal-corpus-audit-results.md).", "",
             "## Latest calibration scope: short words matter", "",
             "A subsequent, explicitly exploratory length-sensitivity check found that the",
             "initial length>=4 scope omitted **34/89 (38.2022%)** of the alphabetic elongation",
             "shapes. Preserve that completed scope as the comparability analysis below; use",
             "the separately versioned family-specific profile to inform the next recipe.",
             "[All length thresholds](informal-corpus-length-sensitivity.json) · "
             "[Full family-specific profile](informal-corpus-calibration-profile.json).", "",
             "| Minimum corrected length | Common alphabetic pairs | Elongation | Internal omission | Partial folding / multi-foldable pairs |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for minimum, view in length["views"].items():
        f = view["families"]
        lines.append(f'| {minimum} | {view["common_opportunities"]} | {f["elongation"]["matches"]} | '
                     f'{f["internal_omission"]["matches"]} | '
                     f'{f["partial_folding"]["matches"]}/{f["partial_folding"]["opportunities"]} |')
    lines += ["", "The broader profile uses **length>=2 for elongation, >=3 for an interior",
              "omission, and >=2 plus at least two foldable occurrences for partial folding**.",
              "These are family-specific opportunity rules, not changes to training-v1 or its",
              "frozen minimum-swap length. No one-letter elongation match was observed; an",
              "interior omission is structurally impossible below length3.", "",
              "| Shape | Matches / own opportunities | Conditional rate | Affected tweets /570 | 95% sentence-resampling stability |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for f, p in calibration["families"].items():
        lo, hi = p["sentence_cluster_bootstrap_stability"]["pooled_opportunity_rate"]["interval"]
        lines.append(f'| {NAMES[f]} | {p["occurrences"]}/{p["opportunities"]} | '
                     f'{pct(p["opportunity_fraction"])} | {p["affected_sentences"]} '
                     f'({pct(p["fraction_all_sentences"])}) | {pct(lo)}–{pct(hi)} |')
    lines += ["", "These denominators differ: the rates cannot be added or copied onto all words.",
              "On the **same 4,557 alphabetic one-to-one pairs**, the respective occurrence shares",
              "are **1.9530%, 1.0972%, 0.9436%**. The three shapes affect **182** distinct words",
              "and **131/570 (22.9825%)** tweets. They remain selected patterns, not a complete",
              "taxonomy of informal Turkish. All intervals are conditional stability summaries.", "",
              "![Broader family-specific calibration profile](figures/informal-corpus-calibration.png)", "",
              "Broader amplitude findings: elongation P25/median/P75 = **1/2/3** extra characters,",
              "P95=11, max25; **77/89** change one run and **12/89** multiple runs. There are",
              "104 changed runs: 62 final, 38 interior, 4 initial. Omission remains exactly one",
              "character: 47 unique and 3 ambiguous alignments, median missing fraction1/7.",
              "Partial folding changes one character in **40/43 (93.0233%)** words; its",
              "folded/available pairs are **1/2:30, 1/3:9, 1/4:1, 2/3:2, 4/5:1**.", "",
              "In the broader scope too, each family's opportunity-sentence P25, median and P75",
              "are zero. Positive-only density medians are **16.6667%, 12.5%, 66.6667%**",
              "respectively, with a different family-eligible denominator in each tweet. The",
              "partial-fold fraction quartiles still all equal0.5. Do not label these unrelated",
              "quantities as three natural mild/medium/heavy levels.", "",
              "This broader profile was specified after inspecting length sensitivity and before",
              "any new model scores. The original min4 profile and its uncertainty results remain",
              "unchanged. The next protocol should explicitly select the wider family rules,",
              "retain the whole observed amplitude distribution, and disclose any controlled",
              "stress multipliers or simplification of placement/clustering.", "",
              "## Initial length>=4 depth analysis: source and denominators", "",
             "Pinned MultiLexNorm Turkish **train**, 570 sentences / 6,443 source slots.",
             "The strict opportunity pool contains **3,598** one-to-one alphabetic pairs,",
             "corrected length >=4, with unchanged eligible pairs retained. Disjoint exclusions:",
             "354 structural slots, 1,583 remaining slots shorter than four corrected characters,",
             "908 remaining nonalphabetic pairs. The latter categories follow that precedence.",
             "Partial folding has **617** opportunities with at least two foldable occurrences.",
             "The original 6,089 one-to-one slots include punctuation; none of these denominators",
             "is automatically the punctuation-free all-word count in our original experiment.", "",
             "Rates are conditional on this selected corpus, annotation scheme and eligibility;",
             "they are not population frequencies of informal Turkish. Shapes use Turkish-aware",
             "lowercase comparison, not annotator-provided error categories or judgments of intent.", "",
             stat_table(profiles), "",
             "The larger partial-fold opportunity rate reflects its narrower eligibility. Use the",
             "common-pool column for common-denominator comparisons. The three shapes cover",
             f'**{summary["joint"]["word_level_union"]}** distinct words and '
             f'**{summary["joint"]["affected_sentences_any_new_shape"]}/570 '
             f'({pct(summary["joint"]["affected_sentences_any_new_shape"] / 570)})** sentences.',
             "There is no same-word overlap between these three length-defined shapes.", "",
             "## Length>=4: per-word amplitude is different from word exposure", "",
             "![Corpus opportunity rates, message presence and per-word amplitude](figures/informal-corpus-depth.png)", ""]
    e = profiles["elongation"]["amplitude"]
    p = profiles["partial_folding"]["amplitude"]
    lines += ["### Elongation", "",
              f'Added-character histogram: `{json.dumps(e["extra_characters"]["histogram"], ensure_ascii=False)}`.',
              f'Median **2**, P25 **1**, P75 **3**, P95 **12**, maximum **25**; mean '
              f'**{e["extra_characters"]["mean"]:.4f}** added characters. The five largest words account for '
              f'**{pct(e["top_five_word_share_of_extra_characters"])}** of all 168 added characters.',
              "A mean-only or a fixed +1-character recipe would hide this tail.", "",
              "**48/55 (87.2727%)** cases change one run, **5** change two and **2** change three.",
              "Across 64 changed runs: **32 final, 31 interior, 1 initial**; **32/64** repeat vowels.",
              "Do not confuse run counts with word counts. A one-run-only generator would represent",
              "48 observed cases and should use **48/3,598**, not the full 55-case rate.", "",
              "### One internal omission", "",
              "All **49** detected words omit **exactly one** interior character by definition.",
              "Corrected word-length median **7**, range **4–13**; the missing fraction has P25",
              "**11.1111%**, median **14.2857%**, P75 **16.6667%** and maximum **25%**.",
              "These describe relative word damage, not evidence for two/three deletions per word.",
              "There are **46** unique positional alignments and **3** ambiguous repeated-character",
              "alignments. Position statistics use only the 46 unique cases; median relative index",
              "is **0.6** on index/(length−1). **28/49** omitted letters are vowels; most frequent",
              "letters are `r`13, `a`11, `e`7. These are surface observations, not inferred causes.", "",
              "### Partial Turkish folding", "",
              f'Folded/available occurrence histogram: `{json.dumps(p["folded_available_joint"])}`.',
              "**39/42 (92.8571%)** words fold one character; two words fold two, one folds four.",
              "**29/42** are exactly 1-of-2. The folded fraction has **P25 = median = P75 = 50%**,",
              "range **25–80%**. In total **47/100** available character occurrences in these",
              "42 affected words are folded; this is a character-weighted ratio, not the",
              f'mean per-word fraction ({pct(p["folded_fraction"]["mean"])}).', "",
              "Within all 617 multi-foldable opportunities: **270** full folds, **42** partial folds,",
              "**230** case-neutral unchanged and **75** other changes. Thus partial folds are",
              "**42/312 (13.4615%)** of these literal full-or-partial folding patterns.", "",
              "## Tweet density, zeros and concentration", "",
              "Rates below use each sentence's family-eligible word count. Positive-only quantiles",
              "condition on the family occurring; they must not be applied to every sentence.", "",
              "| Shape | Opportunity-bearing tweets | Zero-hit tweets among them | Positive-only rate P25 / median / P75 | Max hits in one tweet | Top-five tweet share of all hits |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for f, a in profiles.items():
        q = a["affected_sentence_rates"]["quantiles"]
        lines.append(f'| {NAMES[f]} | {a["sentences_with_opportunities"]} | '
                     f'{a["zero_hit_sentences_with_opportunities"]} | '
                     f'{pct(q["p25"])} / {pct(q["p50"])} / {pct(q["p75"])} | '
                     f'{a["hits_per_sentence"]["maximum"]:g} | {pct(a["top_five_sentence_share_of_hits"])} |')
    lines += ["", "For **every family**, all-opportunity-sentence P25, median, P75 and even P90",
              "are **zero**. It would be misleading to manufacture three positive mild/medium/heavy",
              "doses from those quantiles. Positive-only partial-fold density can be high because",
              "there may be only one or two eligible words in the affected tweet.", "",
              "Sentence memberships: elongation only38, omission only36, partial folding only33,",
              "elongation+omission4, elongation+partial folding1, none458. No omission+partial",
              "folding or all-three sentence occurs in this selected sample. Among the 112 new-shape",
              "sentences, **63 (56.25%)** also contain an exact old-operator-reachable pair elsewhere",
              "(**48** using case-neutral reachability). This is co-occurrence, not evidence of",
              "actual historical sampled training exposure.", "",
              "## Resampling stability, duplication and scope", "",
              "5,000 whole-sentence bootstrap draws, PCG64 seed586; 95% percentile stability",
              "intervals. Every family has 5,000 defined replicates and zero undefined replicates.",
              "These do not establish population confidence, training uncertainty or significance.", "",
              "| Shape | Pooled opportunity rate | Sentence-bootstrap stability interval | Neutral unique-pair matches / opportunities | Unique-pair rate |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for f, a in profiles.items():
        lo, hi = a["sentence_cluster_bootstrap_stability"]["pooled_opportunity_rate"]["interval"]
        u = a["unique_pair_sensitivity"]["case_neutral"]
        lines.append(f'| {NAMES[f]} | {pct(a["opportunity_fraction"])} | {pct(lo)}–{pct(hi)} | '
                     f'{u["matches"]}/{u["opportunities"]} | {pct(u["fraction"])} |')
    lines += ["", "Unique pairs are a reweighting, not independent observations. Complete aggregate",
              "JSON also contains exact-pair sensitivity, length strata, leave-one-sentence-out",
              "ranges, letter/position histograms and exact versus case-neutral edit distances.", "",
              "The three shapes cover **146/732 (19.9454%)** of the previous audit's case-neutral",
              "unreachable one-word changes. The restricted 3,598-pair pool separately contains",
              "2,343 unchanged, 786 old-reachable, 146 new-shape and 323 other-unreachable pairs.",
              "These selected mechanisms do not exhaust informal Turkish or all uncovered errors.", "",
              "## Consequences for the next evaluation", "",
              "1. The primary extension should measure the three new mechanisms separately, with",
              "   fixed existing models and a fresh clean anchor. Preserve the completed old grid;",
              "   do not rerun it merely to add the extension.",
              "2. Specify an empirically anchored condition per family using the stated opportunity",
              "   denominator and a declared distribution of amplitude/placement. Decide explicitly",
              "   whether to retain multi-run elongation and sentence clustering; simplifying them",
              "   changes what the rate means. Do not copy these percentages onto all word slots.",
              "   Keep folded/available joint support valid; unobserved word/run shapes need a",
              "   declared adaptation rule, not independent sampling of incompatible marginals.",
              "3. Include a small **seen-mechanism control**, particularly full folding on the same",
              "   selected multi-foldable words as partial folding. It matches word exposure while",
              "   changing character dose. Historical case/swap/combined results remain references.",
              "   This matched control has a new input identity; old full-fold predictions cannot",
              "   be reused as if they were generated on this newly selected pool.",
              "4. Do not force three natural severity levels. Extra-character intensity supports",
              "   lower/middle/upper descriptive bins of 21/24/10 words (thresholds1,3); omission",
              "   always removes one character; partial-fold fraction quartiles collapse at0.5",
              "   and give 39/0/3 words. Additional dose multipliers or mixed-noise conditions must",
              "   be explicitly labeled controlled stress scenarios, chosen before model scores.",
              "5. A mixed seen+new condition could address co-occurrence, but uniform addition of",
              "   old 5% per-family noise would not reproduce the observed joint distribution.",
              "   Freeze any such secondary condition separately rather than silently enlarging",
              "   the matrix or asserting natural realism.", "",
              "No evaluation recipe is frozen by these findings; no corrupted model input or score",
              "has been generated. Existing 2/5/10% recipes and final report PDFs are unchanged.",
              "New corruptions on old test text are supplemental controlled transfer, with shared",
              "sentences/targets. Licensing/exact provenance, model historical exposure and",
              "population generalization remain unresolved. This corpus supplies no punctuation gold.", "",
              "## Reproducibility", "",
              f'Depth identity: `{summary["identity"]}`.',
              f'Original audit: `{summary["original_audit_identity"]}`.',
              f'Length-sensitivity identity: `{length["identity"]}`.',
              f'Broader calibration identity: `{calibration["identity"]}`.',
              "Private immutable run: `artifacts/corpus-audit/multilexnorm-depth-v1/depth-" +
              summary["identity"][:16] + "/`.",
              "The public aggregate is byte-identical to its saved summary. Private slot/sentence",
              "features contain IDs/hashes/measurements, no full words or tweets. The original",
              "audit manifest and source bytes were verified; no old reachability was rerun.",
              "See the local verification/publication receipts for actual conservation, source",
              "hash and figure review outcomes. No unit suite or model was run.", ""]
    return "\n".join(lines)


def figure(summary, path, scope_label="corrected length >=4"):
    profiles = summary["families"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), layout="constrained")
    labels = [f'{name}\n(n={p["opportunities"]:,})' for name, p in zip(
        ["Elongation", "Omission", "Partial fold"], profiles.values(), strict=True)]
    rates = np.array([p["opportunity_fraction"] * 100 for p in profiles.values()])
    intervals = np.array([p["sentence_cluster_bootstrap_stability"]["pooled_opportunity_rate"]
                          ["interval"] for p in profiles.values()]) * 100
    axes[0, 0].bar(range(3), rates, color=COLORS)
    axes[0, 0].errorbar(range(3), rates, yerr=[rates - intervals[:, 0], intervals[:, 1] - rates],
                       fmt="none", color="#263238", capsize=5)
    axes[0, 0].set_xticks(range(3), labels)
    axes[0, 0].set_ylabel("Matches / own opportunities (%)")
    axes[0, 0].set_title("A. Conditional word rates\n95% sentence-resampling stability")
    presences = [p["fraction_all_sentences"] * 100 for p in profiles.values()]
    axes[0, 1].bar(range(3), presences, color=COLORS)
    axes[0, 1].set_xticks(range(3), ["Elongation", "Omission", "Partial fold"])
    axes[0, 1].set_ylabel("Affected tweets / 570 (%)")
    axes[0, 1].set_title("B. Message prevalence")
    for i, v in enumerate(presences):
        axes[0, 1].text(i, v + .12, f"{v:.2f}%", ha="center")
    axes[0, 1].set_ylim(0, max(presences) * 1.22)
    positive = [p["affected_sentence_rates"]["quantiles"] for p in profiles.values()]
    medians = np.array([q["p50"] for q in positive]) * 100
    lower = np.array([q["p25"] for q in positive]) * 100
    upper = np.array([q["p75"] for q in positive]) * 100
    axes[0, 2].errorbar(range(3), medians, yerr=[medians - lower, upper - medians], fmt="o",
                       color="#263238", capsize=6, markersize=7)
    axes[0, 2].set_xticks(range(3), ["Elongation", "Omission", "Partial fold"])
    axes[0, 2].set_xlim(-.5, 2.5)
    axes[0, 2].set_ylabel("Within-message opportunity rate (%)")
    axes[0, 2].set_title("C. Affected tweets only\nMedian and P25/P75; excludes zero-hit tweets")
    histogram_specs = [
        (profiles["elongation"]["amplitude"]["extra_characters"]["histogram"],
         f'D. Elongation amplitude ({profiles["elongation"]["occurrences"]} words)', "Added characters", COLORS[0]),
        (profiles["internal_omission"]["amplitude"]["omitted_letters"],
         f'E. Internal omission ({profiles["internal_omission"]["occurrences"]} words)', "Omitted letter; ambiguous positions retained", COLORS[1]),
        (profiles["partial_folding"]["amplitude"]["folded_available_joint"],
         f'F. Partial folding ({profiles["partial_folding"]["occurrences"]} words)', "Folded / available character occurrences", COLORS[2]),
    ]
    for axis, (hist, title, xlabel, color) in zip(axes[1], histogram_specs, strict=True):
        axis.bar(range(len(hist)), list(hist.values()), color=color)
        axis.set_xticks(range(len(hist)), list(hist))
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Word occurrences")
        for i, value in enumerate(hist.values()):
            axis.text(i, value + .25, str(value), ha="center", fontsize=9)
        axis.set_ylim(0, max(hist.values()) * 1.18)
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=.16)
        axis.set_axisbelow(True)
    fig.suptitle("MultiLexNorm Turkish train: " + scope_label + "\n"
                 "570 tweets; conditional corpus descriptors, not population rates or model scores",
                 fontsize=15)
    fig.savefig(path, dpi=180, metadata={"Title": "MultiLexNorm Turkish depth profile",
                                      "Description": summary["identity"]})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--length-run", type=Path, required=True)
    parser.add_argument("--calibration-run", type=Path, required=True)
    args = parser.parse_args()
    run = (ROOT / args.run).resolve()
    manifest = json.loads((run / "commit.json").read_text())
    for name, expected in manifest["files"].items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Immutable depth manifest mismatch")
    summary = json.loads((run / "summary.json").read_text())
    if (summary["identity"] != manifest["identity"] or summary["identity"] != DEPTH_IDENTITY
            or summary["version"] != "informal-corpus-depth-v1"):
        raise ValueError("Unexpected profile identity/version")
    additional = []
    for value, expected in [(args.length_run, LENGTH_IDENTITY),
                            (args.calibration_run, CALIBRATION_IDENTITY)]:
        parent = ROOT / value
        committed = json.loads((parent / "commit.json").read_text())
        for name, expected_hash in committed["files"].items():
            target = (parent / name).resolve()
            if not target.is_relative_to(parent.resolve()) or hashlib.sha256(target.read_bytes()).hexdigest() != expected_hash:
                raise ValueError("Supplement profile manifest mismatch")
        aggregate = json.loads((parent / "summary.json").read_text())
        if committed["identity"] != expected or aggregate["identity"] != expected:
            raise ValueError("Supplement profile identity mismatch")
        additional.append((parent, aggregate))
    (length_run, length), (calibration_run, calibration) = additional
    summary_path = ROOT / "docs/informal-corpus-depth-summary.json"
    report_path = ROOT / "docs/informal-corpus-depth-results.md"
    figure_path = ROOT / "docs/figures/informal-corpus-depth.png"
    length_path = ROOT / "docs/informal-corpus-length-sensitivity.json"
    calibration_path = ROOT / "docs/informal-corpus-calibration-profile.json"
    calibration_figure = ROOT / "docs/figures/informal-corpus-calibration.png"
    published = [summary_path, report_path, figure_path, length_path, calibration_path, calibration_figure]
    if any(path.exists() for path in published):
        raise FileExistsError("Existing published depth outputs are preserved")
    figure(summary, figure_path)
    figure(calibration, calibration_figure, "family-specific minima 2 / 3 / 2")
    report_path.write_text(report(summary, length, calibration), encoding="utf-8")
    shutil.copyfile(run / "summary.json", summary_path)
    shutil.copyfile(length_run / "summary.json", length_path)
    shutil.copyfile(calibration_run / "summary.json", calibration_path)
    print(json.dumps({"published": [str(p.relative_to(ROOT)) for p in published],
                      "original_analysis_reexecuted": False}, indent=2))


if __name__ == "__main__":
    main()
