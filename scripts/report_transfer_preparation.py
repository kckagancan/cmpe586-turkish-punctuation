"""Publish aggregate input evidence after saved-view verification, without model scores."""

import argparse
import csv
import io
import json
from pathlib import Path

from turkish_punctuation.heldout import verify_commit
from turkish_punctuation.training_io import sha256, value_digest
from turkish_punctuation.transfer_preparation import load_frozen, read

ROOT = Path(__file__).resolve().parents[1]


def publish(path, payload):
    if path.exists() and path.read_text() != payload:
        raise FileExistsError(f"Existing aggregate differs; preserve/version it: {path}")
    path.write_text(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-receipt", type=Path, required=True)
    args = parser.parse_args()
    config, noise, _, freeze, _ = load_frozen(ROOT)
    directory = ROOT / config["paths"]["prepared_directory"]
    marker = verify_commit(directory)
    manifest = read(directory / "manifest.json")
    verification = read(args.verification_receipt)
    result = verification["result"]
    if (verification["prepared_complete_sha256"] != sha256(directory / "complete.json")
            or result["status"] != "saved_inputs_reconciled_not_gpu_ready"
            or result["identity"] != marker["identity"]
            or marker["identity"] != value_digest(manifest["specification"])
            or result["freeze_identity"] != freeze["identity"]
            or not result["retokenization_performed"] or result["model_evaluation_performed"]):
        raise ValueError("Require a matching saved-text/target/morphology/fresh-tokenization verification receipt")
    # Do not publish source row IDs, words, edit examples or model predictions.
    aggregate = {
        "version": "transfer-input-audit-v1", "specified_on": config["specified_on"],
        "status": "inputs_prepared_and_reconciled_not_gpu_ready", "freeze_identity": freeze["identity"],
        "input_identity": marker["identity"], "profile_sha256": config["profile"]["sha256"],
        "verification_receipt_sha256": sha256(args.verification_receipt),
        "run_matrix": config["run_matrix"], "rates": noise.profile["rates"],
        "partial_operational_scope": noise.profile["partial_operational_scope"],
        "clean_encoding_reused_by_hash_reference": True, "old_noise_views_regenerated": False,
        "conditions": manifest["conditions"], "max_input_length": manifest["max_input_length"],
        "preparation_python": manifest["specification"]["python"],
        "preparation_packages": manifest["specification"]["packages"],
        "saved_inputs_retokenized_and_compared": True,
        "models_loaded": False, "new_model_scores": False, "training_performed": False,
        "gpu_readiness_checked": False, "raw_text_or_source_row_ids_published": False,
    }
    publish(ROOT / "docs/transfer-evaluation-input-audit.json", json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n")
    output = io.StringIO()
    fields = ["condition", "cohort", "rows", "word_targets", "eligible_words", "changed_words",
              "changed_rows", "changed_all_words_percent", "changed_eligible_words_percent", "character_edits",
              "added_characters", "removed_characters", "max_subword_length", "unknown_subwords",
              "elongation_conditioned_words", "partial_extrapolated_words"]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    lines = []
    for name, audit in manifest["conditions"].items():
        token = audit["tokenization"]
        for cohort, key in [("primary", "changes"), ("sensitivity", "sensitivity_changes")]:
            changes = audit[key]
            writer.writerow(dict(zip(fields, [name, cohort, changes["rows"], changes["words"],
                changes["eligible_words"], changes["changed_words"], changes["changed_rows"],
                100 * changes["changed_word_fraction_all"],
                100 * changes["changed_fraction_of_opportunities"] if changes["eligible_words"] else "",
                changes["declared_character_edit_count"], changes["added_characters"], changes["removed_characters"],
                token["length_including_special_tokens"]["max"] if cohort == "primary" else "",
                token["unknown_subwords"] if cohort == "primary" else "",
                changes["elongation_feasible_template_conditioning_words"],
                changes["partial_unseen_foldable_count_words"]], strict=True)))
        c = audit["changes"]
        lines.append(f"| {name} | {c['changed_words']:,} | {c['changed_rows']:,} | "
                     f"{c['declared_character_edit_count']:,} | {token['length_including_special_tokens']['max']} |")
    publish(ROOT / "docs/transfer-evaluation-input-coverage.csv", output.getvalue())
    body = """# Supplemental transfer inputs: preparation outcome

Prepared **2026-11-04** under the [fixed protocol](transfer-evaluation-protocol.md).
All **13 input views** are audited: one existing clean encoding reused through
its exact read-only hash reference, plus12 new noisy views. **No model weights
were loaded, no F1 was computed, and no GPU readiness was established.**
The later fixed-model evaluation has52 forwards /104 cohort score sets.
The original experiments, corpus analyses and public final PDF are preserved.

## Scientific scope and corpus anchors

The three separate mechanisms are elongation, one internal character omission,
and partial Turkish folding. A fourth condition applies the original full-fold
operator to exactly the same words as partial folding. Three corruption seeds
(586/587/588) retain one empirical anchor per family; there is no new severity
search, combined condition, checkpoint selection or training.

The completed MultiLexNorm Turkish **train-only** findings supply conditional
rates89/4512,50/4174,43/621. The new surface-preserving partial-fold implementation
excludes four casing-incompatible source opportunities and no hits, so its
separate operational anchor is43/617. This does not replace the earlier43/621
analysis, nor denote the initial min4 pool which also happened to have617 items.
The compiled profile contains anonymous aggregate morphology templates rather
than source words. Donor conditioning, position mapping and unsupported target
foldable-count extrapolation are explicit in the protocol and aggregate audit.

Uniform whole-split word sampling transports these finite-corpus rates; it
does not reproduce natural message clustering. Partial/full folding has matched
word exposure but unequal character dose. New corruptions reuse the previously
scored test text: this is supplemental controlled transfer, not an independent
natural-text test or a blind preregistration. The natural source test remains
unread; no natural punctuation gold is claimed.

## Realized primary input coverage

Each view retains14,600 rows /181,485 original word targets. The sensitivity
view filters the same primary records/edits to14,445 rows /181,116 targets;
it does not rerank positions or calculate a new corruption budget.

| Input view | Changed words | Changed rows | Character edits | Max sequence length |
| --- | ---: | ---: | ---: | ---: |
""" + "\n".join(lines) + "\n\n"
    body += f"""Character edits count declared insertions, deletions or folded occurrences;
the full [aggregate JSON](transfer-evaluation-input-audit.json) records amplitude,
donor, placement, conditioning/extrapolation and label-exposure histograms.
The [coverage CSV](transfer-evaluation-input-coverage.csv) includes both cohorts.
Sequence lengths/unknown counts in that CSV are for primary encodings only.

## Audits actually performed

1. Data-only validation probe: source prefix512 rows,510 eligible under the
   unchanged whole-row quarantine;13 views, no model metrics. All first-subword
   targets were preserved; maximum93 tokens. No recipe was chosen from F1.
2. Exact code/config/profile/model-metadata/cohort freeze **before** generating
   new test views. Freeze identity: `{freeze['identity']}`.
3. Full-source audit: unchanged row IDs/order, word counts, tags, nonempty NFC
   output, unselected-word protection, exact floor budgets and SHA-ranked masks.
   Separate insertion/deletion/folding shape checks preserve the old equal-length
   noise auditor. Partial and full-fold controls share identical selected masks.
4. Every new view was retokenized with the pinned cached fast tokenizer and
   unchanged alignment. No missing word, truncation, condition-specific drop or
   sequence over512 was accepted. Maximum realized length: **{manifest['max_input_length']}**.
5. Completed files were read back: hashes/file membership, raw targets, edit
   semantics, deterministic donor parameters and independently reranked masks.
   Fresh tokenization of all saved noisy views and the reused clean reference
   matched every saved column/label and audit. No test suite was added or run.

Input identity: `{marker['identity']}`. Private artifacts are under
`data/processed/transfer-noise-v1/` and
`artifacts/evaluation-preparation/transfer-noise-v1/`. Source snapshots and
complete-file manifests bind the inputs. The freeze's historical status string
records its pre-generation stage; this report records completed preparation.

## Remaining action

Implement the separate orchestration/runtime gate, verify retained model files
and tensors, and perform validation-only GPU calibration on the pinned T4
runtime before the52-forward evaluation. Scoring remains disabled by default.
Report per-class support/precision/recall/F1 and paired contrasts for all outcomes;
seed SD measures corruption variation only, with one training seed unchanged.
No job is running after this preparation. No new result has been folded into
the original final paper or PDF.
"""
    publish(ROOT / "docs/transfer-evaluation-preparation.md", body)
    print(json.dumps({"status": aggregate["status"], "input_identity": marker["identity"],
                      "max_input_length": manifest["max_input_length"], "new_model_scores": False}, indent=2))


if __name__ == "__main__":
    main()
