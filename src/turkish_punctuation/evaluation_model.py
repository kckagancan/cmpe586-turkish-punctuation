"""Verify a completed fine-tuning export before using it for validation inference."""

import json
from pathlib import Path
from typing import cast

import torch
from transformers import AutoModelForTokenClassification, PreTrainedModel

from turkish_punctuation.tokenization import LABEL_TO_ID
from turkish_punctuation.training_io import file_hashes, model_digest, sha256, value_digest


def training_reference(root: Path, settings: dict) -> dict:
    """Pin the selected clean-validation epoch by run, file and actual tensor hashes.

    Historical training metrics remain a separate reference. Fresh clean/noisy
    inference will use one matched runtime and batching policy.
    """
    directory = (root / settings["run_directory"]).resolve()
    run_path, summary_path = directory / "run.json", directory / "summary.json"
    run = json.loads(run_path.read_text())
    summary = json.loads(summary_path.read_text())
    progress = json.loads((directory / "progress.json").read_text())
    specification = run["specification"]
    identity = value_digest(specification)
    if not identity == run["identity"] == summary["identity"] == settings["run_identity"]:
        raise ValueError("Training identity differs from the pinned completed run.")
    history = summary["history"]
    epochs = specification["configuration"]["optimization"]["epochs"]
    if (
        specification["mode"] != "train"
        or specification["arm"] != summary["arm"]
        or specification["arm"] != settings["arm"]
        or progress["status"] != "complete"
        or progress["global_step"] != summary["global_step"]
        or summary["global_step"] != run["budget"]["total_updates"]
        or [item["epoch"] for item in history] != list(range(1, epochs + 1))
    ):
        raise ValueError("Expected a completed formal fine-tuning run with its declared budget.")
    selected = min(
        history,
        key=lambda item: (-item["validation"]["metrics"]["punctuation_macro"]["f1"], item["epoch"]),
    )
    epoch = selected["epoch"]
    if (
        summary["best_epoch"] != epoch
        or summary["best_model"] != f"models/epoch-{epoch:02d}"
        or summary["best_macro_f1"] != selected["validation"]["metrics"]["punctuation_macro"]["f1"]
    ):
        raise ValueError("Selected export differs from the declared clean-validation selection.")
    model_directory = directory / summary["best_model"]
    export_path = model_directory / "export.json"
    export = json.loads(export_path.read_text())
    if (
        export != selected["model_export"]
        or export["identity"] != identity
        or export["model_tensors_sha256"] != settings["model_tensors_sha256"]
        or file_hashes(model_directory, exclude=("export.json",)) != export["files"]
    ):
        raise ValueError("Selected model files differ from the pinned export.")
    report_path = directory / "validation" / f"epoch-{epoch:02d}" / "metrics.json"
    prediction_path = report_path.with_name("predictions.jsonl")
    report = json.loads(report_path.read_text())
    if (
        report != selected["validation"]
        or report["identity"] != identity
        or report["model_tensors_sha256"] != export["model_tensors_sha256"]
        or sha256(report_path) != selected["validation_report_sha256"]
        or sha256(prediction_path) != report["predictions_sha256"]
    ):
        raise ValueError("Selected epoch's original validation reference changed.")
    source_path = directory / "source_snapshot/tokenizer-config.json"
    if sha256(source_path) != specification["source_hashes"]["tokenizer-config.json"]:
        raise ValueError("Archived tokenizer provenance changed.")
    source = json.loads(source_path.read_text())
    metadata = {
        "kind": "fine_tuned_export",
        "training_identity": identity,
        "training_arm": specification["arm"],
        "training_seed": specification["seed"],
        "selected_epoch": epoch,
        "selection": specification["configuration"]["selection"],
        "model_tensors_sha256": export["model_tensors_sha256"],
        "base_model_id": source["base_model_id"],
        "base_model_revision": source["base_model_revision"],
        "tokenizer_source": source,
        "training_environment": specification["environment"],
        "source_sha256": {
            "run.json": sha256(run_path),
            "summary.json": sha256(summary_path),
            "export.json": sha256(export_path),
            "validation_metrics.json": sha256(report_path),
            "validation_predictions.jsonl": sha256(prediction_path),
            "tokenizer-config.json": sha256(source_path),
        },
        "export_file_sha256": export["files"],
    }
    return {
        "model_directory": model_directory,
        "report_path": report_path,
        "prediction_path": prediction_path,
        "report": report,
        "tokenizer_config": source,
        "validation_source_sha256": specification["inputs"]["validation"]["raw_sha256"],
        "metadata": metadata,
    }


def load_exported_model(reference: dict, tokenizer, *, max_length: int) -> PreTrainedModel:
    """Load only the verified local safetensors export, with no training updates."""
    model, loading = AutoModelForTokenClassification.from_pretrained(
        str(reference["model_directory"]),
        local_files_only=True,
        trust_remote_code=False,
        token=False,
        use_safetensors=True,
        output_loading_info=True,
        torch_dtype=torch.float32,
    )
    if not isinstance(model, PreTrainedModel) or any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Exported model did not load exactly: {loading}")
    if (
        model.config.model_type != "bert"
        or model.config.label2id != LABEL_TO_ID
        or model.config.id2label != {value: key for key, value in LABEL_TO_ID.items()}
        or model.config.max_position_embeddings != max_length
        or model.config.vocab_size != len(tokenizer)
    ):
        raise ValueError("Exported model architecture/labels/vocabulary/position limit differ.")
    module = cast(torch.nn.Module, model)
    if model_digest(module) != reference["metadata"]["model_tensors_sha256"]:
        raise ValueError("Loaded model tensors differ from the selected training export.")
    if not all(torch.isfinite(value).all().item() for value in module.state_dict().values()):
        raise ValueError("Exported model has non-finite tensors.")
    model.eval()
    return model
