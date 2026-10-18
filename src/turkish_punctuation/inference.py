"""Load the published checkpoint and inspect word-level punctuation predictions."""

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from turkish_punctuation.preprocessing import encode_records, make_dataloader
from turkish_punctuation.tokenization import IGNORE_INDEX, LABEL_TO_ID


def load_published_model(
    config: dict, *, root: Path, allow_download: bool = False
) -> PreTrainedModel:
    """Load the pinned four-class checkpoint; reject an incomplete weight load.

    The released revision contains pytorch_model.bin. Use PyTorch's restricted
    weights-only loader and disable automatic safetensors conversion/network lookup.
    """
    source = {
        "revision": config["revision"],
        "cache_dir": str(root / config["cache_directory"]),
        "local_files_only": not allow_download,
        "trust_remote_code": False,
        "token": False,
    }
    model_config = AutoConfig.from_pretrained(config["model_id"], **source)
    if config["label_to_id"] != LABEL_TO_ID or model_config.label2id != LABEL_TO_ID:
        raise ValueError("Checkpoint label mapping differs from the project's targets.")
    if model_config.id2label != {value: key for key, value in LABEL_TO_ID.items()}:
        raise ValueError("Checkpoint id2label mapping differs from the project's targets.")
    if model_config.max_position_embeddings != config["max_input_length"]:
        raise ValueError("Checkpoint position limit differs from the tokenizer config.")
    model, loading = AutoModelForTokenClassification.from_pretrained(
        config["model_id"],
        config=model_config,
        use_safetensors=False,
        weights_only=True,
        output_loading_info=True,
        **source,
    )
    if not isinstance(model, PreTrainedModel):
        raise TypeError("Expected a Transformers pretrained model.")
    if any(
        loading.get(key)
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError(f"Checkpoint did not load exactly: {loading}")
    model.eval()
    return model


def iter_predictions(
    encoded: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    *,
    max_length: int,
    batch_size: int = 8,
    device: str = "cpu",
    include_logits: bool = False,
) -> Iterator[dict]:
    """Yield word-level class IDs from prepared records in source order.

    Labels locate first-subword positions; their class values are never passed to
    the model. Optional logits support small inspections without retaining scores
    for an entire evaluation corpus. Yield outside inference_mode to restore the
    caller's gradient state between records.
    """
    if model.config.label2id != LABEL_TO_ID or model.config.id2label != {
        value: key for key, value in LABEL_TO_ID.items()
    }:
        raise ValueError("Prediction model label mapping differs from the project's targets.")
    if max_length > model.config.max_position_embeddings:
        raise ValueError("Requested max_length exceeds the model's position limit.")
    loader = make_dataloader(encoded, tokenizer, batch_size=batch_size, shuffle=False)
    # Use the nn.Module type: Transformers' decorated .to() confuses static typing.
    cast(torch.nn.Module, model).to(device=torch.device(device))
    model.eval()
    row_ids = list(encoded["row_id"])
    cursor = 0
    for batch in loader:
        if batch["input_ids"].shape[1] > max_length:
            raise ValueError("Prepared input exceeds max_length; no truncation applied.")
        with torch.inference_mode():
            word_positions = batch["labels"] != IGNORE_INDEX
            inputs = {key: value.to(device) for key, value in batch.items() if key != "labels"}
            logits = model(**inputs).logits.cpu()
            if logits.shape[:2] != word_positions.shape or logits.shape[-1] != len(LABEL_TO_ID):
                raise RuntimeError("Model output dimensions do not match the aligned inputs.")
            if not torch.isfinite(logits).all().item():
                raise RuntimeError("Model output contains non-finite logits.")
            batch_results = []
            for row_logits, targets, positions in zip(
                logits, batch["labels"], word_positions, strict=True
            ):
                word_logits = row_logits[positions]
                result = {
                    "row_id": row_ids[cursor],
                    "reference_ids": targets[positions].tolist(),
                    "prediction_ids": word_logits.argmax(-1).tolist(),
                }
                if include_logits:
                    result["word_logits"] = word_logits.tolist()
                batch_results.append(result)
                cursor += 1
        yield from batch_results
    if cursor != len(encoded):
        raise RuntimeError("Prediction count differs from the input record count.")


def predict_records(
    records: pd.DataFrame,
    tokenizer: PreTrainedTokenizerFast,
    model: PreTrainedModel,
    *,
    max_length: int,
    batch_size: int = 8,
    device: str = "cpu",
) -> list[dict]:
    """Attach words and reference tags to small-example predictions and logits."""
    encoded = encode_records(records, tokenizer, max_length=max_length)
    predictions = iter_predictions(
        encoded,
        tokenizer,
        model,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
        include_logits=True,
    )
    id_to_label = {value: key for key, value in LABEL_TO_ID.items()}
    results = []
    columns = records[["row_id", "tokens", "tags"]]
    for (row_id, tokens, tags), prediction in zip(
        columns.itertuples(index=False, name=None), predictions, strict=True
    ):
        if prediction["row_id"] != row_id or len(prediction["prediction_ids"]) != len(tokens):
            raise RuntimeError(f"Row {row_id}: prediction count differs from word count.")
        results.append(
            {
                "row_id": row_id,
                "words": tokens,
                "reference_tags": tags,
                "predicted_tags": [id_to_label[value] for value in prediction["prediction_ids"]],
                "word_logits": prediction["word_logits"],
            }
        )
    return results
