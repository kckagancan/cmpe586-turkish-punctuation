"""Inspect the published BERT on invented examples; not a benchmark or training run."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from turkish_punctuation.inference import load_published_model, predict_records
from turkish_punctuation.tokenization import LABEL_TO_ID

ROOT = Path(__file__).resolve().parents[1]
PUNCTUATION = {"O": "", "PERIOD": ".", "COMMA": ",", "QUESTION_MARK": "?"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-model", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/tokenizer.json").read_text())
    # These are authored examples, not selected validation/test records.
    records = pd.DataFrame(
        [
            {
                "row_id": 0,
                "tokens": ["Bugün", "hava", "çok", "güzel"],
                "tags": ["O", "O", "O", "PERIOD"],
            },
            {
                "row_id": 1,
                "tokens": ["Yarın", "bizimle", "gelecek", "misin"],
                "tags": ["O", "O", "O", "QUESTION_MARK"],
            },
            {
                "row_id": 2,
                "tokens": ["Evet", "yarın", "sizinle", "geleceğim"],
                "tags": ["COMMA", "O", "O", "PERIOD"],
            },
        ]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["revision"],
        cache_dir=str(ROOT / config["cache_directory"]),
        local_files_only=not args.download_model,
        use_fast=True,
        trust_remote_code=False,
        token=False,
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError("A fast tokenizer with word_ids() is required.")
    model = load_published_model(config, root=ROOT, allow_download=args.download_model)
    results = predict_records(
        records,
        tokenizer,
        model,
        max_length=config["max_input_length"],
        batch_size=3,
        device="cpu",
    )
    for result in results:
        result["restored_text"] = " ".join(
            word + PUNCTUATION[tag]
            for word, tag in zip(result["words"], result["predicted_tags"], strict=True)
        )
        print(result["restored_text"])
    report = {
        "purpose": "Invented examples for understanding inference; not a quality estimate.",
        "model_id": config["model_id"],
        "revision": config["revision"],
        "architecture": model.__class__.__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "hidden_size": model.config.hidden_size,
        "encoder_layers": model.config.num_hidden_layers,
        "attention_heads_per_layer": model.config.num_attention_heads,
        "num_labels": model.config.num_labels,
        "logit_class_order": list(LABEL_TO_ID),
        "device": "cpu",
        "model_training_flag": model.training,
        "torch_version": torch.__version__,
        "transformers_version": version("transformers"),
        "training_performed": False,
        "benchmark_evaluation_performed": False,
        "examples": results,
    }
    destination = ROOT / "artifacts/model-preview.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved model preview: {destination}")


if __name__ == "__main__":
    main()
