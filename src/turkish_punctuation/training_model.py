"""Pinned BERTurk initialization with a fresh head shared by compared training arms."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BertForTokenClassification,
    PreTrainedTokenizerFast,
)

from turkish_punctuation.tokenization import LABEL_TO_ID
from turkish_punctuation.training_io import (
    directory_lock,
    file_hashes,
    model_digest,
    seed_everything,
    value_digest,
    write_json,
)


def load_training_tokenizers(
    source: dict, *, root: Path, allow_download: bool
) -> tuple[PreTrainedTokenizerFast, PreTrainedTokenizerFast]:
    """Check vocabulary/normalization compatibility before re-encoding real records."""
    options = {
        "cache_dir": str(root / source["cache_directory"]),
        "local_files_only": not allow_download,
        "use_fast": True,
        # The encoder's actual position table is checked separately when loading it.
        "model_max_length": source["max_input_length"],
        "trust_remote_code": False,
        "token": False,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        source["model_id"], revision=source["revision"], **options
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        source["base_model_id"], revision=source["base_model_revision"], **options
    )
    if not isinstance(tokenizer, PreTrainedTokenizerFast) or not isinstance(
        base_tokenizer, PreTrainedTokenizerFast
    ):
        raise TypeError("Both tokenizers must provide fast word alignment.")
    if tokenizer.get_vocab() != base_tokenizer.get_vocab():
        raise ValueError("Published/base vocabulary IDs differ.")
    if tokenizer.special_tokens_map != base_tokenizer.special_tokens_map:
        raise ValueError("Published/base special tokens differ.")
    if any(
        t.padding_side != "right" or t.model_max_length != source["max_input_length"]
        for t in (tokenizer, base_tokenizer)
    ):
        raise ValueError("Expected right padding and the pinned length limit.")
    left, right = (json.loads(t.backend_tokenizer.to_str()) for t in (tokenizer, base_tokenizer))
    for component in ("model", "normalizer", "pre_tokenizer", "decoder", "added_tokens"):
        if left[component] != right[component]:
            raise ValueError(f"Published/base tokenizer component differs: {component}")
    # Postprocessor serialization can differ while implementing the same BERT layout.
    probes = [["IŞIK", "İzmir", "çocuklarımızla"], ["gititm", "ışık", "ASCII"], ["a"]]
    a, b = (t(probes, is_split_into_words=True, padding=False) for t in (tokenizer, base_tokenizer))
    if a.data != b.data or any(a.word_ids(i) != b.word_ids(i) for i in range(len(probes))):
        raise ValueError("Published/base special-token processing differs.")
    return tokenizer, base_tokenizer


def export_model(
    model: BertForTokenClassification,
    tokenizer: PreTrainedTokenizerFast,
    destination: Path,
    *,
    identity: str,
) -> dict:
    """Save immutable HF-loadable weights and verify an export left by interruption."""
    tensor_hash = model_digest(model)
    if destination.exists():
        manifest = json.loads((destination / "export.json").read_text())
        if (
            manifest["identity"] != identity
            or manifest["model_tensors_sha256"] != tensor_hash
            or manifest["files"] != file_hashes(destination, exclude=("export.json",))
        ):
            raise ValueError(f"Existing model export differs: {destination}")
        return manifest
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".model-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "model"
        cpu_state = {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()}
        model.save_pretrained(staging, safe_serialization=True, state_dict=cpu_state)
        tokenizer.save_pretrained(staging)
        manifest = {
            "identity": identity,
            "model_tensors_sha256": tensor_hash,
            "files": file_hashes(staging),
        }
        write_json(staging / "export.json", manifest)
        staging.rename(destination)
    return manifest


def load_shared_initialization(
    config: dict,
    source: dict,
    tokenizer: PreTrainedTokenizerFast,
    *,
    seed: int,
    root: Path,
    directory: Path,
    allow_download: bool,
) -> tuple[BertForTokenClassification, dict]:
    """Create one complete initial state per seed, then reuse its exact tensors."""
    specification = {"model": config["model"], "source": source, "seed": seed}
    identity = value_digest(specification)
    destination = directory / f"seed-{seed}"
    with directory_lock(directory):
        if destination.exists():
            manifest = json.loads((destination / "initialization.json").read_text())
            if manifest["identity"] != identity:
                raise ValueError(
                    "Initialization specification differs; use a new directory/version."
                )
            if manifest["files"] != file_hashes(destination, exclude=("initialization.json",)):
                raise ValueError("Saved initialization files differ from their hashes.")
            model = BertForTokenClassification.from_pretrained(
                destination,
                local_files_only=True,
                use_safetensors=True,
                trust_remote_code=False,
                torch_dtype=torch.float32,
                attn_implementation=config["model"]["attention_implementation"],
            )
        else:
            seed_everything(seed)
            options = {
                "revision": source["base_model_revision"],
                "cache_dir": str(root / source["cache_directory"]),
                "local_files_only": not allow_download,
                "trust_remote_code": False,
                "token": False,
            }
            base = AutoConfig.from_pretrained(source["base_model_id"], **options)
            if base.model_type != "bert" or "BertForTokenClassification" in (
                base.architectures or []
            ):
                raise ValueError(
                    "Expected a base BERT encoder rather than a punctuation checkpoint."
                )
            if base.vocab_size != len(tokenizer) or base.max_position_embeddings != 512:
                raise ValueError("Base embedding vocabulary/position limit differs.")
            base.num_labels = len(LABEL_TO_ID)
            base.label2id = LABEL_TO_ID
            base.id2label = {value: key for key, value in LABEL_TO_ID.items()}
            for key in (
                "hidden_dropout_prob",
                "attention_probs_dropout_prob",
                "classifier_dropout",
            ):
                setattr(base, key, config["model"][key])
            # Transformers returns a pair with this flag; its annotation omits that case.
            model, loading = cast(
                tuple[BertForTokenClassification, dict],
                BertForTokenClassification.from_pretrained(
                    source["base_model_id"],
                    config=base,
                    output_loading_info=True,
                    use_safetensors=False,
                    weights_only=True,
                    torch_dtype=torch.float32,
                    attn_implementation=config["model"]["attention_implementation"],
                    **options,
                ),
            )
            if set(loading["missing_keys"]) != {"classifier.weight", "classifier.bias"}:
                raise ValueError(f"Expected only a fresh classifier: {loading}")
            allowed = ("cls.", "bert.pooler.", "pooler.")
            if (
                any(not name.startswith(allowed) for name in loading["unexpected_keys"])
                or loading["mismatched_keys"]
                or loading["error_msgs"]
            ):
                raise ValueError(f"Base encoder did not load completely: {loading}")
            with TemporaryDirectory(prefix=".initialization-", dir=directory) as temporary:
                staging = Path(temporary) / "model"
                model.save_pretrained(staging, safe_serialization=True)
                tokenizer.save_pretrained(staging)
                manifest = {
                    "identity": identity,
                    "specification": specification,
                    "model_tensors_sha256": model_digest(model),
                    "loading_info": loading,
                    "files": file_hashes(staging),
                }
                write_json(staging / "initialization.json", manifest)
                staging.rename(destination)
    if model_digest(model) != manifest["model_tensors_sha256"]:
        raise ValueError("Loaded initial tensors differ from the shared initialization.")
    if model.config.label2id != LABEL_TO_ID or not all(p.requires_grad for p in model.parameters()):
        raise ValueError("Expected the four project labels and full fine-tuning.")
    cast(torch.nn.Module, model).float()
    return model, manifest
