"""Prepare validated records for token classification without truncating targets."""

from typing import cast

import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
from transformers import DataCollatorForTokenClassification, PreTrainedTokenizerFast

from turkish_punctuation.tokenization import IGNORE_INDEX, align_labels


def partition_records(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate entire rows containing empty words; retain source IDs and row order.

    Input must come from read_split(), which validates token/label structure.
    The second frame is quarantine, not a repaired or relabeled training split.
    """
    has_empty = pd.Series(
        ["" in tokens for tokens in records["tokens"].tolist()], index=records.index, dtype=bool
    )
    return records.loc[~has_empty].copy(), records.loc[has_empty].copy()


def encode_records(
    records: pd.DataFrame,
    tokenizer: PreTrainedTokenizerFast,
    *,
    max_length: int,
    batch_size: int = 256,
) -> Dataset:
    """Encode eligible rows, keeping row_id for tracing predictions to raw text.

    Any missing word or overlong sequence stops preparation. Future noisy inputs
    must use the same checks; they must not silently lose words or whole rows.
    """
    if records.empty or batch_size < 1 or max_length < 1:
        raise ValueError("Nonempty records and positive lengths/batch sizes are required.")
    if max_length > tokenizer.model_max_length:
        raise ValueError("Requested max_length exceeds the tokenizer's model limit.")
    if records["row_id"].duplicated().any():
        raise ValueError("Source row IDs must be unique within a split.")
    columns = records[["row_id", "tokens", "tags"]]
    for row_id, tokens, tags in columns.itertuples(index=False, name=None):
        # A disappearing empty word can otherwise hide a malformed token/tag count.
        if len(tokens) != len(tags):
            raise ValueError(f"Row {row_id}: token and label counts differ.")
    dataset = Dataset.from_dict(
        {column: records[column].tolist() for column in ("row_id", "tokens", "tags")}
    )

    def encode_batch(batch: dict) -> dict:
        encoded = tokenizer(
            batch["tokens"], is_split_into_words=True, truncation=False, padding=False
        )
        labels = []
        for i, (row_id, tags) in enumerate(zip(batch["row_id"], batch["tags"], strict=True)):
            if len(encoded.data["input_ids"][i]) > max_length:
                raise ValueError(
                    f"Row {row_id}: exceeds {max_length} tokens; no truncation applied."
                )
            try:
                labels.append(align_labels(encoded.word_ids(batch_index=i), tags))
            except ValueError as error:
                raise ValueError(f"Row {row_id}: {error}") from error
        return {**encoded.data, "labels": labels}

    return dataset.map(
        encode_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=["tokens", "tags"],
        keep_in_memory=True,
        load_from_cache_file=False,
        desc="Encode and align labels",
    )


def make_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerFast,
    *,
    batch_size: int = 8,
    shuffle: bool = False,
    seed: int = 586,
) -> DataLoader:
    """Pad within each batch; hide provenance metadata from the model inputs.

    Enable shuffle for training only. Keep the unmodified dataset's row_id column
    to associate sequential evaluation predictions with their original records.
    """
    if batch_size < 1 or len(dataset) == 0:
        raise ValueError("A nonempty dataset and positive batch size are required.")
    model_inputs = dataset.remove_columns("row_id")
    collator = DataCollatorForTokenClassification(
        tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX, return_tensors="pt"
    )
    return DataLoader(
        # HF Dataset supports PyTorch's indexing contract without inheriting its class.
        cast(TorchDataset, model_inputs),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        generator=torch.Generator().manual_seed(seed),
    )
