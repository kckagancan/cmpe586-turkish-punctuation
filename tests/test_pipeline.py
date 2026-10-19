"""Offline regression checks with a tiny tokenizer and randomly initialized BERT.

These fixtures check pipeline behavior only; they are not research experiments.
No dataset download, pretrained weights or GPU is required.
"""

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import BertConfig, BertForTokenClassification, PreTrainedTokenizerFast

from scripts import prepare_data
from turkish_punctuation.inference import load_published_model, predict_records
from turkish_punctuation.preprocessing import encode_records, make_dataloader, partition_records
from turkish_punctuation.tokenization import IGNORE_INDEX, LABEL_TO_ID


def tiny_tokenizer() -> PreTrainedTokenizerFast:
    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "Dün",
        "okula",
        "gittim",
        "git",
        "##it",
        "##m",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
    ]
    vocabulary = {word: index for index, word in enumerate(words)}
    backend = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        special_tokens=[("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=64,
    )


def sample_records() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"row_id": 10, "tokens": ["Dün", "okula", "gititm"], "tags": ["O", "COMMA", "PERIOD"]},
            {"row_id": 42, "tokens": ["gittim"], "tags": ["QUESTION_MARK"]},
        ],
        index=[4, 9],
    )


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tiny_tokenizer()
        self.records = sample_records()

    def test_partition_preserves_source_ids_order_labels_and_original_frame(self):
        records = pd.concat(
            [
                self.records,
                pd.DataFrame(
                    [{"row_id": 99, "tokens": ["a", ""], "tags": ["O", "PERIOD"]}], index=[17]
                ),
            ]
        )
        original = records.copy(deep=True)
        eligible, quarantine = partition_records(records)
        self.assertEqual(eligible["row_id"].tolist(), [10, 42])
        self.assertEqual(eligible.index.tolist(), [4, 9])
        self.assertEqual(quarantine.iloc[0]["tokens"], ["a", ""])
        self.assertEqual(quarantine.iloc[0]["tags"], ["O", "PERIOD"])
        pd.testing.assert_frame_equal(records, original)

    def test_encoding_keeps_one_target_per_word_and_source_ids(self):
        encoded = encode_records(self.records, self.tokenizer, max_length=64, batch_size=1)
        self.assertEqual(encoded["row_id"], [10, 42])
        self.assertNotIn("tokens", encoded.column_names)
        self.assertNotIn("tags", encoded.column_names)
        expected = [[0, 2, 1], [3]]
        for record, labels in zip(encoded.to_list(), expected, strict=True):
            self.assertEqual([x for x in record["labels"] if x != IGNORE_INDEX], labels)
            self.assertEqual(len(record["input_ids"]), len(record["labels"]))

    def test_malformed_counts_cannot_hide_behind_a_disappearing_empty_word(self):
        records = pd.DataFrame([{"row_id": 99, "tokens": ["gittim", ""], "tags": ["PERIOD"]}])
        with self.assertRaisesRegex(ValueError, "Row 99: token and label counts differ"):
            encode_records(records, self.tokenizer, max_length=64)

    def test_missing_words_and_overlong_inputs_are_never_silently_removed(self):
        missing = pd.DataFrame([{"row_id": 17, "tokens": ["a", ""], "tags": ["O", "COMMA"]}])
        with self.assertRaisesRegex(ValueError, "Row 17:.*coverage"):
            encode_records(missing, self.tokenizer, max_length=64)
        with self.assertRaisesRegex(ValueError, "Row 10: exceeds 4 tokens"):
            encode_records(self.records, self.tokenizer, max_length=4)

    def test_duplicate_ids_and_invalid_limits_or_sizes_are_rejected(self):
        duplicate = self.records.copy()
        duplicate["row_id"] = 10
        with self.assertRaisesRegex(ValueError, "unique"):
            encode_records(duplicate, self.tokenizer, max_length=64)
        for kwargs in ({"max_length": 65}, {"max_length": 0}, {"max_length": 64, "batch_size": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                encode_records(self.records, self.tokenizer, **kwargs)
        with self.assertRaises(ValueError):
            encode_records(self.records.iloc[:0], self.tokenizer, max_length=64)

    def test_unknown_piece_still_has_a_supervised_word_position(self):
        records = pd.DataFrame([{"row_id": 7, "tokens": ["zzzz"], "tags": ["PERIOD"]}])
        encoded = encode_records(records, self.tokenizer, max_length=64)
        self.assertIn(self.tokenizer.unk_token_id, encoded[0]["input_ids"])
        self.assertEqual(encoded[0]["labels"], [-100, 1, -100])

    def test_padding_masks_and_metadata_exclusion(self):
        encoded = encode_records(self.records, self.tokenizer, max_length=64)
        loader = make_dataloader(encoded, self.tokenizer, batch_size=2)
        batch = next(iter(loader))
        self.assertNotIn("row_id", batch)
        self.assertIn("row_id", encoded.column_names)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 7))
        self.assertEqual(batch["input_ids"].dtype, torch.int64)
        self.assertEqual(int((batch["labels"] != IGNORE_INDEX).sum()), 4)
        padding = batch["attention_mask"] == 0
        self.assertTrue(torch.all(batch["labels"][padding] == IGNORE_INDEX).item())
        self.assertTrue(
            torch.all(batch["input_ids"][padding] == self.tokenizer.pad_token_id).item()
        )

    def test_shuffle_is_reproducible_and_sequential_order_is_preserved(self):
        records = pd.DataFrame(
            [
                {"row_id": i + 100, "tokens": [word], "tags": ["O"]}
                for i, word in enumerate("abcdef")
            ]
        )
        encoded = encode_records(records, self.tokenizer, max_length=64)

        def order(shuffle: bool) -> list[int]:
            loader = make_dataloader(
                encoded, self.tokenizer, batch_size=2, shuffle=shuffle, seed=586
            )
            return [value for batch in loader for value in batch["input_ids"][:, 1].tolist()]

        self.assertEqual(order(False), [row["input_ids"][1] for row in encoded.to_list()])
        self.assertEqual(order(True), order(True))
        self.assertEqual(encoded["row_id"], list(range(100, 106)))


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = tiny_tokenizer()
        self.records = sample_records()
        config = BertConfig(
            vocab_size=len(self.tokenizer),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=64,
            label2id=LABEL_TO_ID,
            id2label={value: key for key, value in LABEL_TO_ID.items()},
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(586)
            self.model = BertForTokenClassification(config)

    def predict(self, records=None, **kwargs):
        return predict_records(
            self.records if records is None else records,
            self.tokenizer,
            self.model,
            max_length=64,
            **kwargs,
        )

    def test_eval_mode_no_gradients_and_reference_labels_are_not_model_inputs(self):
        self.model.train()
        original_forward = self.model.forward

        def checked_forward(**inputs):
            self.assertNotIn("labels", inputs)
            self.assertFalse(torch.is_grad_enabled())
            self.assertFalse(self.model.training)
            return original_forward(**inputs)

        with patch.object(self.model, "forward", side_effect=checked_forward):
            results = self.predict()
        self.assertEqual([row["row_id"] for row in results], [10, 42])
        for result, (_, original) in zip(results, self.records.iterrows(), strict=True):
            self.assertEqual(result["words"], original["tokens"])
            self.assertEqual(len(result["predicted_tags"]), len(original["tokens"]))
            self.assertTrue(all(len(logits) == 4 for logits in result["word_logits"]))

    def test_reference_class_values_do_not_change_predictions(self):
        original = self.predict()
        changed = self.records.copy(deep=True)
        changed["tags"] = [["QUESTION_MARK", "PERIOD", "COMMA"], ["O"]]
        alternate = self.predict(changed)
        for left, right in zip(original, alternate, strict=True):
            self.assertEqual(left["predicted_tags"], right["predicted_tags"])
            self.assertEqual(left["word_logits"], right["word_logits"])

    def test_batch_padding_does_not_change_word_logits(self):
        single = self.predict(batch_size=1)
        batched = self.predict(batch_size=2)
        for left, right in zip(single, batched, strict=True):
            torch.testing.assert_close(
                torch.tensor(left["word_logits"]),
                torch.tensor(right["word_logits"]),
                rtol=1e-5,
                atol=1e-6,
            )

    def test_mismatched_model_label_mapping_is_rejected(self):
        self.model.config.label2id = {**LABEL_TO_ID, "O": 1, "PERIOD": 0}
        with self.assertRaisesRegex(ValueError, "label mapping"):
            self.predict()

    def test_overlarge_position_limit_is_rejected(self):
        self.model.config.max_position_embeddings = 32
        with self.assertRaisesRegex(ValueError, "position limit"):
            self.predict()

    def test_nonfinite_logits_are_rejected(self):
        def nonfinite_forward(**inputs):
            shape = (*inputs["input_ids"].shape, 4)
            return SimpleNamespace(logits=torch.full(shape, float("nan")))

        with patch.object(self.model, "forward", side_effect=nonfinite_forward):
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                self.predict()

    def test_wrong_output_shape_is_rejected(self):
        def wrong_shape(**inputs):
            return SimpleNamespace(logits=torch.zeros((*inputs["input_ids"].shape, 3)))

        with patch.object(self.model, "forward", side_effect=wrong_shape):
            with self.assertRaisesRegex(RuntimeError, "dimensions"):
                self.predict()

    def test_checkpoint_loader_rejects_mapping_and_incomplete_weights(self):
        config = {
            "model_id": "local-test-fixture",
            "revision": "fixture",
            "cache_directory": ".cache",
            "label_to_id": LABEL_TO_ID,
            "max_input_length": 64,
        }
        with (
            patch(
                "turkish_punctuation.inference.AutoConfig.from_pretrained",
                return_value=self.model.config,
            ),
            patch(
                "turkish_punctuation.inference.AutoModelForTokenClassification.from_pretrained"
            ) as load,
        ):
            load.return_value = (self.model, {"missing_keys": ["classifier.weight"]})
            with self.assertRaisesRegex(ValueError, "did not load exactly"):
                load_published_model(config, root=Path("."))
            self.assertTrue(load.call_args.kwargs["weights_only"])
            self.assertTrue(load.call_args.kwargs["local_files_only"])
            self.assertFalse(load.call_args.kwargs["trust_remote_code"])
            load.return_value = (self.model, {})
            loaded = load_published_model(config, root=Path("."))
            self.assertFalse(loaded.training)
            self.model.config.label2id = {"WRONG": 0}
            with self.assertRaisesRegex(ValueError, "label mapping"):
                load_published_model(config, root=Path("."))


class PreparationScriptTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "configs").mkdir()
        (self.root / "docs").mkdir()
        (self.root / "data/raw").mkdir(parents=True)
        self.destination = self.root / "data/processed/clean-v1"
        self.tokenizer = tiny_tokenizer()
        tokenizer_config = {
            "model_id": "local-test-fixture",
            "revision": "fixture",
            "cache_directory": ".cache",
            "label_to_id": LABEL_TO_ID,
            "max_input_length": 64,
            "ignore_index": IGNORE_INDEX,
            "alignment": "first_subword",
        }
        preprocessing = {
            "version": "clean-v1",
            "empty_token_policy": "quarantine_whole_row",
            "output_directory": "data/processed/clean-v1",
            "tokenizer_batch_size": 2,
            "preview_batch_size": 2,
            "seed": 586,
        }
        self.manifest = {"local_directory": "data/raw", "revision": "fixture", "files": []}
        for split in ("train", "validation", "test"):
            path = self.root / "data/raw" / f"{split}.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["sentence", "tags"])
                writer.writerows(
                    [
                        ("['Dün', 'gittim']", "['O', 'PERIOD']"),
                        ("['a', '']", "['O', 'COMMA']"),
                        ("['gititm']", "['QUESTION_MARK']"),
                    ]
                )
            self.manifest["files"].append(
                {
                    "split": split,
                    "filename": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        for name, value in (
            ("configs/tokenizer.json", tokenizer_config),
            ("configs/preprocessing.json", preprocessing),
            ("docs/data-manifest.json", self.manifest),
        ):
            (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def run_preparation(self):
        with (
            patch("sys.argv", ["prepare_data.py"]),
            patch.object(prepare_data, "ROOT", self.root),
            patch.object(
                prepare_data.AutoTokenizer, "from_pretrained", return_value=self.tokenizer
            ),
            patch("builtins.print"),
        ):
            prepare_data.main()

    def test_full_preparation_saves_eligible_rows_quarantine_and_provenance(self):
        from datasets import load_from_disk

        self.run_preparation()
        prepared = load_from_disk(str(self.destination / "dataset"))
        report = json.loads((self.destination / "manifest.json").read_text())
        for item in self.manifest["files"]:
            split = item["split"]
            self.assertEqual(prepared[split]["row_id"], [0, 2])
            self.assertEqual(report["splits"][split]["eligible_row_ids"], [0, 2])
            self.assertEqual(report["splits"][split]["quarantined_row_ids"], [1])
            quarantine = json.loads(
                (self.destination / "quarantine" / f"{split}.jsonl").read_text()
            )
            self.assertEqual(quarantine["row_id"], 1)
            self.assertEqual(quarantine["tags"], ["O", "COMMA"])
            original = self.root / "data/raw" / item["filename"]
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(), item["sha256"])
        original_manifest = (self.destination / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_preparation()
        self.assertEqual((self.destination / "manifest.json").read_bytes(), original_manifest)

    def test_corrupt_source_fails_without_partial_outputs(self):
        with (self.root / "data/raw/train.csv").open("a") as stream:
            stream.write("corrupt\n")
        with self.assertRaisesRegex(ValueError, "differs from manifest"):
            self.run_preparation()
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.iterdir()), [])

    def test_encoding_failure_cleans_staging_and_preserves_raw_files(self):
        with patch.object(
            prepare_data, "encode_records", side_effect=ValueError("fixture failure")
        ):
            with self.assertRaisesRegex(ValueError, "fixture failure"):
                self.run_preparation()
        self.assertFalse(self.destination.exists())
        self.assertEqual(list(self.destination.parent.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
