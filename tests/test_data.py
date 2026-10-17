"""Check that the reader preserves labels and rejects structurally unusable data."""

import csv
import tempfile
import unittest
from pathlib import Path

from turkish_punctuation.data import (
    find_empty_tokens,
    parse_record,
    read_split,
    summarize_split,
)


class ReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "sample.csv"

    def write_csv(self, rows, columns=("sentence", "tags")):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            writer.writerows(rows)

    def test_labeled_empty_tokens_and_turkish_characters_survive_reading(self):
        self.write_csv(
            [
                (repr(["İzmir", "", "güzel"]), repr(["O", "COMMA", "PERIOD"])),
                (repr(["Nasıl", ""]), repr(["O", "QUESTION_MARK"])),
            ]
        )
        original_bytes = self.path.read_bytes()
        records = read_split(self.path)
        self.assertEqual(records.iloc[0]["tokens"], ["İzmir", "", "güzel"])
        self.assertEqual(records.iloc[0]["tags"], ["O", "COMMA", "PERIOD"])
        self.assertEqual(
            find_empty_tokens(records).to_dict("records"),
            [
                {"row_id": 0, "token_index": 1, "tag": "COMMA"},
                {"row_id": 1, "token_index": 1, "tag": "QUESTION_MARK"},
            ],
        )
        summary = summarize_split(records)
        self.assertEqual(summary["token_entries"], 5)
        self.assertEqual(summary["empty_token_entries"], 2)
        self.assertEqual(summary["rows_with_empty_tokens"], 2)
        self.assertEqual(self.path.read_bytes(), original_bytes)

    def test_invalid_records_raise_with_the_source_row_id(self):
        cases = [
            ("['broken'", "['O']"),
            ("('tuple',)", "['O']"),
            ("['word', 3]", "['O', 'O']"),
            ("[]", "[]"),
            ("['one', 'two']", "['O']"),
            ("['word']", "['EXCLAMATION']"),
        ]
        for sentence, tags in cases:
            with self.subTest(sentence=sentence, tags=tags):
                with self.assertRaisesRegex(ValueError, "Row 7"):
                    parse_record(sentence, tags, row_id=7)

    def test_invalid_later_row_is_not_silently_dropped(self):
        self.write_csv(
            [
                ("['Good']", "['PERIOD']"),
                ("['Bad', 'alignment']", "['PERIOD']"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "Row 1"):
            read_split(self.path)

    def test_prefix_read_preserves_original_positions(self):
        self.write_csv(
            [
                ("['First']", "['PERIOD']"),
                ("['Second']", "['PERIOD']"),
            ]
        )
        records = read_split(self.path, nrows=1)
        self.assertEqual(records["row_id"].tolist(), [0])
        self.assertEqual(records.iloc[0]["tokens"], ["First"])
        self.assertTrue(find_empty_tokens(records).empty)
        with self.assertRaises(ValueError):
            read_split(self.path, nrows=0)

    def test_empty_or_wrong_schema_files_are_rejected(self):
        self.write_csv([])
        with self.assertRaisesRegex(ValueError, "No records"):
            read_split(self.path)
        self.write_csv([("['x']", "['O']")], columns=("text", "labels"))
        with self.assertRaisesRegex(ValueError, "Expected sentence/tags"):
            read_split(self.path)


if __name__ == "__main__":
    unittest.main()
