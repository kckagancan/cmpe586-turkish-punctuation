"""Check supervision survives word splitting and rejects missing word positions."""

import unittest

from turkish_punctuation.tokenization import align_labels


class AlignmentTests(unittest.TestCase):
    def test_subword_splitting_does_not_duplicate_punctuation_targets(self):
        tags = ["O", "COMMA", "QUESTION_MARK"]
        aligned = align_labels([None, 0, 0, 1, 1, 1, 2, None, None], tags)
        self.assertEqual(aligned, [-100, 0, -100, 2, -100, -100, 3, -100, -100])
        self.assertEqual([value for value in aligned if value != -100], [0, 2, 3])

    def test_missing_words_and_truncation_are_rejected(self):
        for word_ids in ([None, 0, 2, None], [None, 0, 1, None]):
            with self.subTest(word_ids=word_ids):
                with self.assertRaisesRegex(ValueError, "coverage"):
                    align_labels(word_ids, ["O", "COMMA", "PERIOD"])

    def test_invalid_order_extra_words_and_unknown_labels_are_rejected(self):
        cases = [([1, 0], ["O", "PERIOD"]), ([0, 1], ["O"]), ([0], ["UNKNOWN"])]
        for word_ids, tags in cases:
            with self.subTest(word_ids=word_ids, tags=tags):
                with self.assertRaises(ValueError):
                    align_labels(word_ids, tags)

    def test_empty_example_is_rejected(self):
        with self.assertRaises(ValueError):
            align_labels([None, None], [])


if __name__ == "__main__":
    unittest.main()
