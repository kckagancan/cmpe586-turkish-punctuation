"""Word-level punctuation metrics; retain O positions when counting false positives."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from turkish_punctuation.data import LABELS
from turkish_punctuation.tokenization import LABEL_TO_ID

PUNCTUATION_LABELS = ("PERIOD", "COMMA", "QUESTION_MARK")


def compute_metrics(reference: Sequence[int], prediction: Sequence[int]) -> dict:
    """Score aligned word IDs, after removing ignored subword/padding positions.

    Macro equally averages the three fixed punctuation classes, including any
    zero-support class as zero. Micro pools their TP/FP/FN. O remains in the input
    so insertions on O words count as false positives and omissions as false negatives.
    """
    if not reference or len(reference) != len(prediction):
        raise ValueError("Nonempty reference and prediction lists must have equal lengths.")
    allowed = set(LABEL_TO_ID.values())
    if set(reference) - allowed or set(prediction) - allowed:
        raise ValueError("Expected word-level class IDs 0..3; remove -100 positions first.")
    all_ids = list(LABEL_TO_ID.values())
    punctuation_ids = [LABEL_TO_ID[label] for label in PUNCTUATION_LABELS]
    precision, recall, f1, support = precision_recall_fscore_support(
        reference, prediction, labels=all_ids, average=None, zero_division=0
    )
    if support is None:
        raise RuntimeError("Per-class scoring did not return class supports.")
    precision, recall, f1, support = (
        np.asarray(values) for values in (precision, recall, f1, support)
    )
    matrix = confusion_matrix(reference, prediction, labels=all_ids)
    per_class = {
        label: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "predicted_count": int(matrix[:, i].sum()),
        }
        for i, label in enumerate(LABELS)
    }
    result = {
        "evaluated_words": len(reference),
        "per_class": per_class,
        "accuracy_all_classes_diagnostic": float(accuracy_score(reference, prediction)),
        "confusion_matrix": {
            "labels": list(LABELS),
            "rows": "reference",
            "columns": "prediction",
            "counts": matrix.tolist(),
        },
    }
    for average in ("macro", "micro"):
        p, r, f, _ = precision_recall_fscore_support(
            reference, prediction, labels=punctuation_ids, average=average, zero_division=0
        )
        result[f"punctuation_{average}"] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        }
    return result
