"""Align word labels to subwords without silently losing supervised word positions."""

from collections.abc import Sequence

from turkish_punctuation.data import LABELS

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
IGNORE_INDEX = -100


def align_labels(word_ids: Sequence[int | None], tags: Sequence[str]) -> list[int]:
    """Label the first subword of each word; reject missing or reordered words.

    None identifies special/padding tokens. Empty source words and truncation
    can make word IDs disappear; the caller must resolve those cases explicitly.
    """
    observed = [word_id for word_id in word_ids if word_id is not None]
    if sorted(set(observed)) != list(range(len(tags))):
        raise ValueError("Word coverage differs from labels; check empty words or truncation.")
    if observed != sorted(observed):
        raise ValueError("Word IDs must follow the original word order.")
    if not tags:
        raise ValueError("At least one word label is required.")
    if set(tags) - set(LABEL_TO_ID):
        raise ValueError("Unknown punctuation label.")

    aligned, seen = [], set()
    for word_id in word_ids:
        if word_id is None or word_id in seen:
            aligned.append(IGNORE_INDEX)
        else:
            aligned.append(LABEL_TO_ID[tags[word_id]])
            seen.add(word_id)
    return aligned
