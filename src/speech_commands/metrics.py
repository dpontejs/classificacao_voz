"""Label mappings and the metrics reported during training and evaluation."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def label_maps(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """The two dictionaries the model config stores, so the Hub shows words, not indices."""
    label2id = {label: i for i, label in enumerate(labels)}
    return label2id, {i: label for label, i in label2id.items()}


def compute_metrics(eval_pred) -> dict[str, float]:
    """Accuracy and macro F1 from (logits, labels), the shape the Trainer hands over."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
    }
