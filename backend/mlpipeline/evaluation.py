"""OOS classification metrics; no return, gain/loss or trade PnL analytics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    log_loss, precision_recall_fscore_support,
)

from mlpipeline.labels import DOWN, NEUTRAL, UP, LABEL_NAMES

ALL_LABELS = np.array([DOWN, NEUTRAL, UP])


def evaluate(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Return confusion matrix (true rows, predicted columns), supports and scores."""
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if len(y_true) != len(probs):
        raise ValueError("Inconsistent truth/probability lengths")
    if not len(y_true):
        return {"n": 0, "balanced_accuracy": None, "macro_f1": None,
                "log_loss": None, "multiclass_brier": None, "confusion": None}
    if probs.shape != (len(y_true), 3) or not np.isfinite(probs).all():
        raise ValueError("Expected finite probabilities with shape (N, 3)")
    if (probs < 0).any() or (probs > 1).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid probability distribution")
    # sklearn can warn on LightGBM's tiny row-sum rounding differences.
    # Only normalize probabilities already validated as within 1e-5 of one.
    # Use the same normalized values for both log loss and multiclass Brier.
    probs = probs / probs.sum(axis=1, keepdims=True)
    predicted = probs.argmax(axis=1)
    recall_values = precision_recall_fscore_support(
        y_true, predicted, labels=ALL_LABELS, zero_division=0
    )
    p, r, f, supports = recall_values
    matrix = confusion_matrix(y_true, predicted, labels=ALL_LABELS)
    present_classes = set(y_true.tolist())
    onehot = np.eye(3)[y_true]
    # Sum of squared class probability errors per sample, 0..2, smaller is better.
    brier = np.mean(np.sum((probs - onehot)**2, axis=1))
    return {
        "n": int(len(y_true)),
        "class_distribution": {LABEL_NAMES[i]: int((y_true == i).sum()) for i in ALL_LABELS},
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)) if len(present_classes)>1 else None,
        "macro_f1": float(f1_score(y_true, predicted, labels=ALL_LABELS, average="macro", zero_division=0))
        if len(present_classes)>1 else None,
        "log_loss": float(log_loss(y_true, probs, labels=ALL_LABELS)) if len(present_classes)>1 else None,
        "multiclass_brier": float(brier),
        "confusion": matrix.tolist(),
        "class_metrics": {
            LABEL_NAMES[i]: {"precision": float(p[i]), "recall": float(r[i]),
                             "f1": float(f[i]), "support": int(supports[i])}
            for i in ALL_LABELS
        },
    }
