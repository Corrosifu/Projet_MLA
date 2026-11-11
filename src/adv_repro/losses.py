"""Per-sample training losses (FGSM must use the same loss the model was trained with)."""

from __future__ import annotations

import keras
import numpy as np


def sparse_ce(y_true, y_pred):
    """Softmax models: integer labels, probabilities out."""
    return keras.losses.sparse_categorical_crossentropy(y_true, y_pred)


def onehot_bce(y_true, y_pred):
    """RBF network: one independent probability per class, one-hot targets."""
    return keras.losses.binary_crossentropy(y_true, y_pred)


def binary_ce(y_true, y_pred):
    """Logistic regression: one sigmoid output, {0, 1} targets of shape (n, 1)."""
    return keras.losses.binary_crossentropy(y_true, y_pred)


LOSSES = {"sparse_ce": sparse_ce, "onehot_bce": onehot_bce, "binary_ce": binary_ce}


def encode_labels(kind: str, y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Encode integer labels the way the loss `kind` expects them."""
    y = np.asarray(y)
    if kind == "sparse_ce":
        return y.astype("int32")
    if kind == "onehot_bce":
        return np.eye(num_classes, dtype="float32")[y]
    if kind == "binary_ce":
        return y.astype("float32").reshape(-1, 1)
    raise ValueError(f"unknown loss kind {kind!r}")
