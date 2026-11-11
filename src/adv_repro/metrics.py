"""Metrics used by the paper: error rate and *confidence* (probability given to the prediction)."""

from __future__ import annotations

import numpy as np


def predict_proba(model, x, batch_size: int = 2048) -> np.ndarray:
    return np.asarray(
        model.predict(np.asarray(x, dtype="float32"), batch_size=batch_size, verbose=0)
    )


def as_class_probs(probs: np.ndarray) -> np.ndarray:
    """(n, 1) sigmoid outputs -> (n, 2) [P(0), P(1)]; (n, k) outputs are returned unchanged."""
    probs = np.asarray(probs)
    return np.hstack([1.0 - probs, probs]) if probs.ndim == 2 and probs.shape[1] == 1 else probs


def predictions(probs: np.ndarray) -> np.ndarray:
    return as_class_probs(probs).argmax(axis=1)


def error_rate(probs: np.ndarray, y) -> float:
    return float(np.mean(predictions(probs) != np.asarray(y)))


def mean_confidence(probs: np.ndarray) -> float:
    """Average probability assigned to the predicted class (over all examples)."""
    return float(as_class_probs(probs).max(axis=1).mean())


def confidence_on_errors(probs: np.ndarray, y) -> float:
    """Average probability assigned to the predicted class on *misclassified* examples.

    NaN when there is no mistake (the paper notes it is undefined in that case).
    """
    p = as_class_probs(probs)
    wrong = p.argmax(axis=1) != np.asarray(y)
    return float(p.max(axis=1)[wrong].mean()) if wrong.any() else float("nan")


def attack_report(model, x_clean, x_adv, y) -> dict:
    """Clean vs adversarial error and confidence for one model."""
    pc, pa = predict_proba(model, x_clean), predict_proba(model, x_adv)
    return {
        "clean_error": error_rate(pc, y),
        "clean_confidence": mean_confidence(pc),
        "adv_error": error_rate(pa, y),
        "adv_confidence_on_mistakes": confidence_on_errors(pa, y),
    }


def rubbish_report(model, n: int = 10_000, dim: int = 784, seed: int = 0, threshold: float = 0.5):
    """Appendix A: feed N(0, I) noise; a class with probability > threshold is a false detection."""
    x = np.random.default_rng(seed).normal(size=(n, dim)).astype("float32")
    p = as_class_probs(predict_proba(model, x))
    fooled = p.max(axis=1) > threshold
    return {
        "error_rate": float(fooled.mean()),
        "confidence_on_mistakes": float(p.max(axis=1)[fooled].mean())
        if fooled.any()
        else float("nan"),
    }


def class_agreement(pred_ref: np.ndarray, pred_other: np.ndarray, y, restrict_to_both_wrong=False):
    """How often ``pred_other`` gives the same class as ``pred_ref`` on ``pred_ref``'s mistakes."""
    y = np.asarray(y)
    mask = pred_ref != y
    if restrict_to_both_wrong:
        mask &= pred_other != y
    return float(np.mean(pred_other[mask] == pred_ref[mask])) if mask.any() else float("nan")
