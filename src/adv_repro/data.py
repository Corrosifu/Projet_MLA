"""Data loading. Labels and images are always aligned (no shuffled-dataset pitfalls)."""

from __future__ import annotations

import numpy as np


def load_mnist(flatten: bool = True):
    """MNIST as float32 in [0, 1] (the paper's setting, footnote 1). Downloads on first use."""
    import keras

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    scale = lambda a: (a.astype("float32") / 255.0).reshape(len(a), -1 if flatten else 28 * 28)  # noqa: E731
    return (scale(x_train), y_train.astype("int64")), (scale(x_test), y_test.astype("int64"))


def load_mnist_binary(class_a: int = 3, class_b: int = 7):
    """MNIST restricted to two classes (3 vs 7 in Figure 2). Label 1 = class_a, 0 = class_b."""
    (x_tr, y_tr), (x_te, y_te) = load_mnist()
    out = []
    for x, y in ((x_tr, y_tr), (x_te, y_te)):
        keep = (y == class_a) | (y == class_b)
        out.append((x[keep], (y[keep] == class_a).astype("int64")))
    return out[0], out[1]


def synthetic_blobs(n_per_class: int = 100, dim: int = 784, classes: int = 10, seed: int = 0):
    """Tiny image-like dataset in [0, 1] for tests and smoke runs: one noisy template per class."""
    rng = np.random.default_rng(seed)
    templates = rng.uniform(0.0, 1.0, size=(classes, dim)) > 0.7
    x = np.concatenate(
        [np.clip(t * 0.8 + rng.normal(0.1, 0.15, size=(n_per_class, dim)), 0, 1) for t in templates]
    ).astype("float32")
    y = np.repeat(np.arange(classes), n_per_class).astype("int64")
    order = rng.permutation(len(x))
    return x[order], y[order]


def split_validation(x, y, fraction: float = 0.1, seed: int = 0):
    """Deterministic train/validation split. Returns (x_tr, y_tr), (x_val, y_val)."""
    order = np.random.default_rng(seed).permutation(len(x))
    n_val = int(round(len(x) * fraction))
    val, tr = order[:n_val], order[n_val:]
    return (x[tr], y[tr]), (x[val], y[val])


def class_means(x, y, num_classes: int = 10) -> np.ndarray:
    return np.stack([x[y == k].mean(axis=0) for k in range(num_classes)])
