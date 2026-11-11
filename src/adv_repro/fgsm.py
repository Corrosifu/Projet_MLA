"""The fast gradient sign method (Goodfellow et al., 2015, section 4).

    eta = eps * sign(grad_x J(theta, x, y))        x_adv = x + eta

``fgsm`` works for any differentiable Keras model; ``fgsm_logistic_exact`` is the closed form for
logistic regression, where the method is exact (section 5) and which doubles as a test oracle.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import tensorflow as tf

LossFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]  # per-sample loss


def _as_tensor(y) -> tf.Tensor:
    y = np.asarray(y)
    return tf.convert_to_tensor(y.astype("float32") if y.dtype.kind == "f" else y)


def gradient_sign(model, x, y, loss_fn: LossFn) -> tf.Tensor:
    """sign(grad_x J) for one batch. Dropout is disabled (``training=False``)."""
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = _as_tensor(y)
    with tf.GradientTape() as tape:
        tape.watch(x)
        # sum, not mean: each sample's gradient is independent of the batch size
        loss = tf.reduce_sum(loss_fn(y, model(x, training=False)))
    return tf.sign(tape.gradient(loss, x))


def fgsm(
    model,
    x,
    y,
    eps: float,
    loss_fn: LossFn,
    clip: tuple[float, float] | None = (0.0, 1.0),
    batch_size: int = 1024,
) -> np.ndarray:
    """Adversarial examples ``x + eps * sign(grad_x J)``, optionally clipped to the valid range.

    Note: where the gradient is exactly zero (fully saturated softmax) ``sign`` is 0 and the
    example is left unchanged, so very confident predictions can resist a single step.
    """
    x = np.asarray(x, dtype="float32")
    y = np.asarray(y)
    out = np.empty_like(x)
    for i in range(0, len(x), batch_size):
        xb = tf.convert_to_tensor(x[i : i + batch_size])
        adv = xb + eps * gradient_sign(model, xb, y[i : i + batch_size], loss_fn)
        if clip is not None:
            adv = tf.clip_by_value(adv, clip[0], clip[1])
        out[i : i + batch_size] = adv.numpy()
    return out


def pgd(
    model,
    x,
    y,
    eps: float,
    loss_fn: LossFn,
    steps: int = 20,
    step_size: float | None = None,
    clip: tuple[float, float] | None = (0.0, 1.0),
    random_start: bool = True,
    batch_size: int = 1024,
    seed: int = 0,
) -> np.ndarray:
    """Projected gradient descent attack (Madry et al., 2018): iterated FGSM inside the eps-box.

    Not in the 2015 paper. It is used here as a *stronger* check, because training and testing
    with the same single-step attack can overestimate robustness. With ``steps=1``,
    ``step_size=eps`` and ``random_start=False`` it is exactly :func:`fgsm`.
    """
    step_size = 2.5 * eps / steps if step_size is None else step_size
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype="float32")
    y = np.asarray(y)
    out = np.empty_like(x)
    lo, hi = (-np.inf, np.inf) if clip is None else clip
    for i in range(0, len(x), batch_size):
        x0 = x[i : i + batch_size]
        adv = x0 + (rng.uniform(-eps, eps, x0.shape).astype("float32") if random_start else 0.0)
        adv = np.clip(adv, lo, hi)
        for _ in range(steps):
            adv = (
                adv + step_size * gradient_sign(model, adv, y[i : i + batch_size], loss_fn).numpy()
            )
            adv = np.clip(x0 + np.clip(adv - x0, -eps, eps), lo, hi)
        out[i : i + batch_size] = adv
    return out


def fgsm_logistic_exact(x, y01, w, eps: float, clip: tuple[float, float] | None = (0.0, 1.0)):
    """Closed form for P(y=1|x) = sigmoid(w.x + b) with {0,1} labels: eta = -eps * (2y-1) * sign(w).

    The bias does not matter and the perturbation is the worst case inside the max-norm box.
    """
    x = np.asarray(x, dtype="float32")
    y_pm = 2.0 * np.asarray(y01, dtype="float32").reshape(-1, 1) - 1.0
    adv = x - eps * y_pm * np.sign(np.asarray(w, dtype="float32")).reshape(1, -1)
    return adv if clip is None else np.clip(adv, clip[0], clip[1])


def activation_shift(w, eps: float) -> float:
    """Largest change of the activation w.T x under a max-norm perturbation: eps * ||w||_1.

    With n weights of mean magnitude m this is eps * m * n: it grows linearly with the
    dimension while ||eta||_inf does not (the paper's "linear explanation", section 3).
    """
    return float(eps * np.abs(np.asarray(w)).sum())
