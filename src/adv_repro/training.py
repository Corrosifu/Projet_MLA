"""Standard and adversarial training (Goodfellow et al., 2015, section 6)."""

from __future__ import annotations

import keras
import numpy as np
import tensorflow as tf

from adv_repro.fgsm import LossFn, fgsm
from adv_repro.metrics import error_rate, predict_proba


def fit_standard(
    model, x, y, loss_fn: LossFn, epochs: int = 10, batch_size: int = 128, lr: float = 1e-3
) -> dict:
    """Plain training on numpy arrays (examples and labels are shuffled together by Keras)."""
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=loss_fn)
    hist = model.fit(x, y, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
    return {"loss": [float(v) for v in hist.history["loss"]]}


def fit_adversarial(
    model,
    x,
    y,
    loss_fn: LossFn,
    eps: float = 0.25,
    alpha: float = 0.5,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    perturbation: str = "fgsm",
    clip: tuple[float, float] | None = (0.0, 1.0),
    val: tuple[np.ndarray, np.ndarray] | None = None,
    seed: int = 0,
) -> dict:
    """Minimise  alpha * J(x) + (1 - alpha) * J(x + eps * sign(grad_x J))  (paper, section 6).

    ``perturbation`` selects the second term's input: ``"fgsm"`` (the paper's method, recomputed
    from the current model at every step) or one of the control experiments of the paper:
    ``"random_sign"`` (x +- eps) and ``"uniform"`` (x + U(-eps, eps)).

    With ``val`` the best epoch is chosen on the *adversarial* validation error (the paper's early
    stopping criterion) and its weights are restored.
    """
    if perturbation not in {"fgsm", "random_sign", "uniform"}:
        raise ValueError(f"unknown perturbation {perturbation!r}")
    opt = keras.optimizers.Adam(lr)
    variables = model.trainable_variables

    @tf.function
    def step(xb, yb):
        if perturbation == "fgsm":
            with tf.GradientTape() as tape:
                tape.watch(xb)
                inner = tf.reduce_sum(loss_fn(yb, model(xb, training=False)))
            x_pert = xb + eps * tf.sign(tape.gradient(inner, xb))
        elif perturbation == "random_sign":
            x_pert = xb + eps * tf.sign(tf.random.normal(tf.shape(xb)))
        else:
            x_pert = xb + tf.random.uniform(tf.shape(xb), -eps, eps)
        if clip is not None:
            x_pert = tf.clip_by_value(x_pert, clip[0], clip[1])
        with tf.GradientTape() as tape:
            clean = tf.reduce_mean(loss_fn(yb, model(xb, training=True)))
            pert = tf.reduce_mean(loss_fn(yb, model(x_pert, training=True)))
            loss = alpha * clean + (1.0 - alpha) * pert
        opt.apply_gradients(zip(tape.gradient(loss, variables), variables, strict=True))
        return loss

    x = np.asarray(x, dtype="float32")
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    history, best_score, best_weights = [], np.inf, None
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        losses = []
        for start in range(0, len(x), batch_size):
            idx = order[start : start + batch_size]
            losses.append(float(step(tf.constant(x[idx]), tf.constant(y[idx]))))
        record = {"epoch": epoch + 1, "loss": float(np.mean(losses))}
        if val is not None:
            xv, yv = val
            record["val_error"] = error_rate(predict_proba(model, xv), yv)
            x_adv = fgsm(model, xv, yv, eps, loss_fn, clip)
            record["val_adv_error"] = error_rate(predict_proba(model, x_adv), yv)
            score = record["val_adv_error"] if perturbation == "fgsm" else record["val_error"]
            if score < best_score:
                best_score, best_weights, record["best"] = score, model.get_weights(), True
        history.append(record)
    if best_weights is not None:
        model.set_weights(best_weights)
    return {"history": history, "best_val_score": None if best_weights is None else best_score}
