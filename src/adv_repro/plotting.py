"""Figures mirroring the paper (Fig. 2: logistic regression, 3: maxout filters, 4: eps)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _img(ax, a, cmap="gray", **kw):
    ax.imshow(np.asarray(a).reshape(28, 28), cmap=cmap, **kw)
    ax.axis("off")


def logistic_figure(w, x, x_adv, path: Path, labels=("3", "7")):
    """Fig. 2: (a) weights, (b) their sign, (c) clean 3 and 7, (d) their adversarial versions."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    _img(axes[0], w, cmap="RdBu_r")
    axes[0].set_title("(a) weights")
    _img(axes[1], np.sign(w), cmap="RdBu_r")
    axes[1].set_title("(b) sign of the weights")
    for ax, data, title in ((axes[2], x, "(c) clean"), (axes[3], x_adv, "(d) FGSM, eps = .25")):
        tile = np.hstack([data[0].reshape(28, 28), data[1].reshape(28, 28)])
        ax.imshow(tile, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{title}  ({labels[0]} | {labels[1]})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def examples_figure(x, x_adv_by_model: dict, path: Path, n: int = 10):
    """Clean images on the first row, then the adversarial images generated for each model."""
    rows = [("clean", x[:n])] + [(name, adv[:n]) for name, adv in x_adv_by_model.items()]
    fig, axes = plt.subplots(len(rows), n, figsize=(1.1 * n, 1.2 * len(rows)))
    for r, (name, imgs) in enumerate(rows):
        for c in range(n):
            _img(axes[r, c], imgs[c], vmin=0, vmax=1)
        axes[r, 0].set_title(name, fontsize=8, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def maxout_filters_figure(kernels: dict, path: Path, units: int = 8, pieces: int = 5):
    """Fig. 3: filters of single maxout units (rows) for each model (naive vs adversarial)."""
    fig, axes = plt.subplots(1, len(kernels), figsize=(4.2 * len(kernels), 4.2))
    axes = np.atleast_1d(axes)
    for ax, (name, k) in zip(axes, kernels.items(), strict=True):
        k = np.asarray(k).reshape(784, -1, pieces)  # (pixels, units, pieces)
        tile = np.vstack(
            [np.hstack([k[:, u, p].reshape(28, 28) for p in range(pieces)]) for u in range(units)]
        )
        ax.imshow(tile, cmap="gray")
        ax.axis("off")
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def epsilon_sweep_figure(eps_values, z, correct_class: int, path: Path):
    """Fig. 4 (left): pre-softmax activations of each class along the gradient-sign direction."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k in range(z.shape[1]):
        ax.plot(eps_values, z[:, k], label=str(k), lw=2.2 if k == correct_class else 1)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("eps")
    ax.set_ylabel("argument to softmax")
    ax.legend(ncol=5, fontsize=7, title=f"class (correct = {correct_class})")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def error_vs_eps_figure(eps_values, errors: dict, path: Path):
    """Extra (not in the paper): error rate under FGSM as a function of eps, per model."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for name, e in errors.items():
        ax.plot(eps_values, e, marker="o", ms=3, label=name)
    ax.set_xlabel("eps")
    ax.set_ylabel("error rate under FGSM")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
