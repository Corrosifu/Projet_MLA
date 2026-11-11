"""The reproduction pipeline: train the paper's models, attack them, compare with the paper."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import keras
import numpy as np

from adv_repro import data, plotting
from adv_repro.fgsm import fgsm, fgsm_logistic_exact, pgd
from adv_repro.losses import binary_ce, encode_labels, onehot_bce, sparse_ce
from adv_repro.metrics import (
    attack_report,
    class_agreement,
    error_rate,
    predict_proba,
    predictions,
    rubbish_report,
)
from adv_repro.models import (
    logistic_regression,
    logits,
    maxout_network,
    rbf_network,
    softmax_regression,
)
from adv_repro.training import fit_adversarial, fit_standard


@dataclass
class Config:
    dataset: str = "mnist"  # "mnist" or "synthetic" (smoke runs)
    eps: float = 0.25
    seed: int = 0
    epochs_logistic: int = 10
    epochs_softmax: int = 10
    epochs_maxout: int = 20
    epochs_adv: int = 30
    epochs_rbf: int = 20
    batch_size: int = 128
    val_fraction: float = 0.1
    controls: bool = False  # also train the two noise-augmentation controls (slow)
    n_rubbish: int = 10_000
    out_dir: str = "results"
    fig_dir: str = "figures"

    @classmethod
    def quick(cls, **kw) -> Config:
        base = {
            "epochs_logistic": 2, "epochs_softmax": 2, "epochs_maxout": 2,
            "epochs_adv": 2, "epochs_rbf": 2, "n_rubbish": 500,
        }  # fmt: skip
        return cls(**{**base, **kw})


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load(cfg: Config):
    if cfg.dataset == "mnist":
        return data.load_mnist(), data.load_mnist_binary()
    x, y = data.synthetic_blobs(n_per_class=80, dim=784, classes=10, seed=cfg.seed)
    n = int(len(x) * 0.8)
    full = (x[:n], y[:n]), (x[n:], y[n:])
    binary = []
    for xs, ys in full:
        keep = (ys == 3) | (ys == 7)
        binary.append((xs[keep], (ys[keep] == 3).astype("int64")))
    return full, (binary[0], binary[1])


def _attack(model, x, y, cfg, loss_fn, y_enc=None):
    return fgsm(model, x, y if y_enc is None else y_enc, cfg.eps, loss_fn)


def reproduce(cfg: Config) -> dict:
    keras.utils.set_random_seed(cfg.seed)
    out_dir, fig_dir = Path(cfg.out_dir), Path(cfg.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    (x_train, y_train), (x_test, y_test) = _load(cfg)[0]
    (xb_train, yb_train), (xb_test, yb_test) = _load(cfg)[1]
    (x_tr, y_tr), (x_val, y_val) = data.split_validation(
        x_train, y_train, cfg.val_fraction, cfg.seed
    )
    res: dict = {}
    _log(f"data: train {len(x_tr)}, validation {len(x_val)}, test {len(x_test)}")

    # -- Figure 2: logistic regression, 3 vs 7 (the method is exact here) ------------------------
    _log("logistic regression 3 vs 7")
    log = logistic_regression(xb_train.shape[1])
    fit_standard(
        log, xb_train, encode_labels("binary_ce", yb_train), binary_ce, cfg.epochs_logistic
    )
    w = log.layers[0].kernel.numpy().ravel()
    adv_exact = fgsm_logistic_exact(xb_test, yb_test, w, cfg.eps)
    adv_auto = fgsm(log, xb_test, encode_labels("binary_ce", yb_test), cfg.eps, binary_ce)
    res["logistic_3v7"] = {
        **attack_report(log, xb_test, adv_exact, yb_test),
        "autodiff_matches_closed_form": float(np.mean(np.all(np.isclose(adv_auto, adv_exact), 1))),
    }
    pick = [np.argmax(yb_test == 1), np.argmax(yb_test == 0)]  # one "3" and one "7"
    plotting.logistic_figure(w, xb_test[pick], adv_exact[pick], fig_dir / "fig2_logistic.png")

    # -- Section 4: softmax regression and maxout under FGSM --------------------------------------
    _log("shallow softmax regression")
    soft = softmax_regression(x_tr.shape[1])
    fit_standard(soft, x_tr, y_tr, sparse_ce, cfg.epochs_softmax, cfg.batch_size)
    adv_soft = _attack(soft, x_test, y_test, cfg, sparse_ce)
    res["softmax"] = attack_report(soft, x_test, adv_soft, y_test)

    _log("maxout network (dropout, standard training)")
    maxout = maxout_network(x_tr.shape[1])
    fit_standard(maxout, x_tr, y_tr, sparse_ce, cfg.epochs_maxout, cfg.batch_size)
    adv_max = _attack(maxout, x_test, y_test, cfg, sparse_ce)
    res["maxout"] = attack_report(maxout, x_test, adv_max, y_test)

    # -- Section 6: adversarial training ----------------------------------------------------------
    _log("maxout network, adversarial training (alpha = 0.5, early stopping on adversarial val.)")
    robust = maxout_network(x_tr.shape[1])
    fit_info = fit_adversarial(
        robust, x_tr, y_tr, sparse_ce, cfg.eps, 0.5, cfg.epochs_adv, cfg.batch_size,
        val=(x_val, y_val), seed=cfg.seed,
    )  # fmt: skip
    adv_rob = _attack(robust, x_test, y_test, cfg, sparse_ce)
    res["maxout_adv_trained"] = {
        **attack_report(robust, x_test, adv_rob, y_test),
        "best_epoch": int(np.argmin([r["val_adv_error"] for r in fit_info["history"]]) + 1),
        "transfer_to_standard_error": error_rate(predict_proba(maxout, adv_rob), y_test),
    }
    res["maxout"]["transfer_to_adv_trained_error"] = error_rate(
        predict_proba(robust, adv_max), y_test
    )

    # Beyond the paper: a stronger, iterative attack on the same models (does the FGSM-trained
    # model really resist, or only the attack it was trained on?)
    _log("beyond the paper: PGD check")
    pgd_steps = 5 if cfg.n_rubbish < 1000 else 20
    res["pgd"] = {"steps": pgd_steps}
    for name, m in (("maxout", maxout), ("maxout_adv_trained", robust)):
        adv_pgd = pgd(m, x_test, y_test, cfg.eps, sparse_ce, steps=pgd_steps, seed=cfg.seed)
        res["pgd"][name] = {
            "fgsm_error": res[name]["adv_error"],
            "pgd_error": error_rate(predict_proba(m, adv_pgd), y_test),
        }

    if cfg.controls:
        for name, mode in (("noise_sign", "random_sign"), ("noise_uniform", "uniform")):
            _log(f"control: training with {mode} noise")
            ctl = maxout_network(x_tr.shape[1])
            fit_adversarial(
                ctl, x_tr, y_tr, sparse_ce, cfg.eps, 0.5, cfg.epochs_adv, cfg.batch_size,
                perturbation=mode, val=(x_val, y_val), seed=cfg.seed,
            )  # fmt: skip
            res[name] = attack_report(
                ctl, x_test, _attack(ctl, x_test, y_test, cfg, sparse_ce), y_test
            )

    # -- Section 7: shallow RBF network -----------------------------------------------------------
    _log("shallow RBF network (no hidden layer)")
    rbf = rbf_network(x_tr.shape[1], class_means=data.class_means(x_tr, y_tr))
    y_oh_tr, y_oh_test = encode_labels("onehot_bce", y_tr), encode_labels("onehot_bce", y_test)
    fit_standard(rbf, x_tr, y_oh_tr, onehot_bce, cfg.epochs_rbf, cfg.batch_size, lr=1e-2)
    adv_rbf = fgsm(rbf, x_test, y_oh_test, cfg.eps, onehot_bce)
    res["rbf"] = attack_report(rbf, x_test, adv_rbf, y_test)

    # -- Section 8: do adversarial examples transfer? ---------------------------------------------
    _log("transferability")
    p_max = predictions(predict_proba(maxout, adv_max))
    p_soft = predictions(predict_proba(soft, adv_max))
    p_rbf = predictions(predict_proba(rbf, adv_max))
    res["transfer"] = {
        "maxout_to_softmax_agreement": class_agreement(p_max, p_soft, y_test),
        "maxout_to_rbf_agreement": class_agreement(p_max, p_rbf, y_test),
        "maxout_to_softmax_agreement_both_wrong": class_agreement(p_max, p_soft, y_test, True),
        "maxout_to_rbf_agreement_both_wrong": class_agreement(p_max, p_rbf, y_test, True),
    }

    # -- Appendix A: rubbish class examples -------------------------------------------------------
    _log("rubbish class examples (Gaussian noise)")
    to_test = {"softmax": soft, "maxout": maxout, "maxout_adv_trained": robust, "rbf": rbf}
    res["rubbish"] = {
        name: rubbish_report(m, cfg.n_rubbish, x_tr.shape[1], cfg.seed)
        for name, m in to_test.items()
    }

    # -- Figures 3 and 4 and an extra error-vs-eps curve ------------------------------------------
    _log("figures")
    plotting.examples_figure(
        x_test, {"softmax": adv_soft, "maxout": adv_max, "adv. trained": adv_rob},
        fig_dir / "adversarial_examples.png",
    )  # fmt: skip
    plotting.maxout_filters_figure(
        {"naive maxout": maxout.layers[0].affine.kernel.numpy(),
         "adversarially trained": robust.layers[0].affine.kernel.numpy()},
        fig_dir / "fig3_maxout_filters.png",
    )  # fmt: skip
    i = int(np.argmax(predictions(predict_proba(maxout, x_test)) == y_test))
    direction = np.sign(adv_max[i] - x_test[i]) if np.any(adv_max[i] != x_test[i]) else 1.0
    sweep = np.linspace(-15, 15, 121)
    z = logits(maxout, np.stack([x_test[i] + e * direction for e in sweep]))
    plotting.epsilon_sweep_figure(sweep, z, int(y_test[i]), fig_dir / "fig4_epsilon_sweep.png")
    grid = np.linspace(0.0, 0.5, 11)
    curves = {}
    for name, m, loss, enc in (
        ("softmax", soft, sparse_ce, y_test), ("maxout", maxout, sparse_ce, y_test),
        ("maxout adv. trained", robust, sparse_ce, y_test), ("RBF", rbf, onehot_bce, y_oh_test),
    ):  # fmt: skip
        curves[name] = [
            error_rate(predict_proba(m, fgsm(m, x_test, enc, float(e), loss)), y_test) for e in grid
        ]
    plotting.error_vs_eps_figure(grid, curves, fig_dir / "error_vs_epsilon.png")
    res["error_vs_epsilon"] = {"eps": grid.tolist(), **curves}

    res["meta"] = {
        "config": asdict(cfg),
        "runtime_seconds": round(time.time() - t0, 1),
        "python": platform.python_version(),
        "tensorflow": __import__("tensorflow").__version__,
        "keras": keras.__version__,
        "numpy": np.__version__,
    }
    (out_dir / "results.json").write_text(
        json.dumps(res, indent=2, default=float), encoding="utf-8"
    )
    _log(f"done in {res['meta']['runtime_seconds']} s -> {out_dir / 'results.json'}")
    return res
