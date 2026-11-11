import numpy as np
import pytest

from adv_repro.data import class_means, split_validation, synthetic_blobs
from adv_repro.fgsm import fgsm
from adv_repro.losses import encode_labels, sparse_ce
from adv_repro.metrics import error_rate, predict_proba
from adv_repro.models import softmax_regression
from adv_repro.training import fit_adversarial, fit_standard


def adv_error(model, x, y, eps=0.25):
    return error_rate(predict_proba(model, fgsm(model, x, y, eps, sparse_ce)), y)


def test_labels_stay_aligned_with_images_during_training():
    """Regression test for the legacy notebooks, which paired images and labels from two
    different shuffled passes over the data (hence 10 % 'adversarial accuracy' on train)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(600, 20)).astype("float32")
    y = x[:, :4].argmax(axis=1)  # the label is a function of the first four features
    model = softmax_regression(20, 4)
    fit_standard(model, x, y, sparse_ce, epochs=60, batch_size=32, lr=0.05)
    assert error_rate(predict_proba(model, x), y) < 0.15


def small_maxout(dim):
    from adv_repro.models import maxout_network

    return maxout_network(dim, 4, units=32, pieces=3, hidden_layers=1, dropout=0.0)


def test_adversarial_training_reduces_the_error_on_adversarial_examples(blobs):
    """Section 6: a network with a hidden layer can learn to resist the fast gradient method."""
    x, y = blobs
    standard = small_maxout(x.shape[1])
    fit_standard(standard, x, y, sparse_ce, epochs=30, batch_size=32)
    robust = small_maxout(x.shape[1])
    fit_adversarial(robust, x, y, sparse_ce, eps=0.25, epochs=30, batch_size=32)
    assert adv_error(robust, x, y) < adv_error(standard, x, y) - 0.02


def test_a_linear_model_stays_far_less_robust_than_a_network_with_a_hidden_layer(blobs):
    """Sections 6 and 10: linear models lack the capacity to resist adversarial perturbation."""
    x, y = blobs
    linear = softmax_regression(x.shape[1], 4)
    fit_adversarial(linear, x, y, sparse_ce, eps=0.25, epochs=15, batch_size=32)
    hidden = small_maxout(x.shape[1])
    fit_adversarial(hidden, x, y, sparse_ce, eps=0.25, epochs=30, batch_size=32)
    assert adv_error(linear, x, y) > 0.5
    assert adv_error(linear, x, y) > adv_error(hidden, x, y) + 0.4


@pytest.mark.parametrize("perturbation", ["fgsm", "random_sign", "uniform"])
def test_all_perturbation_modes_train(blobs, perturbation):
    x, y = blobs
    model = softmax_regression(x.shape[1], 4)
    out = fit_adversarial(
        model, x, y, sparse_ce, epochs=2, batch_size=32, perturbation=perturbation
    )
    assert len(out["history"]) == 2 and np.isfinite(out["history"][-1]["loss"])


def test_unknown_perturbation_is_rejected(blobs):
    x, y = blobs
    with pytest.raises(ValueError, match="perturbation"):
        fit_adversarial(softmax_regression(x.shape[1], 4), x, y, sparse_ce, perturbation="bogus")


def test_early_stopping_restores_the_best_epoch_on_the_adversarial_validation_set(blobs):
    x, y = blobs
    (xt, yt), (xv, yv) = split_validation(x, y, fraction=0.25, seed=0)
    model = softmax_regression(x.shape[1], 4)
    out = fit_adversarial(model, xt, yt, sparse_ce, epochs=6, batch_size=32, val=(xv, yv))
    scores = [r["val_adv_error"] for r in out["history"]]
    assert out["best_val_score"] == min(scores)
    assert np.isclose(adv_error(model, xv, yv), min(scores))


def test_data_helpers():
    x, y = synthetic_blobs(n_per_class=20, dim=32, classes=5, seed=3)
    assert x.shape == (100, 32) and x.min() >= 0 and x.max() <= 1 and set(y) == set(range(5))
    (xt, yt), (xv, yv) = split_validation(x, y, fraction=0.2, seed=0)
    assert len(xv) == 20 and len(xt) == 80
    assert not {tuple(r) for r in xt} & {tuple(r) for r in xv}
    assert class_means(x, y, 5).shape == (5, 32)


def test_label_encoding():
    y = np.array([2, 0])
    assert encode_labels("sparse_ce", y).dtype == np.int32
    assert encode_labels("onehot_bce", y, 3).tolist() == [[0, 0, 1], [1, 0, 0]]
    assert encode_labels("binary_ce", y).shape == (2, 1)
    with pytest.raises(ValueError):
        encode_labels("nope", y)
