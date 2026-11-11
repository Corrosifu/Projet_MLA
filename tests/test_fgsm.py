import numpy as np

from adv_repro.fgsm import activation_shift, fgsm, fgsm_logistic_exact, gradient_sign, pgd
from adv_repro.losses import binary_ce, encode_labels, sparse_ce
from adv_repro.models import logistic_regression, softmax_regression
from adv_repro.training import fit_standard


def trained_softmax(blobs, epochs=10):
    x, y = blobs
    model = softmax_regression(input_dim=x.shape[1], num_classes=4)
    fit_standard(model, x, y, sparse_ce, epochs=epochs, batch_size=32)
    return model


def test_perturbation_is_bounded_and_within_the_valid_range(blobs):
    x, y = blobs
    model = trained_softmax(blobs, epochs=3)
    x_adv = fgsm(model, x, y, eps=0.25, loss_fn=sparse_ce)
    assert np.abs(x_adv - x).max() <= 0.25 + 1e-6
    assert x_adv.min() >= 0.0 and x_adv.max() <= 1.0
    assert x_adv.shape == x.shape and x_adv.dtype == np.float32


def test_zero_epsilon_is_the_identity(blobs):
    x, y = blobs
    model = trained_softmax(blobs, epochs=1)
    assert np.array_equal(fgsm(model, x, y, eps=0.0, loss_fn=sparse_ce), x)


def test_unclipped_perturbation_is_exactly_eps_where_the_gradient_is_nonzero(blobs):
    x, y = blobs
    model = trained_softmax(blobs, epochs=1)
    delta = fgsm(model, x, y, eps=0.1, loss_fn=sparse_ce, clip=None) - x
    moved = np.abs(delta)[np.abs(delta) > 0]
    assert moved.size > 0 and np.allclose(moved, 0.1, atol=1e-6)


def test_the_attack_increases_the_loss_and_the_error(blobs):
    x, y = blobs
    model = trained_softmax(blobs)
    x_adv = fgsm(model, x, y, eps=0.25, loss_fn=sparse_ce)
    loss = lambda a: float(np.mean(sparse_ce(y, model.predict(a, verbose=0))))  # noqa: E731
    err = lambda a: float(np.mean(model.predict(a, verbose=0).argmax(1) != y))  # noqa: E731
    assert loss(x_adv) > loss(x) and err(x_adv) > err(x)


def test_batching_does_not_change_the_result(blobs):
    x, y = blobs
    model = trained_softmax(blobs, epochs=2)
    a = fgsm(model, x, y, 0.2, sparse_ce, batch_size=7)
    b = fgsm(model, x, y, 0.2, sparse_ce, batch_size=1000)
    assert np.array_equal(a, b)


def test_fgsm_matches_the_closed_form_for_logistic_regression():
    """Section 5: for logistic regression the method is exact: eta = -eps * (2y-1) * sign(w)."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, size=(200, 64)).astype("float32")
    y = rng.integers(0, 2, size=200)
    model = logistic_regression(64)  # random weights: no saturation, gradients never vanish
    w = model.layers[0].kernel.numpy().ravel()
    y_enc = encode_labels("binary_ce", y)
    for clip in (None, (0.0, 1.0)):
        got = fgsm(model, x, y_enc, 0.25, binary_ce, clip=clip)
        want = fgsm_logistic_exact(x, y, w, 0.25, clip=clip)
        assert np.allclose(got, want, atol=1e-6)


def test_linear_activation_shift_grows_with_the_dimension():
    """Section 3: w.eta = eps * ||w||_1 ~ eps * m * n, while ||eta||_inf stays at eps."""
    rng = np.random.default_rng(0)
    eps, ratios = 0.25, []
    for n in (10, 100, 1000, 10_000):
        w = rng.normal(size=n)
        eta = eps * np.sign(w)
        assert np.isclose(w @ eta, activation_shift(w, eps))
        assert np.abs(eta).max() == eps
        ratios.append(activation_shift(w, eps) / (eps * np.abs(w).mean() * n))
    assert np.allclose(ratios, 1.0)
    assert activation_shift(rng.normal(size=10_000), eps) > 100 * activation_shift(
        rng.normal(size=10), eps
    )


def test_gradient_sign_takes_values_in_minus_one_zero_one(blobs):
    x, y = blobs
    model = trained_softmax(blobs, epochs=1)
    s = gradient_sign(model, x[:16], y[:16], sparse_ce).numpy()
    assert set(np.unique(s)) <= {-1.0, 0.0, 1.0}


def test_pgd_with_one_step_and_no_random_start_is_fgsm(blobs):
    x, y = blobs
    model = trained_softmax(blobs, epochs=2)
    one_step = pgd(model, x, y, 0.2, sparse_ce, steps=1, step_size=0.2, random_start=False)
    assert np.allclose(one_step, fgsm(model, x, y, 0.2, sparse_ce), atol=1e-6)


def test_pgd_stays_in_the_box_and_is_at_least_as_strong_as_fgsm(blobs):
    x, y = blobs
    model = trained_softmax(blobs)
    x_pgd = pgd(model, x, y, 0.25, sparse_ce, steps=10)
    assert np.abs(x_pgd - x).max() <= 0.25 + 1e-6 and x_pgd.min() >= 0 and x_pgd.max() <= 1
    err = lambda a: float(np.mean(model.predict(a, verbose=0).argmax(1) != y))  # noqa: E731
    assert err(x_pgd) >= err(fgsm(model, x, y, 0.25, sparse_ce)) - 0.01
