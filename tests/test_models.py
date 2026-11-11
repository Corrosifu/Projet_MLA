import keras
import numpy as np

from adv_repro.models import (
    Maxout,
    RBFOutput,
    logits,
    maxout_network,
    rbf_network,
    softmax_regression,
)


def test_maxout_is_the_max_over_its_affine_pieces():
    layer = Maxout(units=3, pieces=4)
    x = np.random.default_rng(0).normal(size=(5, 6)).astype("float32")
    out = keras.ops.convert_to_numpy(layer(x))
    kernel, bias = (keras.ops.convert_to_numpy(w) for w in layer.affine.weights)
    want = (x @ kernel + bias).reshape(5, 3, 4).max(axis=-1)
    assert out.shape == (5, 3) and np.allclose(out, want, atol=1e-5)


def test_rbf_layer_matches_its_formula_and_peaks_at_the_centre():
    means = np.array([[0.0, 0.0], [1.0, 1.0]], dtype="float32")
    model = rbf_network(input_dim=2, num_classes=2, class_means=means, init_beta=0.5)
    x = np.array([[0.0, 0.0], [1.0, 1.0], [0.5, 0.0]], dtype="float32")
    out = model.predict(x, verbose=0)
    d2 = ((x[:, None, :] - means[None]) ** 2).sum(-1)
    assert np.allclose(out, np.exp(-0.5 * d2), atol=1e-5)
    assert np.allclose(out[0, 0], 1.0) and np.allclose(out[1, 1], 1.0)
    far = model.predict(np.full((1, 2), 50.0, dtype="float32"), verbose=0)
    assert far.max() < 1e-6  # far from every centre: "I don't know"


def test_rbf_outputs_are_independent_probabilities_not_a_distribution():
    model = rbf_network(input_dim=4, num_classes=3, init_beta=0.1)
    out = model.predict(np.random.default_rng(0).normal(size=(8, 4)).astype("float32"), verbose=0)
    assert (out >= 0).all() and (out <= 1).all()


def test_logits_reproduce_the_softmax_output():
    model = maxout_network(input_dim=10, num_classes=4, units=5, pieces=3, hidden_layers=2)
    x = np.random.default_rng(1).normal(size=(6, 10)).astype("float32")
    z = logits(model, x)
    soft = np.exp(z - z.max(1, keepdims=True))
    soft /= soft.sum(1, keepdims=True)
    assert np.allclose(soft, model.predict(x, verbose=0), atol=1e-5)
    reg = softmax_regression(10, 4)
    z = logits(reg, x)
    assert np.allclose(
        np.exp(z) / np.exp(z).sum(1, keepdims=True), reg.predict(x, verbose=0), atol=1e-5
    )


def test_custom_layers_survive_a_save_and_load_round_trip(tmp_path):
    for model in (
        maxout_network(input_dim=8, num_classes=3, units=4, pieces=2, hidden_layers=1),
        rbf_network(input_dim=8, num_classes=3),
    ):
        path = tmp_path / f"{model.name}.keras"
        model.save(path)
        loaded = keras.saving.load_model(path)
        x = np.random.default_rng(2).normal(size=(4, 8)).astype("float32")
        assert np.allclose(model.predict(x, verbose=0), loaded.predict(x, verbose=0), atol=1e-6)


def test_layers_are_registered_for_serialization():
    assert Maxout.__name__ in str(keras.saving.get_registered_name(Maxout))
    assert RBFOutput.__name__ in str(keras.saving.get_registered_name(RBFOutput))
