"""The models used in the paper, built with Keras 3 (no tensorflow_addons)."""

from __future__ import annotations

import keras
import numpy as np


@keras.saving.register_keras_serializable(package="adv_repro")
class Maxout(keras.layers.Layer):
    """Maxout unit (Goodfellow et al., 2013): max over ``pieces`` affine functions of the input."""

    def __init__(self, units: int, pieces: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.units, self.pieces = units, pieces
        self.affine = keras.layers.Dense(units * pieces)

    def build(self, input_shape):
        self.affine.build(input_shape)

    def call(self, inputs):
        z = keras.ops.reshape(self.affine(inputs), (-1, self.units, self.pieces))
        return keras.ops.max(z, axis=-1)

    def compute_output_shape(self, input_shape):
        return (*input_shape[:-1], self.units)

    def get_config(self):
        return {**super().get_config(), "units": self.units, "pieces": self.pieces}


@keras.saving.register_keras_serializable(package="adv_repro")
class RBFOutput(keras.layers.Layer):
    """Shallow RBF network with no hidden layer (section 7): one RBF unit per class.

    p(class k | x) = exp(-beta_k * ||x - mu_k||^2). The outputs are independent probabilities,
    not a distribution: they can all be close to 0, which is how the model says "I don't know".
    (The paper's formula has the sign typo exp((x-mu)^T beta (x-mu)); the minus sign is meant.)
    """

    def __init__(self, units: int = 10, init_beta: float = 0.03, **kwargs):
        super().__init__(**kwargs)
        self.units, self.init_beta = units, init_beta

    def build(self, input_shape):
        self.mu = self.add_weight(
            shape=(self.units, int(input_shape[-1])), initializer="zeros", name="mu"
        )
        self.log_beta = self.add_weight(
            shape=(self.units,),
            initializer=keras.initializers.Constant(float(np.log(self.init_beta))),
            name="log_beta",
        )

    def call(self, x):
        ops = keras.ops
        d2 = (
            ops.sum(x * x, axis=1, keepdims=True)
            - 2.0 * ops.matmul(x, ops.transpose(self.mu))
            + ops.sum(self.mu * self.mu, axis=1)[None, :]
        )
        return ops.exp(-ops.exp(self.log_beta)[None, :] * ops.maximum(d2, 0.0))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        return {**super().get_config(), "units": self.units, "init_beta": self.init_beta}


def softmax_regression(input_dim: int = 784, num_classes: int = 10) -> keras.Model:
    return keras.Sequential(
        [keras.layers.Input((input_dim,)), keras.layers.Dense(num_classes, activation="softmax")],
        name="softmax_regression",
    )


def logistic_regression(input_dim: int = 784) -> keras.Model:
    return keras.Sequential(
        [keras.layers.Input((input_dim,)), keras.layers.Dense(1, activation="sigmoid")],
        name="logistic_regression",
    )


def maxout_network(
    input_dim: int = 784,
    num_classes: int = 10,
    units: int = 240,
    pieces: int = 5,
    hidden_layers: int = 2,
    dropout: float = 0.5,
) -> keras.Model:
    """Permutation-invariant maxout network (240 units x 5 pieces, 2 layers, as in the paper)."""
    layers: list = [keras.layers.Input((input_dim,))]
    for _ in range(hidden_layers):
        layers += [Maxout(units, pieces), keras.layers.Dropout(dropout)]
    layers.append(keras.layers.Dense(num_classes, activation="softmax"))
    return keras.Sequential(layers, name="maxout")


def rbf_network(
    input_dim: int = 784,
    num_classes: int = 10,
    class_means: np.ndarray | None = None,
    init_beta: float = 0.03,
) -> keras.Model:
    """RBF network without hidden layer; centres start at the class means when given."""
    layer = RBFOutput(num_classes, init_beta)
    model = keras.Sequential([keras.layers.Input((input_dim,)), layer], name="rbf")
    if class_means is not None:
        layer.mu.assign(np.asarray(class_means, dtype="float32"))
    return model


def logits(model: keras.Model, x) -> np.ndarray:
    """Pre-softmax activations of a softmax model whose last layer is a Dense softmax."""
    h = keras.ops.convert_to_tensor(np.asarray(x, dtype="float32"))
    for layer in model.layers[:-1]:
        h = layer(h, training=False)
    last = model.layers[-1]
    return keras.ops.convert_to_numpy(keras.ops.matmul(h, last.kernel) + last.bias)
