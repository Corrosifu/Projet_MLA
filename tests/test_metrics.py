import numpy as np

from adv_repro.metrics import (
    as_class_probs,
    class_agreement,
    confidence_on_errors,
    error_rate,
    mean_confidence,
    rubbish_report,
)

PROBS = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
Y = np.array([0, 1, 1, 0])  # examples 2 and 3 are mistakes, predicted with 0.7 and 0.6


def test_error_and_confidence():
    assert error_rate(PROBS, Y) == 0.5
    assert np.isclose(mean_confidence(PROBS), (0.9 + 0.8 + 0.7 + 0.6) / 4)
    assert np.isclose(confidence_on_errors(PROBS, Y), 0.65)


def test_confidence_on_errors_is_undefined_without_mistakes():
    assert np.isnan(confidence_on_errors(PROBS, PROBS.argmax(1)))


def test_binary_sigmoid_outputs_are_turned_into_two_class_probabilities():
    p = np.array([[0.2], [0.9]])
    assert np.allclose(as_class_probs(p), [[0.8, 0.2], [0.1, 0.9]])
    assert error_rate(p, np.array([0, 1])) == 0.0
    assert np.isclose(mean_confidence(p), 0.85)


class Constant:
    def __init__(self, row):
        self.row = np.asarray(row, dtype="float32")

    def predict(self, x, **_):
        return np.tile(self.row, (len(x), 1))


def test_rubbish_examples_count_confident_detections():
    sure = rubbish_report(Constant([0.9, 0.05, 0.05]), n=50, dim=8)
    assert sure["error_rate"] == 1.0 and np.isclose(sure["confidence_on_mistakes"], 0.9)
    unsure = rubbish_report(Constant([0.1, 0.1, 0.1]), n=50, dim=8)  # an RBF-like "I don't know"
    assert unsure["error_rate"] == 0.0 and np.isnan(unsure["confidence_on_mistakes"])


def test_class_agreement_on_the_reference_models_mistakes():
    ref, other, y = np.array([1, 1, 2, 0]), np.array([1, 3, 2, 3]), np.array([0, 0, 2, 0])
    assert class_agreement(ref, other, y) == 0.5  # mistakes of ref: examples 0 and 1
    assert class_agreement(ref, other, y, restrict_to_both_wrong=True) == 0.5
