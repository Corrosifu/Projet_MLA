import keras
import pytest


@pytest.fixture(autouse=True)
def seed():
    keras.utils.set_random_seed(0)


@pytest.fixture
def blobs():
    from adv_repro.data import synthetic_blobs

    return synthetic_blobs(n_per_class=60, dim=64, classes=4, seed=1)
