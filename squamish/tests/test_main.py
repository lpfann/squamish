from unittest import TestCase

from hypothesis_auto import auto_test_module
from sklearn.datasets import make_classification
import pytest

from squamish.main import Main


@pytest.fixture
def data():
    X, y = make_classification(
        100,
        10,
        n_informative=2,
        n_redundant=1,
        n_clusters_per_class=2,
        flip_y=0,
        shuffle=False,
        random_state=123,
    )
    return X, y


@pytest.fixture(scope="module")
def model():
    return Main()


def test_fit(data, model):
    X, y = data
    assert len(X) == len(y)

    model.fit(X, y)
    assert model.rel_classes is not None
    assert len(model._get_support_mask()) == X.shape[1]
