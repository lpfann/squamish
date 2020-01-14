from unittest import TestCase

from fri import genClassificationData
from sklearn.datasets import make_classification
import pytest

from squamish.main import Main
import matplotlib as mpl


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
    assert model.relevance_classes_ is not None
    assert len(model._get_support_mask()) == X.shape[1]


def test_linear_data():
    X, y = genClassificationData(
        n_features=5, n_strel=1, n_redundant=2, n_samples=200, random_state=1234
    )
    model = Main()
    model.fit(X, y)
