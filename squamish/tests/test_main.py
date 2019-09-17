from unittest import TestCase
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
    )
    return X, y


def test__get_support_mask():
    pytest.fail()


def test_fit(data):
    X, y = data
    assert len(X) == len(y)

    pytest.fail()
