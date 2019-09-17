from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.ensemble import RandomForestClassifier


class Main(BaseEstimator, SelectorMixin):
    def __init__(self):
        pass

    def _get_support_mask(self):
        pass

    def fit(self, X, y):
        rf = RandomForestClassifier()
