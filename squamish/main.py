from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.ensemble import RandomForestClassifier

from . import utils
from sklearn.preprocessing import scale
from sklearn.model_selection import ParameterGrid

best_params_rf = {
    "max_depth": [5],
    "boosting_type": ["rf"],
    "bagging_fraction": [0.632],
    "bagging_freq": [1],
    "feature_fraction": [0.1],
    "b_perc": [100],
    "b_n_estimators": ["auto"],
    "b_alpha": [0.01],
    "b_max_iter": [50],
    "importance_type": ["gain"],
}
best_params_rf = ParameterGrid(best_params_rf)[0]
best_params_boost = {"boosting_type": ["gbdt"]}
best_params_boost = ParameterGrid(best_params_boost)[0]


class Main(BaseEstimator, SelectorMixin):
    def __init__(self):
        pass

    def _get_support_mask(self):
        return self.support_

    def fit(self, X, y):
        X = scale(X)

        self.rel_classes = utils.get_ar_classes(X, y, best_params_rf, best_params_boost)
        self.support_ = self.rel_classes > 0

    def score(self, X):
        return -1
