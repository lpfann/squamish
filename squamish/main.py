import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection.base import SelectorMixin
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import scale

from squamish.utils import create_support_AR, get_AR_params, get_MR, sort_features

from . import utils, plot

BEST_PARAMS_BORUTA = {
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
BEST_PARAMS_BORUTA = ParameterGrid(BEST_PARAMS_BORUTA)[0]

BEST_PARAMS_ITER = {"boosting_type": ["gbdt"]}
BEST_PARAMS_ITER = ParameterGrid(BEST_PARAMS_ITER)[0]


class Main(BaseEstimator, SelectorMixin):
    def __init__(
        self, problem="classification", params_boruta=None, params_iterative=None
    ):
        self.problem = problem
        if params_boruta is not None:
            self.params_boruta = BEST_PARAMS_BORUTA.update(params_boruta)
        else:
            self.params_boruta = BEST_PARAMS_BORUTA

        if params_iterative is not None:
            self.params_iter = BEST_PARAMS_ITER.update(params_iterative)
        else:
            self.params_iter = BEST_PARAMS_ITER

    def _get_support_mask(self):
        return self.support_

    def fit(self, X, y):
        X = scale(X)
        n, d = X.shape

        AR = np.where(get_AR_params(X, y, self.params_boruta))[0]
        MR, self.score_ = get_MR(X, y, self.params_iter)

        print(f"Features from Boruta:\n {AR}")
        print(f"Features from Lightbgm:\n {MR}")

        # Sort features iteratively into strongly (S) and weakly (W) sets
        S, W = sort_features(X, y, MR, AR, self.params_boruta, self.params_iter)
        # Turn index sets into support vector
        # (2 strong relevant,1 weak relevant, 0 irrelevant)
        all_rel_support = create_support_AR(d, S, W)
        self.relevance_classes_ = all_rel_support

        # Simple boolean vector where relevan features are regarded as one set (1 relevant, 0 irrelevant)
        self.support_ = self.relevance_classes_ > 0

        self.feature_importances_ = utils.mutual_information(X, y, problem=self.problem)

    def plot(self, ticklabels=None):
        return plot.plot_model_intervals(self, ticklabels=ticklabels)
