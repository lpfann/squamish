import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection.base import SelectorMixin
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import scale

from squamish.utils import create_support_AR
from squamish.algorithm import FeatureSorter
from . import utils, plot, models


class Main(BaseEstimator, SelectorMixin):
    def __init__(
        self,
        problem="classification",
        params_boruta=None,
        params_iterative=None,
        random_state=None,
    ):
        self.problem = problem
        if random_state is None:
            random_state = np.random.RandomState()
        self.random_state = random_state

    def _get_support_mask(self):
        return self.support_

    def fit(self, X, y):
        X = scale(X)
        n, d = X.shape

        AR, bor_score = models.fset_and_score(models.MyBoruta, X, y)
        MR, self.score_ = models.fset_and_score(models.RF, X, y)

        print(f"Features from Boruta:\n {AR}")
        print(f"Features from RF:\n {MR}")

        # Sort features iteratively into strongly (S) and weakly (W) sets
        self.fsorter = FeatureSorter(X, y, MR, AR, self.random_state)
        self.fsorter.check_each_feature()
        # Turn index sets into support vector
        # (2 strong relevant,1 weak relevant, 0 irrelevant)
        all_rel_support = create_support_AR(d, self.fsorter.S, self.fsorter.W)
        self.relevance_classes_ = all_rel_support

        # Simple boolean vector where relevan features are regarded as one set (1 relevant, 0 irrelevant)
        self.support_ = self.relevance_classes_ > 0
        # self.feature_importances_ = utils.compute_importances(importances)[1] # Take mean
        # self.interval_ = utils.emulate_intervals(importances)

    def plot(self, ticklabels=None):
        return plot.plot_model_intervals(self, ticklabels=ticklabels)
