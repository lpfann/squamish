from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from squamish.utils import create_support_AR
from squamish.algorithm import FeatureSorter
from . import models
import logging

from .stat import Stats

logging = logging.getLogger(__name__)


class Main(BaseEstimator):
    def __init__(
            self,
            problem="classification",
            params_boruta=None,
            params_iterative=None,
            random_state=None,
    ):
        self.problem = problem
        self.random_state = check_random_state(random_state)

    def _get_support_mask(self):
        return self.support_

    def fit(self, X, y):
        X = scale(X)
        n, d = X.shape

        # All relevant set using Boruta
        m = models.MyBoruta(random_state=self.random_state).fit(X, y)
        # bor_score = m.cvscore(X, y)
        fset = m.fset(X, y)
        AR = np.where(fset)[0]

        # Fit a simple Random Forest to get a minimal feature subset
        m = models.RF(random_state=self.random_state).fit(X, y)
        self.score_ = m.cvscore(X, y)
        logging.info(f"RF score {self.score_}")
        logging.info(m.estimator.feature_importances_)
        self.rfmodel = deepcopy(m)

        self.stat_ = Stats(m, X, y, n_resampling=20, fpr=1e-4, check_importances=True)
        fset = m.fset(X, y, self.stat_)
        fset = np.where(fset)
        MR = fset[0]

        logging.info(f"Features from Boruta: {AR}")
        logging.info(f"Features from RF: {MR}")

        # Sort features iteratively into strongly (S) and weakly (W) sets
        self.fsorter = FeatureSorter(X, y, MR, AR, self.random_state, self.stat_)
        self.fsorter.check_each_feature()
        self.relations_ = self.fsorter.related

        # Turn index sets into support vector
        # (2 strong relevant,1 weak relevant, 0 irrelevant)
        all_rel_support = create_support_AR(d, self.fsorter.S, self.fsorter.W)
        self.relevance_classes_ = all_rel_support
        logging.info(f"Relevance Classes: {self.relevance_classes_}")
        # Simple boolean vector where relevan features are regarded as one set (1 relevant, 0 irrelevant)
        self.support_ = self.relevance_classes_ > 0

        # self.feature_importances_ = utils.compute_importances(importances)[1] # Take mean
        # self.interval_ = utils.emulate_intervals(importances)
    
    def score(self,X, y):
        return self.rfmodel.score(X,y)

    def predict(self,X):
        return self.rfmodel.predict(X)