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

logger = logging.getLogger(__name__)


class Main(BaseEstimator):
    def __init__(
        self,
        problem="classification",
        n_resampling=50,
        fpr=1e-6,
        random_state=None,
        n_jobs=-1,
        debug=True,
    ):
        self.n_jobs = n_jobs
        self.problem = problem
        self.n_resampling = n_resampling
        self.fpr = fpr
        self.random_state = check_random_state(random_state)
        self.debug = debug
        if debug:
            logger.setLevel(logging.DEBUG)

    def _get_support_mask(self):
        return self.support_

    def fit(self, X, y):
        X = scale(X)
        n, d = X.shape

        # All relevant set using Boruta
        m = models.MyBoruta(random_state=self.random_state, n_jobs=self.n_jobs).fit(
            X, y
        )
        # bor_score = m.cvscore(X, y)
        fset = m.fset(X, y)
        AR = np.where(fset)[0]

        # Fit a simple Random Forest to get a minimal feature subset
        m = models.RF(random_state=self.random_state, n_jobs=self.n_jobs).fit(X, y)
        self.score_ = m.cvscore(X, y)
        logger.debug(f"RF score {self.score_}")
        logger.debug(f"importances {m.estimator.feature_importances_}")
        self.rfmodel = deepcopy(m)

        self.stat_ = Stats(
            m,
            X,
            y,
            n_resampling=self.n_resampling,
            fpr=self.fpr,
            random_state=self.random_state,
            check_importances=True,
        )
        fset = self.rfmodel.fset(X, y, self.stat_)
        fset = np.where(fset)
        MR = fset[0]

        logger.debug(f"Features from Boruta: {AR}")
        logger.debug(f"Features from RF: {MR}")

        # Sort features iteratively into strongly (S) and weakly (W) sets
        self.fsorter = FeatureSorter(
            X, y, MR, AR, self.random_state, self.stat_, n_jobs=self.n_jobs
        )
        self.fsorter.check_each_feature()

        # Turn index sets into support vector
        # (2 strong relevant,1 weak relevant, 0 irrelevant)
        all_rel_support = create_support_AR(d, self.fsorter.S, self.fsorter.W)
        self.relevance_classes_ = all_rel_support
        logger.info(f"Relevance Classes: {self.relevance_classes_}")

        # Simple boolean vector where relevan features are regarded as one set (1 relevant, 0 irrelevant)
        self.support_ = self.relevance_classes_ > 0


    def score(self, X, y):
        return self.rfmodel.score(X, y)

    def predict(self, X):
        return self.rfmodel.predict(X)
