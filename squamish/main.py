import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from squamish.utils import create_support_AR
from squamish.algorithm import FeatureSorter
from . import models
import logging
logging = logging.getLogger(__name__)

class Main(BaseEstimator, SelectorMixin):
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

        # All relevant set using boruta
        AR, bor_score = models.fset_and_score(models.MyBoruta, X, y,random_state=self.random_state)
        m = models.RF(random_state=self.random_state).fit(X, y)
        self.score_ = m.cvscore(X, y)
        logging.info(f"RF score {self.score_}")

        fset = m.fset(X, y)
        fset = np.where(fset)
        MR =  fset[0]
        self.rfmodel = m

        logging.info(f"Features from Boruta: {AR}")
        logging.info(f"Features from RF: {MR}")

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
    
    def score(self,X, y):
        return self.rfmodel.score(X,y)

    def predict(self,X):
        return self.rfmodel.predict(X)