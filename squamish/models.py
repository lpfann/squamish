import logging

import lightgbm
from boruta import BorutaPy
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state

from squamish.utils import reduced_data, perm_i_in_X

logging = logging.getLogger(__name__)


def get_RF_class(problem_type):
    if problem_type == "classification":
        return lightgbm.LGBMClassifier
    if problem_type == "regression":
        return lightgbm.LGBMRegressor
    if problem_type == "ranking":
        return lightgbm.LGBMRanker

    raise Exception(
        "Problem Type does not exist. Try 'classification', 'regression' or 'raking'."
    )


class RF(BaseEstimator):
    def __init__(
        self,
        problem_type,
        max_depth=5,
        feature_fraction=0.8,
        random_state=None,
        n_jobs=-1,
        **params,
    ):
        self.problem_type = problem_type
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs
        self.fset_ = None
        num_leaves = 2 ** max_depth
        model = get_RF_class(self.problem_type)
        self.estimator = model(
            boosting_type="rf",
            max_depth=max_depth,
            num_leaves=num_leaves,
            feature_fraction=feature_fraction,
            random_state=self.random_state.randint(1e6),
            n_jobs=self.n_jobs,
            bagging_fraction=0.632,
            bagging_freq=1,
            subsample=None,
            subsample_freq=None,
            verbose=-1,
            colsample_bytree=None,
            importance_type="gain",
            **params,
        )
        self.fset_ = None

    def fset(self, stats):
        if hasattr(self.estimator, "feature_importances_"):
            if self.fset_ is None:
                lo_bound, hi_bound = stats.shadow_stat
                bigger_than_shadow_bound = (
                    self.estimator.feature_importances_ > hi_bound
                )
                self.fset_ = bigger_than_shadow_bound.astype(int)
            return self.fset_
        else:
            raise Exception("Model has no fset_ yet. Not fitted?")

    def score_on_subset(self, X, y, featureset):
        X_c = reduced_data(X, featureset)
        self.estimator.fit(X_c, y)
        return self.score(X_c, y)

    def score_with_i_permuted(self, X, y, i, random_state):
        X_c = perm_i_in_X(X, i, random_state)
        self.estimator.fit(X_c, y)
        return self.score(X_c, y)

    def predict(self, X):
        return self.estimator.predict(X)

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def score(self, X, y):
        return self.estimator.score(X, y)

    def importances(self):
        if hasattr(self.estimator, "feature_importances_"):
            return self.estimator.feature_importances_
        else:
            raise Exception("no importances, no model fitted yet?")


class MyBoruta(BaseEstimator):
    def __init__(
        self,
        problem_type,
        random_state=None,
        n_jobs=-1,
        feature_fraction=0.1,
        max_depth=5,
        perc=70,
        n_estimators="auto",
        alpha=0.01,
        max_iter=100,
        **params,
    ):
        self.problem_type = problem_type
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs
        self.fset_ = None
        self.feature_fraction = feature_fraction
        self.max_depth = max_depth
        self.perc = perc
        self.n_estimators = n_estimators
        self.alpha = alpha
        self.max_iter = max_iter

        num_leaves = 2 ** max_depth
        RfModel = get_RF_class(self.problem_type)
        rfmodel = RfModel(
            random_state=self.random_state.randint(1e6),
            n_jobs=self.n_jobs,
            boosting_type="rf",
            max_depth=max_depth,
            num_leaves=num_leaves,
            feature_fraction=feature_fraction,
            bagging_fraction=0.632,
            bagging_freq=1,
            subsample=None,
            subsample_freq=None,
            verbose=-1,
            colsample_bytree=None,
            importance_type="gain",
            **params,
        )
        self.estimator = BorutaPy(
            rfmodel,
            verbose=0,
            random_state=self.random_state,
            perc=perc,
            n_estimators=n_estimators,
            alpha=alpha,
            max_iter=max_iter,
            **params,
        )

    def fset(self):
        if hasattr(self.estimator, "support_"):
            return self.estimator.support_
        else:
            raise Exception("Model has no fset_ yet. Not fitted?")

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def score(self, X, y):
        return self.estimator.score(X, y)

    def importances(self):
        if hasattr(self.estimator, "feature_importances_"):
            return self.estimator.feature_importances_
        else:
            raise Exception("no importances, no model fitted yet?")
