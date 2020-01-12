from squamish.utils import reduced_data, cv_score
import lightgbm
import numpy as np
import pandas as pd
import sklearn.feature_selection as fs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.preprocessing import scale
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

import boruta
from boruta import BorutaPy


def get_relev_class_RFE(X, y, model, cv=5, random_state=None, params=None):
    rfc = fs.RFECV(model, cv=cv)
    rfc.fit(X, y)
    return rfc.support_.astype(int)


class Model:
    def __init__(self, model, **params):
        self.estimator = model(**params)
        self.fset_ = None

    def fit(self, X, y):
        X = scale(X)
        self.estimator.fit(X, y)
        return self

    def score(self, X, y, cv=5):
        return self.estimator.score(X, y)

    def cvscore(self, X, y, cv=5):
        return np.mean(cross_val_score(self.estimator, X, y, cv=cv))

    def fset(self, X, y):
        raise NotImplementedError

    def importances(self):
        if hasattr(self.estimator, "feature_importances_"):
            return self.estimator.feature_importances_
        else:
            raise Exception("no importances, no model fitted yet?")


class RF(Model):
    BEST_PARAMS = {
        "max_depth": 5,
        "boosting_type": "rf",
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "feature_fraction": 0.1,
        # old setting?
        #"feature_fraction": 0.5,
        "importance_type": "gain",
        "verbose": 0,
    }

    def __init__(self, params=None):
        if params is None:
            params = self.BEST_PARAMS
        super().__init__(lightgbm.LGBMClassifier, **params)

    def fset(self, X, y):
        if hasattr(self.estimator, "feature_importances_"):
            if self.fset_ is None:
                self.fset_ = get_relev_class_RFE(X, y, self.estimator)
            return self.fset_
        else:
            raise Exception("Model has no fset_ yet. Not fitted?")

    def redscore(self, X, y, c):
        X_c = reduced_data(X, c)
        X_c = scale(X_c)
        self.estimator.fit(X_c, y)
        return self.score(X_c, y)


class MyBoruta(Model):
    BEST_PARAMS_BORUTA = {
        "max_depth": 5,
        "boosting_type": "rf",
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "feature_fraction": 0.1,
        "b_perc": 100,
        "verbose": 0,
        "verbose_eval": False,
        "b_n_estimators": "auto",
        "b_alpha": 0.01,
        "b_max_iter": 100,
        "importance_type": "gain",
    }

    def __init__(self, params=None):
        if params is None:
            tree_params = {
                k: v
                for k, v in self.BEST_PARAMS_BORUTA.items()
                if not k.startswith("b_")
            }
            boruta_params = {
                k[2:]: v
                for k, v in self.BEST_PARAMS_BORUTA.items()
                if k.startswith("b_")
            }
        else:
            tree_params = None
            boruta_params = None
        print(tree_params)
        lm = lightgbm.LGBMClassifier(**tree_params)
        self.estimator = BorutaPy(lm, verbose=0, random_state=1, **boruta_params)

    def fset(self, X, y):
        if hasattr(self.estimator, "support_"):
            return self.estimator.support_
        else:
            raise Exception("Model has no fset_ yet. Not fitted?")

    def cvscore(self, X, y, cv=5):
        estimator = self.estimator.estimator  # use inner RF
        return np.mean(cross_val_score(estimator, X, y, cv=cv))


def fset_and_score(model: Model, X, y, params=None):
    m = model(params).fit(X, y)
    score = m.cvscore(X, y)
    fset = m.fset(X, y)
    fset = np.where(fset)
    return fset[0], score
