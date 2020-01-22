import lightgbm
import numpy as np
import sklearn.feature_selection as fs
from boruta import BorutaPy
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
from sklearn.utils import check_random_state
from squamish.utils import reduced_data, perm_i_in_X
import logging

logging = logging.getLogger(__name__)


def get_relev_class_RFE(X, y, model, cv=5, random_state=None, params=None):
    rfc = fs.RFECV(model, cv=cv)
    rfc.fit(X, y)
    return rfc.support_.astype(int)


def get_relev_class_SFM(X, y, model):
    sfm = fs.SelectFromModel(model, prefit=True, threshold="2*mean")
    # rfc.fit(X, y)
    return sfm.get_support().astype(int)

class Model:
    def __init__(self):
        self.estimator = None
        self.fset_ = None
        self.random_state = None

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
        "num_leaves":32,
        "max_depth": 5,
        "boosting_type": "rf",
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "feature_fraction": 0.8,
        "subsample":None,
        "subsample_freq":None,
        "colsample_bytree":None,
        "importance_type": "gain",
        "verbose": -1,
    }

    def __init__(self, random_state=None, **params):
        if params is None:
            params = self.BEST_PARAMS
        self.random_state = check_random_state(random_state)
        
        self.estimator = lightgbm.LGBMClassifier(
            random_state=self.random_state.randint(1e6), **params)
        self.fset_ = None

    def fset(self, X, y, stats):
        if hasattr(self.estimator, "feature_importances_"):
            if self.fset_ is None:
                # d = X.shape[1]
                # if d > d_thresh:
                #     self.fset_ = get_relev_class_SFM(X,y,self.estimator)
                # else:
                #     self.fset_ = get_relev_class_SFM(X,y,self.estimator)
                #     logging.info(f"SFM SET: {self.fset_}")
                #     self.fset_ = get_relev_class_RFE(X, y, self.estimator)
                #     logging.info(f"RFE SET: {self.fset_}")
                lo_bound, hi_bound = stats.shadow_stat
                bigger_than_shadow_bound = self.estimator.feature_importances_ > hi_bound
                self.fset_ = bigger_than_shadow_bound.astype(int)
                # logging.debug(f"Shadow SET: {self.fset_}")
                # self.fset_ = get_relev_class_RFE(X, y, self.estimator)
                #logging.info(f"RFE SET: {self.fset_}")
            return self.fset_
        else:
            raise Exception("Model has no fset_ yet. Not fitted?")

    def redscore(self, X, y, c):
        X_c = reduced_data(X, c)
        X_c = scale(X_c)
        self.estimator.fit(X_c, y)
        return self.score(X_c, y)

    def score_with_i_permuted(self, X, y, i, random_state):
        X_c = perm_i_in_X(X, i, random_state)
        X_c = scale(X_c)
        self.estimator.fit(X_c, y)
        return self.score(X_c, y)
    
    def predict(self, X):
        return self.estimator.predict(X)

class MyBoruta(Model):
    BEST_PARAMS_BORUTA = {
        "num_leaves":32,
        "max_depth": 5,
        "boosting_type": "rf",
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "feature_fraction": 0.1,  # We force low feature fraction to reduce overshadowing of better redundant features
        "b_perc": 100,
        "subsample":None,
        "subsample_freq":None,
        "verbose": -1,
        "colsample_bytree":None,
        "b_n_estimators": "auto",
        "b_alpha": 0.01,
        "b_max_iter": 100,
        "importance_type": "gain",
    }

    def __init__(self, random_state=None, params=None):
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
        self.random_state = check_random_state(random_state)
        lm = lightgbm.LGBMClassifier(random_state=self.random_state.randint(1e6),
                                     **tree_params, )
        self.estimator = BorutaPy(lm, verbose=0, random_state=self.random_state,
                                  **boruta_params)

    def fset(self, X, y):
        if hasattr(self.estimator, "support_"):
            return self.estimator.support_
        else:
            raise Exception("Model has no fset_ yet. Not fitted?")

    def cvscore(self, X, y, cv=5):
        estimator = self.estimator.estimator  # use inner RF
        return np.mean(cross_val_score(estimator, X, y, cv=cv))
