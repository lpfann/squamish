import lightgbm
import numpy as np
import pandas as pd
import sklearn.feature_selection as fs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.preprocessing import scale

import boruta
from boruta import BorutaPy


def get_truth_AR(d, informative, redundant):
    truth = (
        [2] * (informative) + [1] * (redundant) + [0] * (d - (informative + redundant))
    )
    return truth


def get_truth(d, informative, redundant):
    truth = [True] * (informative + redundant) + [False] * (
        d - (informative + redundant)
    )
    return truth


def get_scores(support, truth):
    return [
        (sc.__name__, sc(truth, support))
        for sc in [precision_score, recall_score, f1_score]
    ]


def get_fs(estimator, X=None, y=None):
    fset = fs.SelectFromModel(
        prefit=True, estimator=estimator, threshold="mean"
    ).get_support()
    # fset = fs.RFECV(estimator=estimator,cv=3).fit(X,y).get_support()
    return fset


def RF(X, y, params=None):
    if params is None:
        lm = lightgbm.LGBMClassifier(
            max_depth=3, boosting_type="rf", bagging_fraction=0.632, bagging_freq=1
        )
    else:
        lm = lightgbm.LGBMClassifier(**params)
    lm.fit(X, y)
    return lm


def get_MR(X, y, params=None):
    lm = RF(X, y, params)
    # todo: welcher score?
    score = lm.score(X, y)
    fset = get_fs(lm, X, y)
    fset = np.where(fset)
    return fset[0], score


def get_AR_params(X, y, params):
    print(params)
    tree_params = {k: v for k, v in params.items() if not k.startswith("b_")}
    boruta_params = {k[2:]: v for k, v in params.items() if k.startswith("b_")}

    lm = lightgbm.LGBMClassifier(**tree_params)
    feat_selector = BorutaPy(lm, verbose=0, random_state=1, **boruta_params)
    feat_selector.fit(X, y)

    fset = feat_selector.support_

    return fset


def create_support_AR(d, S, W):
    sup = np.zeros(d)
    sup[S] = 2
    sup[W] = 1
    return sup.astype(int)


def cv_score(X, y, model, cv=20):
    return np.mean(cross_val_score(model, X, y, cv=cv))


def train_and_score(X, y, params=None):
    rf = RF(X, y, params)
    return cv_score(X, y, rf)


def score_with_feature_set(X, y, featureset, params=None):
    featureset = np.ravel(featureset).astype(int)
    return train_and_score(X[:, featureset], y, params)


def sort_features(X, y, MR, AR, params_rf=None, params_boost=None):
    S = []
    W = list(np.setdiff1d(AR, MR))
    print(f"predetermined weakly {W}")

    score_on_MR = score_with_feature_set(X, y, MR, params_boost)
    score_on_AR = score_with_feature_set(X, y, AR, params_boost)
    MR_and_W = np.union1d(MR, W)
    score_on_MR_and_W = score_with_feature_set(X, y, MR_and_W, params_boost)
    scores = {"MR": score_on_MR, "AR": score_on_AR, "MR+W": score_on_MR_and_W}
    for k, sc in scores.items():
        print(f"{k} has score {sc}")

    for f in MR:

        C = np.setdiff1d(MR, f)  # Remove f from minimal set
        C = np.union1d(C, W)  # Combine with weakly relevant features

        score_c = score_with_feature_set(X, y, C, params_boost)
        print(f"score without {f} is {score_c:.3}-> ", end="")

        if score_c < score_on_MR_and_W:
            print(f"S")
            S.append(f)
        else:
            print(f"W")
            W.append(f)
    return S, W
