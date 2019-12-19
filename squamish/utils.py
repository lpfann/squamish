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


def create_support_AR(d, S, W):
    sup = np.zeros(d)
    sup[S] = 2
    sup[W] = 1
    return sup.astype(int)


def cv_score(X, y, model, cv=20):
    return np.mean(cross_val_score(model, X, y, cv=cv))


def reduced_data(X, featureset):
    featureset = np.ravel(featureset).astype(int)
    return X[:, featureset]


def compute_importances(recorded_importances):
    mins, median, maxs = np.percentile(recorded_importances, [0, 50, 100], axis=0)
    return mins, median, maxs


def emulate_intervals(recorded_importances):
    _, median, _ = compute_importances(recorded_importances)
    deviation = np.std(recorded_importances, axis=0)

    upper_bounds = median + deviation
    lower_bounds = median - deviation
    interval = np.zeros((len(median), 2))
    for i in range(len(median)):
        interval[i] = lower_bounds[i], upper_bounds[i]
    return interval
