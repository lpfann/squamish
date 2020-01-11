from squamish.models import RF, fset_and_score
import numpy as np
import lightgbm
import numpy as np
import pandas as pd
import sklearn.feature_selection as fs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.preprocessing import scale
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import squamish.utils
from scipy import stats
from squamish.stat import get_significance_bounds
from squamish.utils import reduced_data
from copy import copy, deepcopy


def get_combined_set_without_f(f, MR, W):
    C = np.setdiff1d(MR, f)  # Remove f from minimal set
    C = np.union1d(C, W)  # Combine with weakly relevant features
    C = np.sort(C).astype(int)
    return C


def sort_features(X, y, MR, AR):

    S = []
    W = list(np.setdiff1d(AR, MR))
    MR_and_W = np.union1d(MR, W)
    print(f"predetermined weakly {W}")

    model = RF()

    score_on_MR = model.redscore(X, y, MR)
    score_on_AR = model.redscore(X, y, AR)
    score_on_MR_and_W = model.redscore(X, y, MR_and_W)
    normal_imps = model.importances()
    print("length MR and W", len(MR_and_W))
    scores = {"MR": score_on_MR, "AR": score_on_AR, "MR+W": score_on_MR_and_W}

    for k, sc in scores.items():
        print(f"{k} has score {sc}")

    # Get Statistic
    # TODO: we only consider relevant here, check all features??
    X_allinformative = reduced_data(X, MR_and_W)
    X_allinformative = scale(X_allinformative)
    score_bounds, imp_bounds_list = get_significance_bounds(
        model, X_allinformative, y, importances=True
    )
    #print(f"sig bounds: {score_bounds}")

    related = {}

    imps = np.zeros((len(MR), X.shape[1]))

    # TODO: Iteration over M u RR
    for i, f in enumerate(MR):
        print("-------------------")
        print(f"Feature i:{i} f:{f}")
        # Remove feature f from MR u W
        fset_without_f = get_combined_set_without_f(f, MR, W)

        # check score if f is removed
        score_without_f = model.redscore(X, y, fset_without_f)
        print(f"removal_score:{score_without_f:.3}-> ", end="")
        # Test if value lies in acceptance range of null distribution
        # i.e. no signif. change compared to perm. feature
        # __We only check lower dist bound for worsening score when f is removed -> Strong relevant
        if score_without_f < score_bounds[0]:
            print(f"S")
            S.append(f)
        else:
            print(f"W")
            W.append(f)

        #
        #
        # Record Importances with this subset of features
        imps_without_f = model.importances()
        imps[i, fset_without_f] = imps_without_f
        # Replace current importance for feature f with median as neutral element
        # imps[i,i] = np.median(imps_without_f)
        imps[
            i, i
        ] = (
            np.nan
        )  # Replace current importance for feature f with median as neutral element
        # TODO: correct checks?
        related[f] = check_related(imps_without_f, imp_bounds_list)
        print(related[f])
        
    print("Related:",related)
    return S, W, imps, normal_imps, imp_bounds_list


def check_related(importances_i, imp_bounds):
    cands = []
    for j in range(len(importances_i)):

        lo,hi = imp_bounds[j]
        imp = importances_i[j]
        if lo <= imp <= hi:
            # No change in relation to null dist
            continue
        else:
            print(j, lo, imp, hi)
            cands.append(j)
    return cands
