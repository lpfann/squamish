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

def is_significant_score_deviation(score_without_f,null_distribution):
        # check score if f is removed
        print(f"removal_score:{score_without_f:.3}-> ", end="")
        # Test if value lies in acceptance range of null distribution
        # i.e. no signif. change compared to perm. feature
        # __We only check lower dist bound for worsening score when f is removed -> Strong relevant
        if score_without_f < null_distribution[0]:
            print(f"S")
            return True
        else:
            print(f"W")
            return False
        
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
    for f in MR:
        print("-------------------")
        print(f"Feature f:{f}")
        # Remove feature f from MR u W
        fset_without_f = get_combined_set_without_f(f, MR, W)

        # Determine Relevance class by checking score without feature f
        score_without_f = model.redscore(X,y, fset_without_f)
        significant = is_significant_score_deviation(score_without_f,score_bounds)
        if significant:
            S.append(f)
        else:
            W.append(f)

        #
        # Record Importances with this subset of features
        if not significant:
            ix_and_imps = zip(fset_without_f,model.importances())
            related[f] = check_related(ix_and_imps,imp_bounds_list)

        
    print("Related:",related)
    return S, W, imps, normal_imps, imp_bounds_list


def check_related(importances_other, importances_null_bounds):
    cands = []
    for f_ix,imp in importances_other:
        lo,hi = importances_null_bounds[f_ix]
        if lo <= imp <= hi:
            # No change in relation to null dist
            continue
        else:
            print(f_ix, lo, imp, hi)
            cands.append(f_ix)
    return cands
