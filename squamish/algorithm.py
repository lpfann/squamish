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

def remove_F(f, MR, W):
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
    scores = {"MR": score_on_MR, "AR": score_on_AR, "MR+W": score_on_MR_and_W}

    for k, sc in scores.items():
        print(f"{k} has score {sc}")


    # Get Statistic
    X_allinformative = reduced_data(X, MR_and_W)
    X_allinformative = scale(X_allinformative)
    sig_bounds = get_significance_bounds(model,X_allinformative,y)
    print(f"sig bounds: {sig_bounds}")
    
    diffs = np.zeros(len(MR))
    imps = np.zeros((len(MR), X.shape[1]))
    for i, f in enumerate(MR):
        # Remove feature f from MR u W
        fset_without_f = remove_F(f, MR, W)
        print(fset_without_f)
        # check score if f is removed
        score_without_f = model.redscore(X, y, fset_without_f)

        # imps[i,C] = imps_c
        # imps[i,i] = np.median(imps_c) # Replace current importance for feature f with median as neutral element
        diffs[i] = score_on_MR_and_W - score_without_f  # Record score when f is missing

        print(f"score without {f} is {score_without_f:.3}-> ", end="")

        # Test if value lies in acceptance range of null distribution 
        # i.e. no signif. change compared to perm. feature
        # __We only check lower dist bound for worsening score when f is removed -> Strong relevant
        if score_without_f < sig_bounds[0]: 
            print(f"S")
            S.append(f)
        else:
            print(f"W")
            W.append(f)

    return S, W, diffs, imps
