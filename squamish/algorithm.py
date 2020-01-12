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


def combine_sets(A, B):
    C = np.union1d(A, B)  # Combine with weakly relevant features
    C = np.sort(C).astype(int)
    return C


def set_without_f(A, f):
    C = np.setdiff1d(A, f)  # Remove f from set
    return C.astype(int)


def is_significant_score_deviation(score_without_f, null_distribution):
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
    # print(f"sig bounds: {score_bounds}")

    related = {}

    imps = np.zeros((len(MR), X.shape[1]))

    for f in MR:
        print("-------------------")
        print(f"Feature f:{f}")

        # Remove feature f from MR u W
        fset_without_f = set_without_f(MR_and_W, f)

        # Determine Relevance class by checking score without feature f
        score_without_f = model.redscore(X, y, fset_without_f)
        significant = is_significant_score_deviation(score_without_f, score_bounds)
        if significant:
            S.append(f)
        else:
            W.append(f)

        #
        # Record Importances with this subset of features
        if not significant:
            finder = FindRelated(MR, (X,y),model,imp_bounds_list)
            relatives = finder.get_relatives(
                f, fset_without_f, prefit=True
            )
            related[f] = relatives

    print("Related:", related)
    return S, W, imps, normal_imps, imp_bounds_list


class FindRelated:
    def __init__(self,seen, data, model, importances_null_bounds):
        self.data = data
        self.model = model
        self.importances_null_bounds = importances_null_bounds
        self.seen = seen

    def get_relatives(self, f, fset, prefit=False):
        if prefit:
            fset_without_f = fset
        else:
            fset_without_f = set_without_f(fset, f)
            self.model.redscore(*self.data, fset_without_f)
        importances = zip(fset_without_f, self.model.importances())
        relatives = self.get_significant_imp_changes(importances)

        unseen = list(filter(lambda x: x not in self.seen, relatives))
        rels = []
        if f not in self.seen:
            rels.append(f)
            self.seen = combine_sets(self.seen, [f])
        print("current", f, "unseen", unseen)

        for fu in unseen:
            child_rel = self.get_relatives(
                fu, fset_without_f
            )
            rels.extend(child_rel)


        return rels

    def get_significant_imp_changes(self, importances_other):
        cands = []
        for f_ix, imp in importances_other:
            lo, hi = self.importances_null_bounds[f_ix]
            if lo <= imp <= hi:
                # No change in relation to null dist
                continue
            else:
                # print(f_ix, lo, imp, hi)
                cands.append(f_ix)
        return cands
