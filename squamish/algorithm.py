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
    synergies = {}
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
        finder = FindRelated([f], (X,y),model,imp_bounds_list)
        if not significant:
            relatives = finder.get_relatives(
                f, fset_without_f, prefit=True
            )
            relatives.remove(f) # Remove self
            related[f] = relatives
        else:
            relatives = finder.check_for_synergies(fset_without_f)
            synergies[f] = relatives

    related = filter_strongly(related,S)

    print("Related:", related)
    print("Synergies:", synergies)
    return S, W, imps, normal_imps, imp_bounds_list

def filter_strongly(related,known_strongly):
    for k,v in related.items():
        related[k] = list(filter(lambda x: x not in known_strongly,v))
    return related
class FindRelated:
    def __init__(self,seen, data, model, importances_null_bounds):
        self.seen = seen
        self.data = data
        self.model = model
        self.importances_null_bounds = importances_null_bounds


    def check_for_synergies(self,fset_without_f):
        # Get importances together with feature index
        importances = zip(fset_without_f, self.model.importances())
        # Find significantly different behaving features in this model
        relatives = self.features_with_significant_negative_change(importances)
        return relatives

    def get_relatives(self, f, fset, prefit=False):
        """
            Recursively check and remove features which are related.
            We keep a state in the object to save already seen features to remove redundancies
        """
        print(f"feature {f} fset:{fset}")
        
        if f not in self.seen:
            # add feature to seen list
            self.seen = combine_sets(self.seen, [f])

        if prefit:
            # Prefit only in first call to save one model fit
            fset_without_f = fset
        else:
            # Fit model without feature f
            fset_without_f = set_without_f(fset, f)
            self.model.redscore(*self.data, fset_without_f)

        # Get importances together with feature index
        importances = zip(fset_without_f, self.model.importances())
        # Find significantly different behaving features in this model
        relatives = self.features_with_significant_positive_change(importances)
        # If features where already handled earlier, filter out
        unseen = list(filter(lambda x: x not in self.seen, relatives))

        # Create list of child features which are related
        rels = []
        rels.append(f)
        # Check unseen features with an importance value which changed significantly
        for fu in unseen:
            if len(fset_without_f)==1:
                rels.append(fu)
            else:
                # Recursion into feature fu 
                child_rel = self.get_relatives(
                    fu, fset_without_f
                )

                # Return child relative list and add it to this list
                rels.extend(child_rel)

        return list(np.unique(rels))

    def features_with_significant_change(self, importances_other):
        cands = []
        for f_ix, imp in importances_other:
            # Check null distribution of pristine model without deleted features
            lo, hi = self.importances_null_bounds[f_ix]
            if hi < imp or imp < lo:
                cands.append(f_ix)
        return cands

    def features_with_significant_negative_change(self, importances_other):
        cands = []
        for f_ix, imp in importances_other:
            lo, _ = self.importances_null_bounds[f_ix]
            if imp < lo:
                cands.append(f_ix)
        return cands

    def features_with_significant_positive_change(self, importances_other):
        cands = []
        for f_ix, imp in importances_other:
            _, hi = self.importances_null_bounds[f_ix]
            if imp > hi:
                cands.append(f_ix)
        return cands
