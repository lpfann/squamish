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


class FeatureSorter:
    PARAMS = {
        "max_depth": 5,
        "boosting_type": "rf",
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "importance_type": "gain",
        "verbose": 0
    }
    #SPARSE_PARAMS = copy(PARAMS)
    #SPARSE_PARAMS["feature_fraction"] = 1
    DENSE_PARAMS = copy(PARAMS)
    DENSE_PARAMS["feature_fraction"] = 0.1

    def __init__(self, X, y, MR, AR,random_state):
        self.random_state = random_state
        self.X = X
        self.y = y
        self.MR = MR
        self.AR = AR
        self.S = []
        self.W = list(np.setdiff1d(AR, MR))
        self.MR_and_W = combine_sets(MR, self.W)
        print(f"predetermined weakly {self.W}")

        self.model = RF(params=self.DENSE_PARAMS)
        print_scores_on_sets(AR, MR, self.MR_and_W, X, self.model, y)

        self.create_null_stat(self.model, X,y)

    def create_null_stat(self,model, X, y):
        X_allinformative = reduced_data(X, self.MR_and_W)
        X_allinformative = scale(X_allinformative)
        self.score_bounds, imp_bounds_list = get_significance_bounds(
            model, X_allinformative, y, importances=True, random_state=self.random_state
        )
        print(f"score bounds: {self.score_bounds}")
        self.fimp_bounds = {}
        for f_ix, imp in zip(self.MR_and_W, imp_bounds_list):
            self.fimp_bounds[f_ix] = imp

    def check_each_feature(self):
        self.related = {}
        self.synergies = {}

        if len(self.MR_and_W)==1:
                # Only one feature, which should be str. relevant
                self.S = self.MR
                print("Only one feature")
                return

        for f in self.MR:
            print("-------------------")
            print(f"Feature f:{f}")

            # Remove feature f from MR u W
            fset_without_f = set_without_f(self.MR_and_W, f)

            # Determine Relevance class by checking score without feature f
            score_without_f = self.model.score_with_i_permuted(self.X, self.y, f,random_state=self.random_state)
            significant = is_significant_score_deviation(
                score_without_f, self.score_bounds
            )
            if significant:
                self.S.append(f)
            else:
                self.W.append(f)

            #
            # Record Importances with this subset of features
            if not significant:
                finder = RelationFinder([f], (self.X, self.y), self.model, self.fimp_bounds, fset_without_f)
                relatives = finder.check_for_redundancies()
                #relatives.remove(f)  # Remove self
                self.related[f] = relatives
            # else:
            #     relatives = finder.check_for_synergies()
            #     if len(relatives)>0:
            #         self.synergies[f] = relatives

        self.related = filter_strongly(self.related, self.S)
        print("Related:", self.related)
        #print("Synergies:", self.synergies)
        print("S:",self.S)
        print("W:",self.W)


def print_scores_on_sets(AR, MR, MR_and_W, X, model, y):
    score_on_MR = model.redscore(X, y, MR)
    score_on_AR = model.redscore(X, y, AR)
    score_on_MR_and_W = model.redscore(X, y, MR_and_W)
    normal_imps = model.importances()
    print(f"normal_imps:{normal_imps}")
    print("length MR and W", len(MR_and_W))
    scores = {"MR": score_on_MR, "AR": score_on_AR, "MR+W": score_on_MR_and_W}
    for k, sc in scores.items():
        print(f"{k} has score {sc}")


def filter_strongly(related, known_strongly):
    for k, v in related.items():
        related[k] = list(filter(lambda x: x not in known_strongly, v))
    return related


class RelationFinder:
    def __init__(self, seen, data, model, importances_null_bounds, used_feature_ids):
        self.seen = seen
        self.data = data
        self.model = model
        self.importances_null_bounds = importances_null_bounds
        self.feature_ids = used_feature_ids

    def check_for_redundancies(self):
        # Get importances together with feature index
        importances = zip(self.feature_ids, self.model.importances())
        # Find significantly different behaving features in this model
        relatives = self.features_with_significant_change(importances)
        return relatives

    # def get_relatives(self, f, fset, prefit=False):
    #     """
    #         Recursively check and remove features which are related.
    #         We keep a state in the object to save already seen features to remove redundancies
    #     """
    #     # print(f"feature {f} fset:{fset}")
    #
    #     if f not in self.seen:
    #         # add feature to seen list
    #         self.seen = combine_sets(self.seen, [f])
    #
    #     if prefit:
    #         # Prefit only in first call to save one model fit
    #         fset_without_f = fset
    #     else:
    #         # Fit model without feature f
    #         fset_without_f = set_without_f(fset, f)
    #         self.model.redscore(*self.data, fset_without_f)
    #
    #     # Get importances together with feature index
    #     importances = zip(fset_without_f, self.model.importances())
    #     # Find significantly different behaving features in this model
    #     relatives = self.features_with_significant_positive_change(importances)
    #     # If features where already handled earlier, filter out
    #     unseen = list(filter(lambda x: x not in self.seen, relatives))
    #
    #     # Create list of child features which are related
    #     rels = []
    #     rels.append(f)
    #     # Check unseen features with an importance value which changed significantly
    #     for fu in unseen:
    #         if len(fset_without_f) == 1:
    #             rels.append(fu)
    #         else:
    #             # Recursion into feature fu
    #             child_rel = self.get_relatives(fu, fset_without_f)
    #
    #             # Return child relative list and add it to this list
    #             rels.extend(child_rel)
    #
    #     return list(np.unique(rels))

    def features_with_significant_change(self, f_ids_with_imp):
        cands = []
        for f_ix, imp in f_ids_with_imp:
            lo, hi = self.importances_null_bounds[f_ix]
            if not lo <= imp <= hi:
                cands.append(f_ix)
        return cands

