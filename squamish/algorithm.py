from copy import copy

import numpy as np
from sklearn.preprocessing import scale

from squamish.models import RF
from squamish.stat import Stats
from squamish.utils import reduced_data
import logging

logger = logging.getLogger(__name__)


def combine_sets(A, B):
    C = np.union1d(A, B)  # Combine with weakly relevant features
    C = np.sort(C).astype(int)
    return C


def set_without_f(A, f):
    C = np.setdiff1d(A, f)  # Remove f from set
    return C.astype(int)


class FeatureSorter:
    PARAMS = {
        # "max_depth": 5,
        "boosting_type": "rf",
        "bagging_fraction": 0.632,
        "bagging_freq": 1,
        "importance_type": "gain",
        "subsample": None,
        "subsample_freq": None,
        "colsample_bytree": None,
        "verbose": -1,
    }
    # SPARSE_PARAMS = copy(PARAMS)
    # SPARSE_PARAMS["feature_fraction"] = 1
    DENSE_PARAMS = copy(PARAMS)
    DENSE_PARAMS["feature_fraction"] = 0.1

    def __init__(self, X, y, MR, AR, random_state, statistics, n_jobs=-1, debug=False):
        self.n_jobs = n_jobs
        if debug:
            logger.setLevel(logging.DEBUG)
        self.random_state = random_state
        self.X = X
        self.y = y
        self.MR = MR
        self.AR = AR
        self.S = []
        self.W = list(np.setdiff1d(AR, MR))
        self.MR_and_W = combine_sets(MR, self.W)
        self.X_onlyrelevant = reduced_data(X, self.MR_and_W)
        self.X_onlyrelevant = scale(self.X_onlyrelevant)
        logger.debug(f"predetermined weakly {self.W}")

        self.model = RF(
            random_state=self.random_state, n_jobs=self.n_jobs, **self.DENSE_PARAMS
        )
        # print_scores_on_sets(AR, MR, self.MR_and_W, X, self.model, y)

        self.score_bounds = statistics.score_stat
        imp_bounds_list = statistics.imp_stat

        logger.debug(f"score bounds: {self.score_bounds}")
        self.fimp_bounds = {}
        for f_ix, imp in zip(self.MR_and_W, imp_bounds_list):
            self.fimp_bounds[f_ix] = imp

    def is_significant_score_deviation(self, score_without_f):
        # check score if f is removed
        # Test if value lies in acceptance range of null distribution
        # i.e. no signif. change compared to perm. feature
        # __We only check lower dist bound for worsening score when f is removed -> Strong relevant
        if score_without_f < self.score_bounds[0]:
            logger.debug(f"removal_score:{score_without_f:.3}-> S")
            return True
        else:
            logger.debug(f"removal_score:{score_without_f:.3}-> W")
            return False

    def check_each_feature(self):
        self.related = {}
        self.synergies = {}

        if len(self.MR_and_W) == 1:
            # Only one feature, which should be str. relevant
            self.S = self.MR
            logger.debug("Only one feature")
            return

        for f in self.MR:
            logger.debug(f"------------------- Feature f:{f}")

            # Remove feature f from MR u W
            fset_without_f = set_without_f(self.MR_and_W, f)

            # Determine Relevance class by checking score without feature f
            rel_f_ix = np.where(self.MR_and_W == f)[0]
            score_without_f = self.model.score_with_i_permuted(
                self.X_onlyrelevant, self.y, rel_f_ix, random_state=self.random_state
            )
            logger.debug(f"score without {f}: {score_without_f}")
            significant = self.is_significant_score_deviation(score_without_f)
            if significant:
                self.S.append(f)
            else:
                self.W.append(f)

            # Record Importances with this subset of features
            if not significant:
                finder = RelationFinder(
                    [f],
                    (self.X_onlyrelevant, self.y),
                    self.model,
                    self.fimp_bounds,
                    fset_without_f,
                )
                relatives = finder.check_for_redundancies()
                # relatives.remove(f)  # Remove self
                self.related[f] = relatives
            # else:
            #     relatives = finder.check_for_synergies()
            #     if len(relatives)>0:
            #         self.synergies[f] = relatives

        self.related = filter_strongly(self.related, self.S)
        logger.debug(f"Related: {self.related}")
        # print("Synergies:", self.synergies)
        logger.debug(f"S: {self.S}")
        logger.debug(f"W: {self.W}")


def print_scores_on_sets(AR, MR, MR_and_W, X, model, y):
    score_on_MR = model.redscore(X, y, MR)
    score_on_AR = model.redscore(X, y, AR)
    score_on_MR_and_W = model.redscore(X, y, MR_and_W)
    normal_imps = model.importances()
    logger.debug(f"normal_imps:{normal_imps}")
    logger.debug(f"length MR and W {len(MR_and_W)}")
    scores = {"MR": score_on_MR, "AR": score_on_AR, "MR+W": score_on_MR_and_W}
    for k, sc in scores.items():
        logger.debug(f"{k} has score {sc}")


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
