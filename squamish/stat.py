from scipy import stats
import numpy as np


def _create_probe_statistic(probe_values, fpr):
    # Create prediction interval statistics based on randomly permutated probe features (based on real features)
    n = len(probe_values)

    if n == 1:
        val = probe_values[0]
        low_t = val
        up_t = val
    else:
        probe_values = np.asarray(probe_values)
        mean = probe_values.mean()
        s = probe_values.std()
        low_t = mean + stats.t(df=n - 1).ppf(fpr) * s * np.sqrt(1 + (1 / n))
        up_t = mean - stats.t(df=n - 1).ppf(fpr) * s * np.sqrt(1 + (1 / n))
    return low_t, up_t



def add_NFeature_to_X(X, feature_i, random_state):
    X_copy = np.copy(X)
    # Permute selected feature
    permutated_feature = random_state.permutation(X_copy[:, feature_i])

    # Append permutation to dataset
    X_copy = np.hstack([X_copy, permutated_feature[:, None]])
    return X_copy


def _perm_scores(model, X, y, n_resampling, importances=False,random_state=None):
    if random_state is None:
        random_state = np.random.RandomState()

    # Random sample n_resampling shadow features by permuting real features
    random_choice = random_state.choice(a=X.shape[1], size=n_resampling)

    # Instantiate objects
    for di in random_choice:
        X_NF = add_NFeature_to_X(X, di, random_state)
        model.fit(X_NF, y)
        score = model.score(X_NF, y)
        if importances:
            # Only get values of non-permutated features
            imps = model.importances()[:-1]
            yield score, imps
        else:
            yield score,


def get_significance_bounds(model, X, y, n_resampling=40, fpr=1e-4, importances=False,random_state=None):
    sample_tuples = list(_perm_scores(model, X, y, n_resampling,importances=importances,random_state=random_state))

    scores = [sc[0] for sc in sample_tuples]
    score_stat = _create_probe_statistic(scores, fpr)
    if importances:
        imps = np.array([sc[1] for sc in sample_tuples]).T
        # Iterate over columns of matrix, column corresponds to all samples for 1 feature
        imp_stat = [_create_probe_statistic(col, fpr) for col in imps]

        return score_stat, imp_stat
    else:
        return score_stat