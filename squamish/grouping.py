import scipy
import umap
import hdbscan

import warnings
from abc import abstractmethod

import numpy as np
import math

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection.base import SelectorMixin
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import make_scorer
from sklearn.externals.joblib import Parallel, delayed


def distance(u, v):
    """
    Distance measure custom made for feature comparison.

    Parameters
    ----------
    u: first feature
    v: second feature

    Returns
    -------

    """
    u = np.asarray(u)
    v = np.asarray(v)
    # Euclidean differences
    diff = (u - v) ** 2
    # Nullify pairwise contribution
    diff[u == 0] = 0
    diff[v == 0] = 0
    return np.sqrt(np.sum(diff))


def grouping(change_matrix, cutoff_threshold=0.55, method="single"):
    """ Find feature clusters based on observed variance when removing features

        Parameters
        ----------
        change_matrix: np.array
            vector of feature importance changes in relation to model with all features included
        cutoff_threshold : float, optional
            Cutoff value for the flat clustering step; decides at which height in the dendrogram the cut is made to determine groups.
        method : str, optional
            Linkage method used in the hierarchical clustering.

        Returns
        -------
        self
        """

    d = len(change_matrix)

    

    # Calculate similarity using custom measure
    dist_mat = scipy.spatial.distance.pdist(feature_points, metric=distance)

    # Single Linkage clustering
    # link = linkage(dist_mat, method="single")

    link = linkage(dist_mat, method=method, optimal_ordering=True)

    # Set cutoff at which threshold the linkage gets flattened (clustering)
    RATIO = cutoff_threshold
    threshold = RATIO * np.max(link[:, 2])  # max of branch lengths (distances)
    feature_clustering = fcluster(link, threshold, criterion="distance")

    self.feature_clusters_, self.linkage_ = feature_clustering, link

    return self.feature_clusters_


def umap(self, n_neighbors=2, n_components=2, min_dist=0.1):
    if self.relevance_variance is None:
        print("Use grouping() first to compute relevance_variance")
        return

    um = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=min_dist,
        metric=distance,
    )
    embedding = um.fit_transform(self.relevance_variance)
    self.relevance_var_embedding_ = embedding
    return embedding


def grouping_umap(
    self,
    only_relevant=False,
    min_group_size=2,
    umap_n_neighbors=2,
    umap_n_components=2,
    umap_min_dist=0.1,
):

    self._umap_embedding = self.umap(
        n_neighbors=umap_n_neighbors,
        n_components=umap_n_components,
        min_dist=umap_min_dist,
    )

    if only_relevant:
        embedding = self._umap_embedding[self.allrel_prediction_]
    else:
        embedding = self._umap_embedding

    hdb = hdbscan.HDBSCAN(min_cluster_size=min_group_size)
    hdb.fit(embedding)
    labels = np.full_like(self.allrel_prediction_, -2, dtype=int)

    if only_relevant:
        labels[self.allrel_prediction_] = hdb.labels_
    else:
        labels = hdb.labels_

    self.group_labels_ = labels

    return labels


def grouping_hdbscan(self, only_relevant=False, min_group_size=2):

    if only_relevant:
        data = self.relevance_variance[self.allrel_prediction_]
    else:
        data = self.relevance_variance

    hdb = hdbscan.HDBSCAN(min_cluster_size=min_group_size, metric=distance)
    hdb.fit(data)
    labels = np.full_like(self.allrel_prediction_, -2, dtype=int)

    if only_relevant:
        labels[self.allrel_prediction_] = hdb.labels_
    else:
        labels = hdb.labels_

    self.group_labels_ = labels

    return labels

