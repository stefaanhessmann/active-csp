import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from collections import defaultdict


__all__ = [
    "Clustering",
    "cluster_by_representations_old",
    "labels_to_clusters",
    "DBSCANClustering",
]


class Clustering:

    def get_clusters(self, representations):
        raise NotImplementedError


class DBSCANClustering(Clustering):

    def __init__(
        self,
        eps: float,
        min_sampled: int,
        min_different: int = 100,
        max_tries: int = 10,
        max_cluster_size: int = 2000,
        recluster_stable: bool = True,
    ):
        self.eps = eps
        self.min_sampled = min_sampled
        self.min_different = min_different
        self.max_tries = max_tries
        self.max_cluster_size = max_cluster_size
        self.recluster_stable = recluster_stable

    def get_clusters(
        self,
        representations: np.ndarray,
    ):
        # todo: check shapes
        # generate labels (-2 means this requires further evaluation)
        n_data = representations.shape[0]
        labels = np.ones(n_data) * -2

        for _ in range(self.max_tries):
            max_cluster_label = max(0, max(labels) + 1)

            # get ids, that require clustering
            ids = np.arange(0, n_data)
            ids_to_cluster = ids[labels == -2]

            # do clustering for all labels == -2
            scaled_representations = StandardScaler().fit_transform(
                representations[ids_to_cluster]
            )
            # todo: check leaf size
            new_labels = DBSCAN(eps=self.eps, min_samples=self.min_sampled).fit_predict(
                scaled_representations
            )

            for idx, label in zip(ids_to_cluster, new_labels):
                if label != -1:
                    label += max_cluster_label
                labels[idx] = label

            # split up big clusters
            for label in np.unique(labels):
                if label < 0:
                    continue
                if sum(labels == label) >= self.max_cluster_size:
                    labels[labels == label] = -2

            # check min number of structures, that are unlabeled or in different clusters
            all_labels = [label for label in np.unique(labels) if label >= 0]
            if len(all_labels) + sum(labels == -1) < self.min_different:
                labels[labels >= 0] = -2
            if self.recluster_stable and _ < 1:
                labels[labels >= 0] = -2

            # return if all structures are labeled
            if sum(labels == -2) == 0:
                return labels

        # set unclustered structures
        labels[labels == -2] = -1

        return labels


def cluster_by_representations_old(
    eps,
    min_sampled,
    representations,
    min_different=100,
    max_tries=10,
    max_cluster_size=500,
):
    n_data = representations.shape[0]
    labels = np.ones(n_data) * -2

    for _ in range(max_tries):
        max_cluster_label = max(0, max(labels))
        ids = np.arange(0, n_data)
        ids_to_cluster = ids[labels == -2]

        # do clustering
        clustered_representations = StandardScaler().fit_transform(
            representations[ids_to_cluster]
        )
        # prevent OOM error in DBSCAN: exclude outliers that have extreme values
        outliers = find_ids_with_large_absolute_values(
            clustered_representations, threshold=100
        )
        if len(outliers) > 0:
            labels[outliers] = -1
            continue

        new_labels = DBSCAN(eps=eps, min_samples=min_sampled).fit_predict(
            clustered_representations
        )

        for idx, label in zip(ids_to_cluster, new_labels):
            if label != -1:
                label += max_cluster_label
            labels[idx] = label

        # split up big clusters
        for label in np.unique(labels):
            if label < 0:
                continue
            if sum(labels == label) >= max_cluster_size:
                labels[labels == label] = -2

        # check min number of structures, that are unlabeled or in different clusters
        all_labels = [label for label in np.unique(labels) if label >= 0]
        if len(all_labels) + sum(labels == -1) < min_different:
            labels[labels >= 0] = -2

        # return if all structures are labeled
        if sum(labels == -2) == 0:
            return labels

    # set unclustered structures
    labels[labels == -2] = -1

    return labels


def find_ids_with_large_absolute_values(data, threshold):
    # Find indices where absolute values of features are greater than threshold
    large_abs_indices = np.any(np.abs(data) > threshold, axis=1)

    # Extract the indices of data points where at least one feature meets the condition
    ids_with_large_abs_values = np.where(large_abs_indices)[0]

    return ids_with_large_abs_values


def labels_to_clusters(cluster_labels, ids=None):
    ids = ids or list(range(len(cluster_labels)))
    clusters = defaultdict(list)
    for idx, label in zip(ids, cluster_labels):
        clusters[label].append(idx)

    return clusters
