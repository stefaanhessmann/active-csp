import numpy as np
from ase.db import connect
from random import shuffle
from deepcryspy.utils.db import (
    get_metadata,
    get_atoms,
    get_energies,
    get_property,
    get_masked_ids,
)
from deepcryspy.selection.clustering import labels_to_clusters
from deepcryspy.selection import build_scoring_function


__all__ = [
    "CandidateSelection",
    "ScoringBasedSelection",
    "MostStableSelection",
    "random_selection",
]


class CandidateSelection:

    def select(self, db_path, cluster_labels=None):
        raise NotImplementedError


class ScoringBasedSelection(CandidateSelection):

    def __init__(
        self,
        n_candidates,
        scoring_function,
        scoring_function_args,
        use_train_ids=True,
        use_select_ids=False,
    ):
        self.n_candidates = n_candidates
        # todo: scoring function also as a class maybe?
        self.scoring_function = build_scoring_function(scoring_function)
        self.scoring_function_args = scoring_function_args
        self.use_train_ids = use_train_ids
        self.use_select_ids = use_select_ids

    def select(self, db_path, cluster_labels=None):
        # compute scores
        scoring_values = self.scoring_function(
            db_path, **(self.scoring_function_args or dict())
        )

        if cluster_labels is None:
            cluster_labels = np.ones_like(scoring_values) * -1

        # get metadata
        ids_metadata = get_metadata(
            db_path=db_path,
            masked_ids=True,
            train_ids=True,
            select_ids=True,
        )
        masked_ids = ids_metadata["masked_ids"]
        train_ids = ids_metadata["train_ids"]
        select_ids = ids_metadata["select_ids"]

        # select candidates
        next_candidates = []
        used_clusters = [
            cluster_labels[idx] for idx in masked_ids if cluster_labels[idx] != -1.0
        ]
        for idx in np.argsort(scoring_values):
            idx = idx.item()

            # ignore masked ids and clusters
            cluster_label = cluster_labels[idx]
            if cluster_label in used_clusters:
                continue

            # filter out unwanted subsets
            if idx in train_ids and not self.use_train_ids:
                continue
            if idx in select_ids and not self.use_select_ids:
                continue

            # add new idx
            next_candidates.append(idx)

            # mark selected cluster
            if cluster_label != -1:
                used_clusters.append(cluster_label)

            if len(next_candidates) == self.n_candidates:
                return [int(idx) for idx in next_candidates]

        # raise exception if not enough datapoints present
        raise IndexError(
            f"Only {len(next_candidates)} of {self.n_candidates} candidates found for selection."
        )


class MostStableSelection(CandidateSelection):
    def __init__(
        self,
        n_best,
    ):
        self.n_best = n_best

    def select(self, db_path, cluster_labels=None):
        # return when no clusters are present
        if cluster_labels is None:
            return [], []

        # find low energy clusters
        clusters = labels_to_clusters(cluster_labels)
        energy_mapping = {
            cluster: np.min(get_energies(db_path, ids=ids))
            for cluster, ids in clusters.items()
            if cluster != -1
        }
        cluster_ids = list(energy_mapping.keys())
        cluster_energies = list(energy_mapping.values())
        sorted_clusters = np.array(cluster_ids)[np.argsort(cluster_energies)]

        # load masked ids
        masked_ids = get_masked_ids(db_path)

        # collect most promising candidates
        # todo: remove loading of masked ids: they are not currently used
        selected_candidates, candidate_mask = [], []
        for cluster_idx in sorted_clusters:

            # stopping criterion
            if len(selected_candidates) >= self.n_best:
                break

            # get cluster ids
            ids = clusters[cluster_idx]

            # check if cluster is masked
            cluster_masked = False
            for idx in ids:
                if idx in masked_ids:
                    cluster_masked = True
            candidate_mask.append(cluster_masked)

            # add next candidate
            energies = get_energies(db_path, ids)
            best_idx = ids[np.argmin(energies)]
            selected_candidates.append(best_idx)

        return selected_candidates, candidate_mask


def random_selection(
    db_path,
    n_candidates,
    use_train_ids=True,
    use_select_ids=False,
):
    ids_metadata = get_metadata(
        db_path=db_path,
        train_ids=True,
        select_ids=False,
    )
    train_ids = ids_metadata["train_ids"]
    select_ids = ids_metadata["select_ids"]
    ids = []
    if use_train_ids:
        ids.extend(train_ids)
    if use_select_ids:
        ids.extend(select_ids)
    shuffle(ids)
    return ids[:n_candidates]
