#!/usr/bin/env python3


import torch
from ase.db import connect
from tqdm import tqdm
import numpy as np
from schnetpack.data import ASEAtomsData, AtomsLoader
from schnetpack.transform import CastTo32, MatScipyNeighborList
from activecsp.selection import DBSCANClustering, labels_to_clusters
from activecsp.utils.db import get_select_ids
import argparse
import shutil
from yaml import dump
from typing import List, Dict


def my_bool(s):
    return s != "False"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process command-line arguments")
    parser.add_argument("model_path", type=str)
    parser.add_argument("current_db_path", type=str)
    parser.add_argument("previous_db_path", type=str)
    parser.add_argument("--n_models", type=int, default=5)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--n_clusters", type=int, default=1)
    parser.add_argument("--eps", type=float, default=0.5)
    parser.add_argument("--min_sampled", type=int, default=3)
    parser.add_argument("--min_different", type=int, default=200)
    parser.add_argument("--max_tries", type=int, default=10)
    parser.add_argument("--max_cluster_size", type=int, default=500)
    parser.add_argument("--recluster_stable", type=bool, default=False)
    parser.add_argument("--cutoff", type=float, default=7.0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--include_train_structures", type=my_bool, default=True)

    return parser.parse_args()


def get_labels(
    model_path,
    db_path,
    device,
    cutoff,
    batch_size,
    model_idx=0,
):
    # load model
    models = torch.load(model_path)
    model = models[model_idx]
    model.eval()
    model.do_postprocessing = True
    model = model.to(device)

    # add metadata
    with connect(db_path) as db:
        metadata = db.metadata
        metadata.update(
            {
                "_property_unit_dict": dict(energy="eV"),
                "_distance_unit": "Ang",
                "atomrefs": dict(),
            }
        )
        db.metadata = metadata

    # load data
    dataset = ASEAtomsData(
        datapath=db_path,
        transforms=[
            CastTo32(),
            MatScipyNeighborList(cutoff=cutoff),
        ],
        load_properties=[],
    )
    data_loader = AtomsLoader(dataset, batch_size=batch_size, num_workers=6)
    # Iterate over the dataset batchwise
    all_representations = []
    all_energies = []
    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass and get the model output
        results = model(batch)

        # get energies per atom
        all_energies.append(
            (results["energy"] / batch["_idx_m"].unique(return_counts=True)[1])
            .detach()
            .cpu()
            .numpy()
        )

        # get global representations
        ids = batch["_idx_m"]
        data = batch["scalar_representation"]
        unique_ids = torch.unique(ids)
        representations = []
        # todo: no pooling of representations before saving to metadata
        for uid in unique_ids:
            representations.append(torch.mean(data[ids == uid], axis=0))
        representations = torch.stack(representations)

        # Move representation back to CPU and convert to numpy if it's a tensor
        if isinstance(representations, torch.Tensor):
            representations = representations.cpu().detach().numpy()

        # Collect the representation
        all_representations.append(representations)

    return np.vstack(all_representations), np.hstack(all_energies)


def get_cluster_energies(clusters: Dict, energies: List, n_clusters: int = 1):
    energy_mapping = {
        cluster: np.min(energies[ids]).item()
        for cluster, ids in clusters.items()
        if cluster != -1
    }
    le_structure_mapping = {
        cluster: ids[np.argmin(energies[ids])]
        for cluster, ids in clusters.items()
        if cluster != -1
    }
    cluster_ids = list(energy_mapping.keys())
    cluster_energies = list(energy_mapping.values())
    sorted_clusters = np.array(cluster_ids)[np.argsort(cluster_energies)]

    low_energies = [
        energy_mapping[cluster_idx] for cluster_idx in sorted_clusters[:n_clusters]
    ]
    le_ids = [
        le_structure_mapping[cluster_idx]
        for cluster_idx in sorted_clusters[:n_clusters]
    ]

    return le_ids, low_energies


if __name__ == "__main__":
    print("starting...")
    args = parse_arguments()
    print("Args:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    # empty lists for results
    results = dict()

    # define clustering method
    clustering = DBSCANClustering(
        eps=args.eps,
        min_sampled=args.min_sampled,
        min_different=args.min_different,
        max_tries=args.max_tries,
        max_cluster_size=args.max_cluster_size,
        recluster_stable=args.recluster_stable,
    )

    # copy data
    tmp_current_db_path = "/tmp/current.db"
    tmp_previous_db_path = "/tmp/previous.db"
    shutil.copyfile(args.current_db_path, tmp_current_db_path)
    shutil.copyfile(args.previous_db_path, tmp_previous_db_path)

    # get energies and ids for low energy clusters
    for tag, db_path in zip(
        ["current", "previous"], [tmp_current_db_path, tmp_previous_db_path]
    ):
        print(f"starting evaluation {tag} at path {db_path}")
        # compute representations and energies
        all_representations = []
        all_energies = []
        for idx in range(args.n_models):
            representations, energies = get_labels(
                args.model_path,
                db_path,
                args.device,
                args.cutoff,
                args.batch_size,
                model_idx=idx,
            )
            all_representations.append(representations)
            all_energies.append(energies)
        all_representations = np.stack(all_representations)
        # todo: check axis
        all_energies = np.mean(all_energies, axis=0)
        np.savez(
            f"results_idx_{tag}.npz",
            representations=all_representations,
            energies=all_energies,
        )

        # filter out train structures if needed
        if args.include_train_structures:
            ids = list(range(len(all_energies)))
        else:
            ids = get_select_ids(db_path)

        # compute energies of most stable clusters
        cluster_labels = clustering.get_clusters(all_representations[0][ids])
        clusters = labels_to_clusters(cluster_labels)
        most_stable_ids, cluster_energies = get_cluster_energies(
            clusters,
            all_energies[ids],
            n_clusters=args.n_clusters,
        )
        # get db ids from filtered selection
        most_stable_ids = [ids[idx] for idx in most_stable_ids]

        # collect results
        results[tag] = dict(ids=most_stable_ids, energies=cluster_energies)

        # write results
        with open("results_log.txt", "a") as file:
            file.write(f"db: {db_path}\n")
            file.write(f"energies: {cluster_energies} --- ids: {most_stable_ids}\n")
            file.write("---\n")
        print(f"energies: {cluster_energies} --- ids: {most_stable_ids}")

    # evaluate stopping criterion
    if len(results["previous"]["energies"]) != len(results["current"]["energies"]):
        diff = np.array([np.inf])
    else:
        diff = np.array(results["previous"]["energies"]) - np.array(
            results["current"]["energies"]
        )
    if diff.max() <= args.epsilon:
        stop = True
    else:
        stop = False
    print(f"stop: {stop}")
    # write results to file
    with open("stop", "w") as file:
        file.write(str(stop) + "\n")
    with open("results.yaml", "w") as file:
        dump(results["current"], file)

    print("done.")
