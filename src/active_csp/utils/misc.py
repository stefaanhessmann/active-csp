import random
from typing import Dict

import numpy as np
import yaml
from ase.db import connect
from filelock import FileLock

from activecsp.utils import Properties


def filter_db(means, stds, db_path):
    clean_ids = []
    with connect(db_path) as db:
        for idx in range(len(db)):
            try:
                atmsrw = db.get(idx + 1)
                data = atmsrw.data
            except KeyError:
                continue

            if (
                np.max(np.abs(data[Properties.f]))
                <= means[Properties.f] + stds[Properties.f] * 3
                and np.max(np.abs(data[Properties.s]))
                <= means[Properties.s] + stds[Properties.s] * 3
            ):
                clean_ids.append(idx)
    return clean_ids


def get_stats(db_path, last_n):
    f_max, s_max = [], []
    with connect(db_path) as db:
        n_data = len(db)
        last_n = min(last_n, n_data)
        for idx in range(min(0, n_data - last_n), len(db)):
            try:
                atmsrw = db.get(idx + 1)
                data = atmsrw.data
            except KeyError:
                continue
            if not Properties.f in data.keys():
                continue

            f_max.append(np.max(np.abs(data[Properties.f])))
            s_max.append(np.max(np.abs(data[Properties.s])))
    return {Properties.f: np.mean(f_max), Properties.s: np.mean(s_max)}, {
        Properties.f: np.std(f_max),
        Properties.s: np.std(s_max),
    }


def read_yaml(
    file_path: str,
) -> Dict:
    if not file_path.endswith(".yaml"):
        raise Exception(f"{file_path} is not a valid yaml file.")
    with FileLock(file_path + ".lock"):
        yaml_args = None
        while yaml_args is None:
            print(f"reading status of file: {file_path}...")
            with open(file_path, "r") as file:
                yaml_args = yaml.full_load(file)

    return yaml_args


def dump_yaml(
    data: Dict,
    file_path: str,
) -> None:
    if not file_path.endswith(".yaml"):
        raise Exception(f"{file_path} is not a valid yaml file.")
    with FileLock(file_path + ".lock"):
        with open(file_path, "w") as file:
            yaml.dump(data, file)


def standardize(data: np.ndarray) -> np.ndarray:
    means = data.mean()
    stds = data.std()
    return (data - means) / stds


def split_ids(ids, n_train=0.8, n_val=0.2, n_test=0.0):
    np.random.shuffle(ids)
    train_ids, val_ids, test_ids, _ = np.split(
        np.array(ids),
        [
            int(len(ids) * n_train),
            int(len(ids) * (n_train + n_val)),
            int(len(ids) * (n_train + n_val + n_test)),
        ],
    )
    train_ids, val_ids, test_ids = (
        train_ids.tolist(),
        val_ids.tolist(),
        test_ids.tolist(),
    )

    return train_ids, val_ids, test_ids
