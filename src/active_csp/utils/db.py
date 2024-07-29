from ase.db import connect
from collections import defaultdict
from tqdm import tqdm
from ase.visualize import view


__all__ = [
    "get_energies",
    "get_atoms",
    "get_property",
    "get_metadata",
    "get_masked_ids",
    "get_train_ids",
    "get_select_ids",
    "get_blocked_ids",
    "mask_relaxation_ids",
    "write_pool",
    "view_atoms",
]


def get_energies(db_path, ids=None):
    energies = [e.item() for e in get_property(db_path, p_name="energy", ids=ids)]
    return energies


def get_atoms(db_path, ids=None):
    if ids == []:
        return []
    with connect(db_path) as db:
        ids = ids or list(range(len(db)))
        atoms_list = [db.get(idx + 1).toatoms() for idx in ids]
    return atoms_list


def get_property(db_path, p_name, ids=None):
    with connect(db_path) as db:
        ids = ids or list(range(len(db)))
        properties = [db.get(idx + 1).data[p_name] for idx in ids]
    return properties


def get_metadata(db_path, masked_ids=True, train_ids=True, select_ids=True):
    metadata = defaultdict(list)
    with connect(db_path) as db:
        for i in tqdm(range(len(db)), "collect idx metadata"):
            atmsrw = db.get(i + 1)
            if masked_ids and atmsrw.is_masked:
                metadata["masked_ids"].append(i)
            if train_ids and atmsrw.is_train:
                metadata["train_ids"].append(i)
            else:
                metadata["select_ids"].append(i)

    return metadata


def get_masked_ids(db_path):
    masked_ids = []
    with connect(db_path) as db:
        for i in range(len(db)):
            atmsrw = db.get(i + 1)
            if atmsrw.is_masked == True:
                masked_ids.append(i)

    return masked_ids


def get_train_ids(db_path):
    train_ids = []
    with connect(db_path) as db:
        for i in range(len(db)):
            atmsrw = db.get(i + 1)
            if atmsrw.is_train:
                train_ids.append(i)

    return train_ids


def get_select_ids(db_path):
    train_ids = []
    with connect(db_path) as db:
        for i in range(len(db)):
            atmsrw = db.get(i + 1)
            if not atmsrw.is_train:
                train_ids.append(i)

    return train_ids


def get_blocked_ids(db_path):
    ids_blocked = []
    with connect(db_path) as db:
        for i in range(len(db)):
            atmsrw = db.get(i + 1)
            if hasattr(atmsrw, "ids_blocked") and atmsrw.ids_blocked == True:
                ids_blocked.append(i)

    return ids_blocked


def mask_relaxation_ids(db_path, relaxation_ids):
    # todo: maybe make use of cluster labels
    with connect(db_path) as db:
        for idx in relaxation_ids:
            db.update(idx + 1, is_masked=True)


def write_pool(db_path, train_pool, selection_pool):
    idx = 0
    with connect(db_path) as db:
        db.metadata = dict(
            _property_unit_dict=dict(),
            _distance_unit="Ang",
        )
        for atms in train_pool:
            db.write(atms, is_masked=False, is_train=True, initial_idx=idx)
            idx += 1
        for i, atms in enumerate(selection_pool):
            db.write(atms, is_masked=False, is_train=False, initial_idx=idx)
            idx += 1


def view_atoms(db_path, ids=None):
    atoms_list = get_atoms(db_path, ids)
    view(atoms_list)
