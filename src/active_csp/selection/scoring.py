import numpy as np
from ase.db import connect
import random
from activecsp.utils import standardize

__all__ = [
    "get_random_scores",
    "get_uncertainty_scores",
    "get_laqa_uncertainty_scores",
    "get_laqa_scores",
    "build_scoring_function",
]


def get_random_scores(db_path: str) -> np.ndarray:
    scores = []
    with connect(db_path) as db:
        for i in range(len(db)):
            scores.append(random.random())
    return np.array(scores)


def get_uncertainty_scores(db_path: str) -> np.ndarray:
    force_uncertainties, stress_uncertainties = [], []
    with connect(db_path) as db:
        for i in range(len(db)):
            data = db.get(i + 1).data
            if "forces" in data.keys():
                force_uncertainties.append(np.abs(data["force_uncertainty"]).max())
                stress_uncertainties.append(np.abs(data["stress_uncertainty"]).max())
            else:
                force_uncertainties.append(random.random())
                stress_uncertainties.append(random.random())
    # todo: check standardize
    scores = (
        np.array(
            [
                max(force_uncertainty, stress_uncertainty)
                for force_uncertainty, stress_uncertainty in zip(
                    standardize(np.array(force_uncertainties)),
                    standardize(np.array(stress_uncertainties)),
                )
            ]
        )
        * -1
    )
    return scores


def get_laqa_uncertainty_scores(db_path):
    energies, max_forces, force_uncertainties, stress_uncertainties = [], [], [], []
    with connect(db_path) as db:
        for i in range(len(db)):
            atmsrw = db.get(i + 1)
            data = atmsrw.data
            n_atoms = len(atmsrw.toatoms().numbers)
            if "forces" in data.keys():
                energies.append(data["energy"].item() / n_atoms)
                max_forces.append(np.max(np.linalg.norm(data["forces"], axis=1)))
                force_uncertainties.append(np.abs(data["force_uncertainty"]).max())
                stress_uncertainties.append(np.abs(data["stress_uncertainty"]).max())
            else:
                energies.append(random.random())
                max_forces.append(random.random())
                force_uncertainties.append(random.random())
                stress_uncertainties.append(random.random())

    energies = standardize(np.array(energies))
    max_forces = standardize(np.array(max_forces))
    force_uncertainties = standardize(np.array(force_uncertainties))
    stress_uncertainties = standardize(np.array(stress_uncertainties))

    scores = energies - max_forces - force_uncertainties - stress_uncertainties

    return scores


def get_laqa_scores(db_path, force_weight=0.5):
    energies, max_forces = [], []
    with connect(db_path) as db:
        for i in range(len(db)):
            atmsrw = db.get(i + 1)
            data = atmsrw.data
            n_atoms = len(atmsrw.toatoms().numbers)
            if "forces" in data.keys():
                energies.append(data["energy"].item() / n_atoms)
                max_forces.append(np.max(np.linalg.norm(data["forces"], axis=1)))
            else:
                energies.append(random.random())
                max_forces.append(random.random())

    laqa_score = np.array(energies) - force_weight * np.array(max_forces)
    return laqa_score


def build_scoring_function(function_name):
    if function_name == "laqa":
        return get_laqa_scores
    elif function_name == "random":
        return get_random_scores
    elif function_name == "uncertainty":
        return get_uncertainty_scores
    elif function_name == "laqa_uncertainty":
        return get_laqa_uncertainty_scores
    else:
        raise NotImplementedError(f"Invalid scoring function: {function_name}")
