import os.path
import traceback
from typing import Optional, Dict

import ase
import numpy as np
import torch
from ase import Atoms
from ase.calculators.espresso import Espresso
from ase.constraints import ExpCellFilter
from ase.db import connect
from ase.optimize import LBFGS
from schnetpack.data import ASEAtomsData, AtomsLoader
from schnetpack.transform import CastTo32, MatScipyNeighborList
from tqdm import tqdm

from deepcryspy import Properties
from deepcryspy.force_field import NNEnsemble, EnsembleCalculator
from ase.io.trajectory import Trajectory


def optimize_pool(
    model_path: str,
    input_db: str,
    output_db: str,
    cutoff: float,
    max_steps: int,
    force_th: float,
    energy_patience: int,
    uncertainty_th: float,
    device: str,
    trajectory_file: Optional[str] = None,
):
    # load model and build calculator + optimizer
    ensemble = NNEnsemble(
        models=torch.load(model_path, map_location=device),
        properties=[Properties.e, Properties.f, Properties.s],
    )

    with connect(input_db) as db:
        db_size = len(db)
        print(db_size)

    for structure_idx in range(db_size):

        calculator = EnsembleCalculator(
            ensemble=ensemble,
            cutoff=cutoff,
            device=device,
            callbacks=[],
        )

        # load structure
        with connect(input_db) as db:
            atmsrw = db.get(structure_idx + 1)
            atoms = atmsrw.toatoms()
            key_value_pairs = atmsrw.key_value_pairs
            atoms.wrap()

        # run optimization and save results
        e_min = np.inf
        e_counter = 0
        best_atoms = None
        best_data = dict()
        try:
            atoms.calc = calculator
            optimizer = LBFGS(ExpCellFilter(atoms), force_consistent=False, damping=0.7)
            # attach trajectory file
            if trajectory_file is not None:
                trajectory = Trajectory(
                    trajectory_file,
                    "a",
                    atoms,
                    properties=[Properties.e, Properties.f, Properties.s],
                )
                optimizer.attach(trajectory.write)
            for converged in optimizer.irun(steps=max_steps, fmax=force_th):
                # energy patience criterion
                e = atoms.calc.results[Properties.e]
                e_u = atoms.calc.results[Properties.e_u]
                f = atoms.calc.results[Properties.f]
                f_u = atoms.calc.results[Properties.f_u]
                s = atoms.calc.results[Properties.s]
                s_u = atoms.calc.results[Properties.s_u]
                f_max = np.max(
                    np.linalg.norm(ExpCellFilter(atoms).get_forces(), axis=1)
                )
                if e <= e_min:
                    e_min = e
                    e_counter = 0
                    best_atoms = atoms.copy()
                    best_data = {
                        Properties.e: e,
                        Properties.e_u: e_u,
                        Properties.f: f,
                        Properties.f_u: f_u,
                        Properties.s: s,
                        Properties.s_u: s_u,
                        "f_max": np.array(f_max),
                    }
                else:
                    e_counter += 1
                if e_counter >= energy_patience:
                    break

                # force uncertainty criterion
                if (
                    np.max(np.abs(f_u)) / np.max(np.abs(f)) >= uncertainty_th
                ):  # and f_max >= 0.5:
                    break

            print("#######")
            with connect(output_db) as db:
                best_atoms.calc = None
                db.write(best_atoms, data=best_data, key_value_pairs=key_value_pairs)
                print("written date to db")
        except Exception as e:
            with open("error.txt", "a") as file:
                file.write(traceback.format_exc())


def espresso_optimization(
    atoms: ase.Atoms,
    calculator_inputs: Dict,
    kspacing: float,
    pseudopotentials: Dict,
    pwx_path: str,
    n_threads: int,
    n_processes: int,
    max_steps: int,
    damping: float,
    force_th: float,
    trajectory_file: Optional[str] = None,
):
    # get calculator
    calculator = Espresso(
        kspacing=kspacing / (2 * np.pi),
        input_data=calculator_inputs,
        pseudopotentials=pseudopotentials,
    )
    calculator.command = f"export OMP_NUM_THREADS={n_threads} ; mpirun -np {n_processes} {pwx_path} -in PREFIX.pwi > PREFIX.pwo"

    # set up optimizer
    atoms.calc = calculator
    optimizer = LBFGS(ExpCellFilter(atoms), force_consistent=False, damping=damping)
    # attach trajectory file
    if trajectory_file is not None:
        trajectory = Trajectory(
            trajectory_file,
            "a",
            atoms,
            properties=[Properties.e, Properties.f, Properties.s],
        )
        optimizer.attach(trajectory.write)

    # run optimization
    for converged in optimizer.irun(steps=max_steps, fmax=force_th):
        pass

    return atoms


def compute_representations(
    model_path,
    db_path,
    device,
    cutoff,
    batch_size,
):
    # load model
    models = torch.load(model_path)
    model = models[0]
    model.eval()
    model.do_postprocessing = False
    model.model_outputs = []
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
    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass and get the model output
        _ = model(batch)

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

    return np.vstack(all_representations)


def espresso_evaluation(
    atoms: Atoms,
    calculator_inputs: Dict,
    kspacing: float,
    pseudopotentials: Dict,
    pwx_path: str,
    n_threads: int,
    n_processes: int,
):
    reference_calculator = Espresso(
        kspacing=kspacing / (2 * np.pi),
        input_data=calculator_inputs,
        pseudopotentials=pseudopotentials,
    )
    reference_calculator.command = f"export OMP_NUM_THREADS={n_threads} ; mpirun -np {n_processes} {pwx_path} -in PREFIX.pwi > PREFIX.pwo"

    # compute reference labels
    atoms.calc = reference_calculator
    e = np.array([atoms.get_potential_energy()])
    f = atoms.get_forces()
    s = atoms.get_stress(voigt=False)

    return e, f, s
