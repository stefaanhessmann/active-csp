#!/usr/bin/env python3


from ase.io import read
import argparse

from deepcryspy.utils.state_tracker import StateTracker
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process command-line arguments")
    parser.add_argument("input_path", type=str, help="Path to input structure as .cif")
    parser.add_argument("calculator_inputs", type=str, help="")
    parser.add_argument("kspacing", type=float, help="")
    parser.add_argument("pseudopotentials", type=str, help="")
    parser.add_argument("pwx_path", type=str, help="")
    parser.add_argument("--n_threads", type=int, default=1, help="")
    parser.add_argument("--n_processes", type=int, default=16, help="")
    parser.add_argument(
        "--state_path",
        type=str,
        default="state.yaml",
        help="path to communicate job state",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # load arguments and set state
    args = parse_arguments()
    state_tracker = StateTracker(args.state_path)
    state_tracker.update_state(
        job_state="running",
    )

    # generate random labels
    atoms = read(args.input_path)
    energy = np.random.random((1))
    forces = np.random.random(atoms.positions.shape)
    stress = np.random.random((3, 3))

    # save results
    state_tracker.update_state(
        job_state="done",
        energy=energy.tolist(),
        forces=forces.tolist(),
        stress=stress.tolist(),
    )
