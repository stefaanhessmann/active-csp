#!/usr/bin/env python3


from ase.io import read, write
import argparse
from activecsp.utils.state_tracker import StateTracker

from scripts.utils.script_functions import espresso_optimization
import json


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process command-line arguments")
    parser.add_argument("input_path", type=str, help="Path to input structure as .cif")
    parser.add_argument(
        "output_path", type=str, help="Path to output structure as .cif"
    )
    parser.add_argument("calculator_inputs", type=str, help="")
    parser.add_argument("kspacing", type=float, help="")
    parser.add_argument("pseudopotentials", type=str, help="")
    parser.add_argument("pwx_path", type=str, help="")
    parser.add_argument("--n_threads", type=int, default=1, help="")
    parser.add_argument("--n_processes", type=int, default=16, help="")
    parser.add_argument("--max_steps", type=int, default=300, help="")
    parser.add_argument("--damping", type=float, default=1.0, help="")
    parser.add_argument("--force_th", type=float, default=0.05, help="")
    parser.add_argument(
        "--trajectory_file", type=str, default="optimization.traj", help=""
    )
    parser.add_argument(
        "--state_path",
        type=str,
        default="job_state.yaml",
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

    # run structure optimization
    atoms = read(args.input_path)
    atoms = espresso_optimization(
        atoms=atoms,
        calculator_inputs=json.loads(args.calculator_inputs),
        kspacing=args.kspacing,
        pseudopotentials=json.loads(args.pseudopotentials),
        pwx_path=args.pwx_path,
        n_threads=args.n_threads,
        n_processes=args.n_processes,
        max_steps=args.max_steps,
        damping=args.damping,
        force_th=args.force_th,
        trajectory_file=args.trajectory_file,
    )

    # save results
    write(args.output_path, atoms)
    state_tracker.update_state(job_state="done")
