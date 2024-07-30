#!/usr/bin/env python3


from activecsp.utils.state_tracker import StateTracker
import argparse
import os

from scripts.utils.script_functions import optimize_pool


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process command-line arguments")
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("input_db", type=str, help="Path to the unoptimized database")
    parser.add_argument("output_db", type=str, help="Path to the optimized database")
    parser.add_argument(
        "--cutoff", type=float, default=7.0, help="Cutoff value (default: 7)"
    )
    parser.add_argument("--max_steps", type=int, default=300, help="")
    parser.add_argument("--force_th", type=float, default=0.05, help="")
    parser.add_argument("--energy_patience", type=float, default=20, help="")
    parser.add_argument("--uncertainty_th", type=float, default=0.5, help="")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (default: cuda)"
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
        job_id=int(os.getenv("SLURM_JOB_ID") or -1),
    )

    # run structure optimization for input db
    optimize_pool(
        model_path=args.model_path,
        input_db=args.input_db,
        output_db=args.output_db,
        cutoff=args.cutoff,
        max_steps=args.max_steps,
        force_th=args.force_th,
        energy_patience=args.energy_patience,
        uncertainty_th=args.uncertainty_th,
        device=args.device,
    )
    # update state
    state_tracker.update_state(job_state="done")
