#!/usr/bin/env python3


import torch
import numpy as np
import argparse

from scripts.utils.script_functions import compute_representations


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process command-line arguments")
    parser.add_argument("model_path", type=str, help="Path to the model")
    parser.add_argument("db_path", type=str, help="Path to the database")
    parser.add_argument("results_path", type=str, help="Path to the processed results")
    parser.add_argument(
        "--cutoff", type=float, default=7.0, help="Cutoff value (default: 7)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (default: cuda)"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    representations = compute_representations(
        model_path=args.model_path,
        db_path=args.db_path,
        device=torch.device(args.device),
        cutoff=args.cutoff,
        batch_size=args.batch_size,
    )
    np.save(args.results_path, representations)
