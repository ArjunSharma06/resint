"""Negative repository fixture. Nothing here may produce a finding.

The seed is read from an argument, so it plainly varies across runs even
though the source shows one call. Every hyperparameter agrees with the paper
once the config layering is resolved.
"""

import argparse
import random

import numpy as np
import torch


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def set_seeds(seed):
    # Read from an argument: this varies across runs, so repro/seed-claim
    # must stay silent even though the paper reports five of them.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = build_parser().parse_args()
    set_seeds(args.seed)
    print(args)


if __name__ == "__main__":
    main()
