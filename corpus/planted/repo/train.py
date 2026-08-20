"""Positive repository fixture.

Two defects are planted here and asserted in tests/test_repro.py:

  * a single fixed seed, while the paper reports a mean over five runs
  * a learning rate that configs/base.yaml overrides to a different value
    than the paper states
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
    parser.add_argument("--dropout", type=float, default=0.1)
    # PLANTED: a computed default cannot be read from source; the tool must
    # report this as unchecked rather than putting a number in the diff table.
    parser.add_argument("--workers", type=int, default=len([1, 2, 3]))
    return parser


def set_seeds():
    # PLANTED: one literal seed, never varied. The paper claims five runs.
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def main():
    args = build_parser().parse_args()
    set_seeds()
    print(args)


if __name__ == "__main__":
    main()
