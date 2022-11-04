import argparse
from pathlib import Path

from src.cvar_opt import cvar_opt

parser = argparse.ArgumentParser()

parser.add_argument("--qubits", type=int, help="Number of qubits/spins")
parser.add_argument("--circ-depth", type=int, help="Depth of the circuit")
parser.add_argument(
    "--shots", nargs="+", type=int, help="Number circuit evaluations, may be a list"
)
parser.add_argument(
    "--maxiter",
    nargs="+",
    type=int,
    default=[None],
    help="Maximum optimizer steps",
)
parser.add_argument(
    "--seed", type=int, default=42, help="Seed to generate initial points (default: 42)"
)
parser.add_argument(
    "--alpha",
    type=int,
    default=25,
    choices={1, 5, 10, 25, 50, 75, 100},
    help="Alpha-th quantile to evaluate the loss (0,100] (default: 25)",
)
parser.add_argument(
    "--save-dir",
    type=Path,
    default=None,
    help="Path to the save directory, None means local dir (default: None)",
)
parser.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    action="store_true",
    help="Flag for verbose prints",
)


def main(args: argparse.ArgumentParser):
    cvar_opt(
        args.qubits,
        args.circ_depth,
        args.shots,
        args.maxiter,
        args.seed,
        args.alpha,
        args.save_dir,
        args.verbose,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
