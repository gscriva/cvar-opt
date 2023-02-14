import argparse
from pathlib import Path

from src import cvar_opt

parser = argparse.ArgumentParser()

parser.add_argument("--qubits", type=int, help="Number of qubits/spins")
parser.add_argument(
    "--circ-depth", type=int, default=1, help="Depth of the circuit (default: 1)"
)
parser.add_argument(
    "--shots",
    nargs="+",
    type=int,
    default=[None],
    help="Number circuit evaluations, may be a list. If None the exact quantum hamiltonian is used. (default: None)",
)
parser.add_argument(
    "--maxiters",
    nargs="+",
    type=int,
    default=[None],
    help="Maximum optimizer steps, may be a list. If None the classical optimizer runs until convergence (default: None)",
)
parser.add_argument(
    "--initial-points",
    nargs="+",
    type=int,
    default=[1000],
    help="Number of initial points, i.e., different runs to produce. Use two values input to resume old run. (default: 1000)",
)
parser.add_argument(
    "--type-ansatz",
    type=str,
    default="vqe",
    choices={"vqe", "qaoa", "qaoa+"},
    help="Choose the quantum ansatz to use (default: 'vqe')",
)
parser.add_argument(
    "--noise-model",
    action="store_true",
    help="Flag to run the circuit with a noisy simulator (default: False)",
)
parser.add_argument(
    "--type-ising",
    type=str,
    default="ferro",
    choices={"ferro", "binary", "random"},
    help="Change between ferromagnetic, random binary model and random gaussian model with external field (default: 'ferro')",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed to generate different ising random models [only apply with random gaussian] (default: 42)",
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
    action="count",
    default=0,
    help="Verbosity level for prints (default=0)",
)


def main(args: argparse.ArgumentParser):
    cvar_opt.cvar_opt(
        args.qubits,
        args.circ_depth,
        args.shots,
        args.maxiters,
        args.initial_points,
        args.type_ansatz,
        args.noise_model,
        args.type_ising,
        args.seed,
        args.alpha,
        args.save_dir,
        args.verbose,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
