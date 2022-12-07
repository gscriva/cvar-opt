import json
import multiprocessing as mp
from datetime import datetime
from pathlib import Path

import numpy as np

from . import utils, vqe

# Use Aer's simulator
SIMULATOR_METHOD = "automatic"
# define specific optimizer
METHOD = "COBYLA"
# limit cpu usage
MAX_CPUS = min(int(mp.cpu_count() / 2), 12)
# dimension of the ising model
# and external field (present only in ferro model)
DIM = 1
H_FIELD = -0.05
# seed random point
SEED = 42


def cvar_opt(
    qubits: int,
    circ_depth: int,
    shots: list[int],
    maxiters: list[int],
    initial_points: list[int],
    type_ansatz: str = "vqe",
    noise_model: bool = False,
    type_ising: str = "ferro",
    seed: int = 42,
    alpha: int = 25,
    save_dir: str = None,
    verbose: int = 0,
) -> None:
    start = datetime.now()

    # create the circuit
    ansatz = utils.create_ansatz(
        qubits,
        circ_depth,
        type_ansatz,
    )
    # define generator for initial point
    rng = np.random.default_rng(seed=SEED)
    # define initial_points different starting points
    if len(initial_points) == 1:
        initial_points.insert(0, 0)
    # generate initial points
    thetas0 = utils.get_init_points(initial_points, ansatz.num_parameters, rng)
    # hamiltonian is defined with +
    # following http://spinglass.uni-bonn.de/ notation
    ising, global_min = utils.create_ising1d(qubits, DIM, type_ising, H_FIELD, seed)
    print(ising)
    print(f"J: {ising.adj_dict}\nh: {ising.h_field}\n")

    # check if directory exists
    if save_dir is not None:
        if not Path(save_dir).is_dir():
            print(f"'{save_dir}' not found")
            raise FileNotFoundError
    else:
        save_dir = Path().absolute()

    jobs = [(shot, maxiter) for shot in shots for maxiter in maxiters]
    for shot, maxiter in jobs:
        start_it = datetime.now()
        print(f"\nShots: {shot}\tMaxiter: {maxiter}\tInitial Points: {initial_points}")
        # define optimization class
        var_problem = vqe.VQE(
            ansatz,
            ising,
            optimizer=METHOD,
            backend=SIMULATOR_METHOD,
            noise_model=noise_model if noise_model else None,
            shots=shot,
            maxiter=maxiter,
            alpha=alpha,
            global_min=global_min,
            verbose=verbose,
        )
        print(var_problem)

        # optimize initial_points instances in parallel
        # up to MAX_CPUS
        processes = min(MAX_CPUS, len(thetas0))
        with mp.Pool(processes) as pool:
            results = pool.map(var_problem.minimize, thetas0)

        # write on json to save results
        filename = (
            f"{save_dir}/shots{str(shot).zfill(4)}_maxiter{str(maxiter).zfill(3)}.json"
        )
        with open(filename, "w") as file:
            json.dump(results, file, cls=utils.NumpyArrayEncoder, indent=4)

        # report iteration execution time
        stop_it = datetime.now()
        # normalize per CPUs and iterations
        per_cpu_runs = (initial_points[1] - initial_points[0]) / processes
        delta_it = (stop_it - start_it) / per_cpu_runs
        print(f"\nSave results in {filename}")
        print(f"CPU-per-job time: {delta_it.total_seconds():.2f}s\n")

    # report total execution time
    stop = datetime.now()
    print(f"Wall time: {stop - start}s\n")
    return
