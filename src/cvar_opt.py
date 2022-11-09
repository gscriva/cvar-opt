import json
from datetime import datetime
from math import pi
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List

import numpy as np
from qiskit.providers.aer import AerSimulator

from src.utils import NumpyArrayEncoder, create_ising1d, param_circ
from src.vqe import VQE

# Use Aer's qasm_simulator
SIMULATOR_METHOD = "automatic"
# define specific optimizer
METHOD = "COBYLA"
# initial points number
NUM_INIT = 1000
# limit cpu usage
MAX_CPUS = min(int(cpu_count() / 2), 16)
# define a ferro ising model
# with uniform external field
# TODO J and h could also be np.ndarray
J = 1.0
h = 0.05 * J
DIM = 1


def cvar_opt(
    qubits: int,
    circ_depth: int,
    shots: List[int],
    maxiter: List[int],
    seed: int = 42,
    alpha: int = 25,
    save_dir: str = None,
    verbose: int = 0,
) -> None:

    # define generator for initial point
    rng = np.random.default_rng(seed=seed)

    # TODO make a function with params (spins, J, h)
    # hamiltonian is defined with +
    # following http://spinglass.uni-bonn.de/ notation
    ising, global_min = create_ising1d(qubits, DIM, J, h)
    if verbose > 0:
        print(ising)
        print(f"J:{ising.adja_dict}, h:{ising.ext_field}\n")

    # check if directory exists
    if save_dir is not None:
        if not Path(save_dir).is_dir():
            print(f"'{save_dir}' not found")
            raise FileNotFoundError
    else:
        save_dir = Path().absolute()

    for shot in shots:
        for steps in maxiter:
            start = datetime.now()
            if verbose > 0:
                print(f"\nShots {shot} Maxiter {steps}")

            thetas0: List[np.ndarray] = []
            num_param = qubits * (circ_depth + 1)
            # define NUM_INIT different starting points
            for _ in range(NUM_INIT):
                # generate initial point
                # uniform in [-2pi,2pi]
                thetas0.append(rng.uniform(-2 * pi, 2 * pi, num_param))
            # create and eval the circuit
            qc = param_circ(qubits, circ_depth)
            # define optimization class
            vqe = VQE(
                qc,
                ising,
                optimizer=METHOD,
                backend=SIMULATOR_METHOD,
                shots=shot,
                maxiter=steps,
                alpha=alpha,
                global_min=global_min,
                verbose=verbose,
            )
            if verbose:
                print(vqe)
            # optimize NUM_INIT instances in parallel
            # up to MAX_CPUS available on the system
            with Pool(processes=MAX_CPUS) as pool:
                results = pool.map(vqe.minimize, thetas0)
            # write on json to save results
            filename = f"{save_dir}/shots{str(shot).zfill(4)}_maxiter{str(steps).zfill(3)}.json"
            with open(filename, "w") as file:
                json.dump(results, file, cls=NumpyArrayEncoder, indent=4)
            # report total execution time
            stop = datetime.now()
            if verbose > 0:
                print(f"\nSave results in {filename}")
                print(f"Total runtime: {(stop - start).seconds}s\n")
    return
