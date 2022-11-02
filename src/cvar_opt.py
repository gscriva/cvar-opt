import json
from datetime import datetime
from math import pi
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List

import numpy as np
from qiskit.providers.aer import AerSimulator

from src.ising import IsingModel
from src.utils import NumpyArrayEncoder, param_circ
from src.vqe import VQE

# Use Aer's qasm_simulator
SIMULATOR_METHOD = "automatic"
# define specific optimizer
METHOD = "COBYLA"
# initial points number
NUM_INIT = 2
# define a ferro ising model
# with uniform external field
# TODO J and h could also be np.ndarray
J = 1
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
    verbose: bool = False,
) -> None:

    # define generator for initial point
    rng = np.random.default_rng(seed=seed)
    # define simulator's backend
    simulator = AerSimulator(method=SIMULATOR_METHOD)

    # TODO make a function with params (spins, J, h)
    # hamiltonian is defined with +
    # following http://spinglass.uni-bonn.de/ notation
    adja_dict = {}
    field = np.zeros(qubits)
    ext_field = h
    for i in range(qubits):
        field[i] = ext_field
        if i == qubits - 1:
            continue
        adja_dict[(i, i + 1)] = -J
    # class devoted to set the couplings and get the energy
    ising = IsingModel(qubits, dim=DIM, adja_dict=adja_dict, ext_field=field)
    min_eng = ising.energy(-np.ones(qubits))
    if verbose:
        print(f"Ising Model\nJ:{ising.AdjaDict}, h:{ising.ExtField}")

    # check if directory exists
    if save_dir is not None:
        if not Path(save_dir).is_dir():
            print(f"'{save_dir}' not found")
            raise FileNotFoundError

    for shot in shots:
        for steps in maxiter:
            start = datetime.now()
            if verbose:
                print(f"\n\nShots {shot} Maxiter {steps}")

            thetas0: List[np.ndarray] = []
            num_param = qubits * (circ_depth + 1)
            for _ in range(NUM_INIT):
                # generate initial point
                # mean 0 and variance pi
                thetas0.append(rng.random(num_param) * pi)
            # create and eval the circuit
            qc = param_circ(qubits, circ_depth)
            # define optimization class
            vqe = VQE(
                qc,
                ising,
                optimizer=METHOD,
                backend=SIMULATOR_METHOD,
                shots=shot,
                maxiter=None,
                cvar_alpha=alpha,
                verbose=verbose,
            )
            # optimize in parallel
            with Pool(processes=int(cpu_count() / 2)) as pool:
                results = pool.map(vqe.minimize, thetas0)
            # write on json to save results
            filename = f"results_shot{shot}_maxiter{steps}.json"
            with open(filename, "w") as file:
                json.dump(results, file, cls=NumpyArrayEncoder, indent=4)

            stop = datetime.now()
            print(f"\nTotal runtime: {(stop - start).seconds}s")
    return
