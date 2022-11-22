import json
from datetime import datetime
from math import pi
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from qiskit.circuit.library import RealAmplitudes

from src.utils import NumpyArrayEncoder, create_ising1d, get_ising
from src.vqe import VQE

# Use Aer's simulator
SIMULATOR_METHOD = "automatic"
# define specific optimizer
METHOD = "COBYLA"
# initial points number
NUM_INIT = 1000
# limit cpu usage
MAX_CPUS = min(int(cpu_count() / 2), 12)
# dimension of the ising model
# and external field
DIM = 1


def cvar_opt(
    qubits: int,
    circ_depth: int,
    shots: list[int],
    maxiter: list[int],
    type_ising: str = "ferro",
    seed: int = 42,
    alpha: int = 25,
    save_dir: str = None,
    verbose: int = 0,
) -> None:
    start = datetime.now()
    # define generator for initial point
    rng = np.random.default_rng(seed=seed)

    J, h = get_ising(qubits, ising_type=type_ising, rng=rng)
    # hamiltonian is defined with +
    # following http://spinglass.uni-bonn.de/ notation
    ising, global_min = create_ising1d(qubits, DIM, J, h)
    print(ising)
    print(f"J: {ising.adja_dict}\nh: {ising.ext_field}\n")

    # check if directory exists
    if save_dir is not None:
        if not Path(save_dir).is_dir():
            print(f"'{save_dir}' not found")
            raise FileNotFoundError
    else:
        save_dir = Path().absolute()

    for shot in shots:
        for steps in maxiter:
            start_it = datetime.now()
            print(f"\nShots: {shot}\tMaxiter: {steps}\tInitial Points: {NUM_INIT}")

            thetas0: list[np.ndarray] = []
            num_param = qubits * (circ_depth + 1)
            # define NUM_INIT different starting points
            for _ in range(NUM_INIT):
                # generate initial points
                # uniform in [-2pi,2pi]
                thetas0.append(rng.uniform(-2 * pi, 2 * pi, num_param))

            # create the circuit
            # standard VQE ansatz
            qc = RealAmplitudes(
                qubits,
                reps=circ_depth,
                insert_barriers=True,
                entanglement="circular",
            )
            # measure all qubits at the end of the circuit
            qc.measure_all()

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
            print(vqe)

            # optimize NUM_INIT instances in parallel
            # up to MAX_CPUS available on the system
            with Pool(processes=MAX_CPUS) as pool:
                results = pool.map(vqe.minimize, thetas0)

            # write on json to save results
            filename = f"{save_dir}/shots{str(shot).zfill(4)}_maxiter{str(steps).zfill(3)}.json"
            with open(filename, "w") as file:
                json.dump(results, file, cls=NumpyArrayEncoder, indent=4)

            # report iteration execution time
            stop_it = datetime.now()
            # normalize per CPUs and iterations
            delta_it = (stop_it - start_it) / (NUM_INIT / MAX_CPUS)
            print(f"\nSave results in {filename}")
            print(f"Iteration runtime: {delta_it.total_seconds():.2f}s\n")

    # report total execution time
    stop = datetime.now()
    print(f"Total runtime: {stop - start}s\n")
    return
