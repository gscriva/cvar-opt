import json
from datetime import datetime
from math import pi
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from qiskit.circuit.library import RealAmplitudes

from src.utils import NumpyArrayEncoder, create_ising1d
from src.vqe import VQE

# Use Aer's simulator
SIMULATOR_METHOD = "automatic"
# define specific optimizer
METHOD = "COBYLA"
# limit cpu usage
MAX_CPUS = min(int(cpu_count() / 2), 12)
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
    maxiter: list[int],
    initial_points: list[int],
    noise_model: bool = False,
    type_ising: str = "ferro",
    seed: int = 42,
    alpha: int = 25,
    save_dir: str = None,
    verbose: int = 0,
) -> None:
    start = datetime.now()

    # define generator for initial point
    rng = np.random.default_rng(seed=SEED)
    thetas0: list[np.ndarray] = []
    num_param = qubits * (circ_depth + 1)
    # define initial_points different starting points
    if len(initial_points) == 1:
        initial_points.insert(0, 0)
    for i in range(initial_points[1]):
        if i < initial_points[0]:
            _ = rng.uniform(-2 * pi, 2 * pi, num_param)
            continue
        # generate initial points
        # uniform in [-2pi,2pi]
        thetas0.append(rng.uniform(-2 * pi, 2 * pi, num_param))

    # hamiltonian is defined with +
    # following http://spinglass.uni-bonn.de/ notation
    ising, global_min = create_ising1d(qubits, DIM, type_ising, H_FIELD, seed)
    print(ising)
    print(f"J: {ising.adj_dict}\nh: {ising.h_field}\n")

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
            print(
                f"\nShots: {shot}\tMaxiter: {steps}\tInitial Points: {initial_points}"
            )
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
                noise_model=noise_model if noise_model else None,
                shots=shot,
                maxiter=steps,
                alpha=alpha,
                global_min=global_min,
                verbose=verbose,
            )
            print(vqe)

            # optimize initial_points instances in parallel
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
            per_cpu_runs = (initial_points[1] - initial_points[0]) / MAX_CPUS
            delta_it = (stop_it - start_it) / per_cpu_runs
            print(f"\nSave results in {filename}")
            print(f"Per CPU time: {delta_it.total_seconds():.2f}s\n")

    # report total execution time
    stop = datetime.now()
    print(f"Wall time: {stop - start}s\n")
    return
