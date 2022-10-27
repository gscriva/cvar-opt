from math import pi
from pathlib import Path
from typing import List

import numpy as np
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize

from src.utils.ising import IsingModel
from src.utils.vqe import create_circ, obj_func, qc_eval

# Use Aer's qasm_simulator
SIMULATOR = QasmSimulator()
# define specific optimizer
METHOD = "COBYLA"
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
):

    # define generator for initial point
    rng = np.random.default_rng(seed=seed)

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
    if verbose:
        print(f"J:{ising.AdjaDict}, h:{ising.ExtField}")

    # check if directory exists
    if save_dir is not None:
        if not Path(save_dir).is_dir():
            print(f"'{save_dir}' not found")
            raise FileNotFoundError

    for shot in shots:
        for steps in maxiter:
            # generate initial point
            # p of the article
            num_param = qubits * (circ_depth + 1)
            thetas0 = rng.random(num_param) * pi
            # create and eval the circuit
            qc = create_circ(thetas0, qubits, circ_depth)
            if verbose:
                qc.draw()
            counts = qc_eval(qc, SIMULATOR, shot)
            if verbose:
                plot_histogram(counts)

            # TODO it has to be an object
            # eval the loss for the initial point
            obj_func(
                thetas0,
                SIMULATOR,
                qubits,
                circ_depth,
                shot,
                ising,
                alpha=alpha,
                verbose=verbose,
            )

            # start the optimization
            res = minimize(
                obj_func,
                thetas0,
                args=(SIMULATOR, qubits, circ_depth, shot, ising, alpha),
                method=METHOD,
                options={"maxiter": steps, "disp": True},
            )

            if verbose:
                # compute and eval the results
                qc = create_circ(res.x, qubits, circ_depth)
                counts = qc_eval(qc, SIMULATOR, shot)
                if verbose:
                    plot_histogram(counts)
