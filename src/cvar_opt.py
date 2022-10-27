from math import pi
from pathlib import Path
from typing import List
import json
from json import JSONEncoder

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
# initial points number
NUM_INIT = 10
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
            # create file to save results
            filename = f"results_shot{shot}_maxiter{steps}.json"
            with open(filename, "w") as file:
                json.dump([], file)

            if verbose:
                print(f"\n\nShot {shot} Maxiter {steps}")

            for _ in range(NUM_INIT):
                # generate initial point
                # p of the article
                num_param = qubits * (circ_depth + 1)
                thetas0 = rng.random(num_param) * pi

                # read json
                with open(filename, "r") as file:
                    data = json.load(file)
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
                    verbose=False,
                )

                # start the optimization
                res = minimize(
                    obj_func,
                    thetas0,
                    args=(SIMULATOR, qubits, circ_depth, shot, ising, alpha),
                    method=METHOD,
                    options={"maxiter": steps, "disp": False},
                )

                qc = create_circ(res.x, qubits, circ_depth)
                counts = qc_eval(qc, SIMULATOR, shot)
                eng_opt = np.inf
                sample_opt = np.empty(qubits)
                for sample in counts.keys():
                    # cast the sample in np.ndarray
                    # and in ising notation {+1,-1}
                    sample_ising = np.asarray(list(sample), dtype=int) * 2 - 1
                    # compute the energy of the sample
                    eng = ising.energy(sample_ising)
                    if eng < eng_opt:
                        eng_opt = eng
                        sample_opt = np.copy(sample_ising)
                if verbose:
                    print(f"Found minimum {sample_opt} energy {eng_opt}")
                    if eng_opt == min_eng:
                        print(f"Global minimum reach!!\n")
                # save results
                result = dict(res)
                result["sample_opt"] = sample_opt
                result["eng_opt"] = eng_opt
                result["global__min"] = eng_opt == min_eng
                # update json
                data.append(result)

                with open(filename, "w") as file:
                    json.dump(data, file, cls=NumpyArrayEncoder, indent=4)

                if verbose:
                    print(result)
                    if eng_opt == min_eng:
                        print(f"Global minimum reach!!\n")


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return JSONEncoder.default(self, obj)
