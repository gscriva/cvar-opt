from math import pi
from pathlib import Path
from typing import List
import json
from json import JSONEncoder
from multiprocessing import Pool

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector
from qiskit.providers.aer import AerSimulator
from qiskit.visualization import plot_histogram
from scipy.optimize import minimize

from src.utils.ising import IsingModel
from src.utils.func_vqe import create_circ, obj_func, qc_eval
from src.vqe import VQE

# Use Aer's qasm_simulator
SIMULATOR_METHOD = "automatic"
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
    use_class: bool = True,
):

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
            # create file to save results
            filename = f"results_shot{shot}_maxiter{steps}.json"
            with open(filename, "w") as file:
                json.dump([], file)

            if verbose:
                print(f"\n\nShots {shot} Maxiter {steps}")

            thetas0: List[np.ndarray] = []
            for _ in range(NUM_INIT):
                # generate initial point
                # mean 0 and variance pi
                num_param = qubits * (circ_depth + 1)
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
                maxiter=maxiter,
                cvar_alpha=alpha,
            )
            # read json
            with open(filename, "r") as file:
                data = json.load(file)
            if use_class:
                with Pool(process=) as pool:
                    res = pool.map(vqe.minimize, thetas0)
                    print(res)
            else:
                # generate initial point
                # mean 0 and variance pi
                num_param = qubits * (circ_depth + 1)
                thetas0 = rng.random(num_param) * pi
                for _ in range(NUM_INIT):
                    # create and eval the circuit
                    qc = create_circ(thetas0[0], qubits, circ_depth)
                    if verbose:
                        qc.draw()
                        counts = qc_eval(qc, simulator, shot)
                        plot_histogram(counts)

                    # TODO it has to be an object
                    # eval the loss for the initial point
                    obj_func(
                        thetas0,
                        simulator,
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
                        args=(simulator, qubits, circ_depth, shot, ising, alpha),
                        method=METHOD,
                        options={"maxiter": steps, "disp": False},
                    )

                qc = create_circ(res.x, qubits, circ_depth)
                counts = qc_eval(qc, simulator, shot)
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
                    print(
                        f"Found minimum: {sample_opt} Energy: {eng_opt} Global minimum: {eng_opt == min_eng}"
                    )
                # save results
                result = dict(res)
                result["sample_opt"] = sample_opt
                result["eng_opt"] = eng_opt
                result["global_min"] = eng_opt == min_eng
                # update json
                data.append(result)

                with open(filename, "w") as file:
                    json.dump(data, file, cls=NumpyArrayEncoder, indent=4)


# TODO replace with https://qiskit.org/documentation/stubs/qiskit.circuit.library.TwoLocal.html#twolocal
def param_circ(num_qubits: int, circ_depth: int) -> QuantumCircuit:
    # define circuit
    qubits = QuantumRegister(num_qubits)
    cbits = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(qubits, cbits)
    # create a parametrized circuit
    thetas = ParameterVector("theta", num_qubits * (circ_depth + 1))
    # add first layer
    for j in range(num_qubits):
        qc.ry(thetas[j], j)
    qc.barrier()

    for i in range(circ_depth):
        for j in range(num_qubits - 1):
            qc.cx(j, j + 1)
        qc.cx(0, num_qubits - 1)
        qc.barrier()

        for j in range(num_qubits):
            qc.ry(thetas[(1 + i) * num_qubits + j], j)
        qc.barrier()

        if i == circ_depth - 1:
            # Map the quantum measurement to the classical bits
            qc.measure(qubits, cbits)
            continue
    return qc


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return JSONEncoder.default(self, obj)
