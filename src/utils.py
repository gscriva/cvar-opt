import json
import os
from json import JSONEncoder
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from src.ising import Ising

EXT_FIELD = 0.05


def get_ising(spins: int, ising_type: str, rng: np.random.Generator) -> np.ndarray:
    if ising_type == "ferro":
        J = -np.ones(spins)
        h = np.zeros(spins) + EXT_FIELD
    elif ising_type == "binary":
        J = (
            rng.integers(
                0,
                2,
                size=spins,
            )
            * 2
            - 1
        )
        h = np.zeros(spins)
    return J.astype(np.float64), h


def create_ising1d(
    spins: int,
    dim: int,
    J: np.ndarray,
    h: np.ndarray,
) -> tuple[Ising, float]:
    # hamiltonian is defined with +
    # following http://spinglass.uni-bonn.de/ notation
    adja_dict = {}
    field = np.zeros(spins) + h
    for i in range(spins):
        if i == spins - 1:
            continue
        adja_dict[(i, i + 1)] = J[i]
    # class devoted to set the couplings and get the energy
    ising = Ising(spins, dim=dim, adja_dict=adja_dict, ext_field=field)
    # TODO if the model is not ferro this is wrong,
    # compute the real minimun exact diagonalization
    min_eng = ising.energy(-np.ones(spins))
    return ising, min_eng


# TODO replace with https://qiskit.org/documentation/stubs/qiskit.circuit.library.TwoLocal.html#twolocal
def param_circ(num_qubits: int, circ_depth: int) -> QuantumCircuit:
    # define circuit
    qc = QuantumCircuit(num_qubits)
    # create a parameter for the circuit
    thetas = ParameterVector("theta", num_qubits * (circ_depth + 1))
    # add first layer
    for j in range(num_qubits):
        qc.ry(thetas[j], j)
    qc.barrier()
    # add other circ_depth layers
    for i in range(circ_depth):
        # add cnot gates
        for j in range(num_qubits - 1):
            qc.cx(j, j + 1)
        qc.cx(0, num_qubits - 1)
        qc.barrier()
        # add Ry parametric gates
        for j in range(num_qubits):
            qc.ry(thetas[(1 + i) * num_qubits + j], j)
        # do not put barrier in the last iteration
        if i == circ_depth - 1:
            continue
        qc.barrier()
    # measure all the qubits
    qc.measure_all()
    return qc


def collect_results(
    qubits: np.ndarray, circ_depth: int
) -> tuple[list, list, list, list]:
    ts = []
    shots = []
    nfevs = []
    psucc = []
    for qubit in qubits:
        dir_path = f"results/N{qubit}/p{circ_depth}/"
        print(f"directory: {dir_path}")
        # init list
        p_everfound = []
        t = []
        s = []
        it = []
        # for each number of qubits
        # we have several number of shots and iterations
        for filename in sorted(os.listdir(dir_path)):
            filename = dir_path + filename
            with open(filename, "r") as file:
                # print(filename)
                data = json.load(file)
            ever_found = []
            # for each shot and iteration param
            # we randomized the initial point
            # to estimate the right probability
            for run in data:
                ever_found.append(run["ever_found"])
            # compute p('found minimum')
            p_everfound.append(
                np.mean(np.asarray(ever_found, dtype=np.float128))
            )  # float128 to avoid to many zeros
            # maxiter*shots = actual number of iteration
            t.append(run["shots"] * run["nfev"])
            s.append(run["shots"])
            it.append(run["nfev"])
        # update list for each number of qubits
        ts.append(t)
        psucc.append(p_everfound)
        shots.append(s)
        nfevs.append(it)
    return psucc, ts, shots, nfevs


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return JSONEncoder.default(self, obj)
