from json import JSONEncoder

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector


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
