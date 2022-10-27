from typing import Any, Dict, List

import numpy as np
from qiskit import (
    ClassicalRegister,
    QuantumCircuit,
    QuantumRegister,
    transpile,
)
from qiskit_aer.backends.aerbackend import AerBackend

from src.utils.ising import IsingModel


# TODO replace with https://qiskit.org/documentation/stubs/qiskit.circuit.library.TwoLocal.html#twolocal
def create_circ(thetas: np.ndarray, num_qubits: int, circ_depth: int) -> QuantumCircuit:
    # define circuit
    qubits = QuantumRegister(num_qubits)
    cbits = ClassicalRegister(num_qubits)
    qc = QuantumCircuit(qubits, cbits)
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


def qc_eval(qc: QuantumCircuit, simulator: AerBackend, shots: int) -> Dict[str, int]:
    # compile the circuit down to low-level QASM instructions
    # supported by the backend (not needed for simple circuits)
    compiled_qc = transpile(qc, simulator)
    # Execute the circuit on the qasm simulator
    job = simulator.run(compiled_qc, shots=shots)
    # Grab results from the job
    result = job.result()
    # return counts
    return result.get_counts(compiled_qc)


# TODO this should be a class, namely VQE
def obj_func(
    thetas: np.ndarray,
    simulator: Any,
    qubits: int,
    circ_depth: int,
    shots: int,
    ising: IsingModel,
    alpha=100,
    verbose=False,
) -> float:
    qc = create_circ(thetas, qubits, circ_depth)
    # get the results from the circuit
    counts = qc_eval(qc, simulator, shots)

    energies: List[float] = []
    for sample, count in counts.items():
        # cast the sample in np.ndarray
        # and in ising notation {+1,-1}
        sample_ising = np.asarray(list(sample), dtype=int) * 2 - 1
        # compute the energy of the sample
        energy = ising.energy(sample_ising)
        energies.extend([energy] * count)
        if verbose:
            print(f"Sample {sample_ising} energy {energy} counted {count} times")

    energies = np.asarray(energies)
    # get the alpha-th percentile
    cvar = np.percentile(energies, alpha)
    # sum all the energies below cvar
    loss_cvar = energies[energies <= cvar].sum()

    if verbose:
        print(f"\nLoss: {loss_cvar/shots} CVaR: {cvar}")
    return loss_cvar / shots
