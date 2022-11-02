from typing import List, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from scipy.optimize import OptimizeResult, minimize

from src.utils.ising import IsingModel


class VQE:
    """Class to define and solve an optimization problem using VQE.

    Args:
        ansatz (QuantumCircuit): Parametric quantum circuit, used as variational ansatz.
        expectation (IsingModel): Ising instance.
        optimizer (str, optional): Method for the classical optimization. Defaults to "COBYLA".
        backend (str, optional): Simulator for the circuit evaluation. Defaults to "automatic".
        shots (int, optional): Number of sample from the circuit. Defaults to 1024.
        maxiter (Optional[int], optional): Maximum number of iteration of the classical optimizer,
            if None it runs until convergence up to a tollerance. Defaults to None.
        cvar_alpha (int, optional): Alpha quantile used in CVaR-VQE. Defaults to 25.
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        expectation: IsingModel,
        optimizer: str = "COBYLA",
        backend: str = "automatic",
        shots: int = 1024,
        maxiter: Optional[int] = None,
        cvar_alpha: int = 25,
    ) -> None:

        self.ansatz = ansatz
        self.expectation = expectation
        self.optimizer = optimizer
        self.simulator: AerSimulator = AerSimulator(method=backend)
        self.shots = shots
        self.maxiter = maxiter
        self.cvar_alpha = cvar_alpha

    def _update_ansatz(self, parameters: np.ndarray) -> QuantumCircuit:
        # assign current thetas to the circuit params
        circuit = self.ansatz.assign_parameters(parameters)
        return circuit

    def _eval_ansatz(self, circuit: QuantumCircuit) -> None:
        # compile the circuit down to low-level instructions
        compiled_qc = transpile(circuit, self.simulator)
        # Execute the circuit with fixed params
        job = self.simulator.run(compiled_qc, shots=self.shots)
        # Grab results from the job
        result = job.result()
        # return counts
        return result.get_counts(compiled_qc)

    def _compute_expectation(self, counts) -> np.ndarray:
        energies: List[float] = []
        for sample, count in counts.items():
            # cast the sample in np.ndarray
            # and in ising notation {+1,-1}
            sample_ising = np.asarray(list(sample), dtype=int) * 2 - 1
            # compute the energy of the sample
            energy = self.expectation.energy(sample_ising)
            energies.extend([energy] * count)
        return np.asarray(energies)

    def _minimize_func(self, parameters: np.ndarray) -> None:
        # update the circuit and get results
        circuit = self._update_ansatz(parameters)
        counts = self._eval_ansatz(circuit)
        # compute energy according to the ising model
        energies = self._compute_expectation(counts)
        # get the alpha-th percentile
        cvar = np.percentile(energies, self.cvar_alpha)
        # sum all the energies below cvar and return
        return energies[energies <= cvar].sum() / self.shots

    def minimize(self, initial_point: np.ndarray) -> OptimizeResult:
        res = minimize(
            self._minimize_func,
            initial_point,
            method=self.optimizer,
            options={"maxiter": self.maxiter},
        )
        return res
