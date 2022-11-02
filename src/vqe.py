from typing import Any, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.result.counts import Counts
from scipy import optimize

from src.ising import IsingModel


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
        verbose: bool = False,
    ) -> None:

        self.ansatz = ansatz
        self.expectation = expectation
        self.optimizer = optimizer
        self.simulator: AerSimulator = AerSimulator(method=backend)
        self.shots = shots
        self.maxiter = maxiter
        self.cvar_alpha = cvar_alpha
        self.verbose = verbose

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

    def _compute_expectation(self, counts: Counts) -> np.ndarray:
        energies: list[float] = []
        for sample, count in counts.items():
            # cast the sample in np.ndarray
            # and in ising notation {+1,-1}
            sample_ising = np.asarray(list(sample), dtype=int) * 2 - 1
            # compute the energy of the sample
            energy = self.expectation.energy(sample_ising)
            energies.extend([energy] * count)
        return np.asarray(energies)

    def _minimize_func(self, parameters: np.ndarray) -> float:
        # update the circuit and get results
        circuit = self._update_ansatz(parameters)
        counts = self._eval_ansatz(circuit)
        # compute energy according to the ising model
        energies = self._compute_expectation(counts)
        # get the alpha-th percentile
        cvar = np.percentile(energies, self.cvar_alpha)
        # sum all the energies below cvar
        loss = energies[energies <= cvar].mean()
        return float(loss)

    def _eval_result(self, opt_res: optimize.OptimizeResult) -> dict[str, Any]:
        # update the circuit and get results
        # usinig the optimized results
        circuit = self._update_ansatz(opt_res.x)
        counts = self._eval_ansatz(circuit)
        # get the energies
        eng_opt = np.inf
        sample_opt = np.empty(self.expectation.SpinSide)
        for sample in counts.keys():
            # cast the sample in np.ndarray
            # and in ising notation {+1,-1}
            sample_ising = np.asarray(list(sample), dtype=int) * 2 - 1
            # compute the energy of the sample
            eng = self.expectation.energy(sample_ising)
            if eng < eng_opt:
                eng_opt = eng
                sample_opt = np.copy(sample_ising)
        min_eng = self.expectation.energy(-np.ones(self.expectation.SpinSide))
        if self.verbose:
            print(
                f"Found minimum: {((sample_opt + 1) / 2).astype(int)} Energy: {eng_opt:.3f} Global minimum: {eng_opt == min_eng}"
            )
        # save results
        result = dict(opt_res)
        result["sample_opt"] = sample_opt
        result["eng_opt"] = eng_opt
        result["global_min"] = eng_opt == min_eng
        return result

    def minimize(self, initial_point: np.ndarray) -> dict[str, Any]:
        opt_res = optimize.minimize(
            self._minimize_func,
            initial_point,
            method=self.optimizer,
            options={"maxiter": self.maxiter} if self.maxiter is not None else None,
        )
        result = self._eval_result(opt_res)
        result["initial_point"] = initial_point
        return result
