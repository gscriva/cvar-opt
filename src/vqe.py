from typing import Any, Optional

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import AerSimulator
from qiskit.result.counts import Counts
from scipy import optimize

from src.ising import IsingModel


class VQE:
    """Class to define and solve an optimization problem using VQE.

    Attributes:
        ansatz (QuantumCircuit): Parametric quantum circuit, used as variational ansatz.
        expectation (IsingModel): Ising instance.
        optimizer (str): Method for the classical optimization. Defaults to "COBYLA".
        backend (str): Simulator for the circuit evaluation. Defaults to "automatic".
        shots (int): Number of sample from the circuit. Defaults to 1024.
        maxiter (Optional[int]): Maximum number of iteration of the classical optimizer,
            if None it runs until convergence up to a tollerance. Defaults to None.
        alpha (int): Alpha quantile used in CVaR-VQE. Defaults to 25.
        global_min (float): Global minimum of the expectation problem. Default to None.
        history (list): List with the minimum reach at every iteration. 

    Methods:
        minimize(initial_point=np.ndarray)
            Run the optimization problem and return results.
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        expectation: IsingModel,
        optimizer: str = "COBYLA",
        backend: str = "automatic",
        shots: int = 1024,
        maxiter: Optional[int] = None,
        alpha: int = 25,
        global_min: float = -np.inf,
        verbose: bool = False,
    ) -> None:
        """Initialization of a VQE instance.

        Args:
            ansatz (QuantumCircuit): Parametric quantum circuit, used as variational ansatz.
            expectation (IsingModel): Ising instance.
            optimizer (str, optional): Method for the classical optimization. Defaults to "COBYLA".
            backend (str, optional): Simulator for the circuit evaluation. Defaults to "automatic".
            shots (int, optional): Number of sample from the circuit. Defaults to 1024.
            maxiter (Optional[int], optional): Maximum number of iteration of the classical optimizer,
                if None it runs until convergence up to a tollerance. Defaults to None.
            alpha (int, optional): Alpha quantile used in CVaR-VQE. Defaults to 25.
            global_min (float, optional): Global minimum of the expectation problem. Default to None.
            verbose (bool, optional): Set verbose mode. Defaltu to False.
        """

        self._ansatz = ansatz
        self._expectation = expectation
        self._optimizer = optimizer
        self._simulator: AerSimulator = AerSimulator(method=backend)
        self._shots = shots
        self._maxiter = maxiter
        self._alpha = alpha
        self._global_min = global_min
        self._verbose = verbose
        # TODO save minimum value, loss at each iteration
        self.history = []

    def __str__(self) -> str:
        return f"""\nVQE instance 
        Ansatz:\n{self.ansatz}
        Optimizer: {self.optimizer}
        Simulator: {self.simulator}
        Shots: {self.shots}
        Maxiter: {self.maxiter}
        Alpha: {self.alpha}
        Global minimum: {self.global_min}
        """

    @property
    def ansatz(self) -> QuantumCircuit:
        return self._ansatz

    @property
    def expectation(self) -> IsingModel:
        return self._expectation

    @property
    def optimizer(self) -> str:
        return self._optimizer

    @property
    def simulator(self) -> AerSimulator:
        return self._simulator

    @property
    def shots(self) -> int:
        return self._shots

    @property
    def maxiter(self) -> Optional[int]:
        return self._maxiter

    @property
    def alpha(self) -> int:
        return self._alpha

    @property
    def global_min(self) -> float:
        return self._global_min

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
            # TODO set verbosity level 1 and 2
            # if self._verbose == 2:
            #     print(f"{sample} [{count}] energy: {energy}")
        return np.asarray(energies)

    def _minimize_func(self, parameters: np.ndarray) -> float:
        # update the circuit and get results
        circuit = self._update_ansatz(parameters)
        counts = self._eval_ansatz(circuit)
        # compute energy according to the ising model
        energies = self._compute_expectation(counts)
        # get the alpha-th percentile
        cvar = np.percentile(energies, self.alpha)
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
        sample_opt = np.empty(self.expectation.spins)
        for sample in counts.keys():
            # cast the sample in np.ndarray
            # and in ising notation {+1,-1}
            sample_ising = np.asarray(list(sample), dtype=int) * 2 - 1
            # compute the energy of the sample
            eng = self.expectation.energy(sample_ising)
            # TODO set verbosity level 1 and 2
            # if self._verbose:
            #     print(f"{sample} energy: {eng}")
            if eng < eng_opt:
                eng_opt = eng
                sample_opt = np.copy(sample_ising)
        if self._verbose:
            print(
                f"Minimum: {((sample_opt + 1) / 2).astype(int)} Energy: {eng_opt:.2f} Global minimum: {eng_opt == self.global_min}"
            )
        # save results
        result = dict(opt_res)
        result["sample_opt"] = sample_opt
        result["eng_opt"] = eng_opt
        result["global_min"] = eng_opt == self.global_min
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
