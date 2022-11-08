from math import isclose
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
        history (dict(str, list(float))): Dict with the minimum and the loss reach at every iteration.

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
        verbose: int = 0,
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
            verbose (int, optional): Set verbosity level. Defaults to 0.
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
        # save minimum value and loss at each iteration
        self._history: dict[str, list[float]] = {"min": [], "loss": []}

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

    @property
    def history(self) -> dict[str, list[float]]:
        return self._history

    def _update_ansatz(self, parameters: np.ndarray) -> QuantumCircuit:
        # assign current thetas to the parametric circuit
        # in_place=False to not overwrite the parametric circuit
        circuit = self.ansatz.assign_parameters(parameters)
        return circuit

    def _eval_ansatz(self, circuit: QuantumCircuit) -> Counts:
        # Execute the circuit with fixed params
        job = self.simulator.run(circuit, shots=self.shots)
        # Grab results from the job
        result = job.result().get_counts()
        # return counts
        return result

    def _compute_expectation(
        self, counts: Counts, last: bool = False
    ) -> tuple[np.ndarray, np.ndarray, float]:
        # init outputs
        energies: list[float] = []
        eng_opt: float = np.inf
        sample_opt: np.ndarray = np.empty(self.expectation.spins)
        for sample, count in counts.items():
            # cast the sample in np.ndarray
            # and in ising notation {+1,-1}
            sample_ising = np.asarray(list(sample), dtype=int) * 2 - 1
            # compute the energy of the sample
            energy = self.expectation.energy(sample_ising)
            energies.extend([energy] * count)
            if last:
                if self._verbose > 1:
                    print(f"{sample} [{count}] energy: {energy}")
                if energy < eng_opt:
                    eng_opt = energy
                    sample_opt = np.copy(sample_ising)
        return np.asarray(energies), sample_opt, eng_opt

    def _update_history(self, min_energy: float, loss: float) -> None:
        self._history["loss"].append(loss)
        self._history["min"].append(min_energy)

    def _minimize_func(self, parameters: np.ndarray) -> float:
        # update the circuit and get results
        circuit = self._update_ansatz(parameters)
        counts = self._eval_ansatz(circuit)
        # compute energy according to the ising model
        energies, _, _ = self._compute_expectation(counts)
        # get the alpha-th percentile
        cvar = np.percentile(energies, self.alpha)
        # sum all the energies below cvar
        loss = energies[energies <= cvar].mean()
        # store results of each iteration
        self._update_history(float(energies.min()), float(loss))
        if self._verbose > 1:
            print(f"loss: {loss:4.3}\tmin: {energies.min():4.3}\t{counts}")
        return float(loss)

    def _eval_result(self, opt_res: optimize.OptimizeResult) -> dict[str, Any]:
        # update the circuit and get results
        # usinig the optimized results
        circuit = self._update_ansatz(opt_res.x)
        counts = self._eval_ansatz(circuit)
        # get the min energy and its relative sample
        _, sample_opt, eng_opt = self._compute_expectation(counts, last=True)
        # collect information for the results dict
        success: bool = isclose(eng_opt, self.global_min)
        ever_found: bool = isclose(
            np.asarray(self.history["min"]).min(), self.global_min
        )
        where_found: int = int(np.argmin(self.history["min"]))
        # print job summary
        if self._verbose > 0:
            print(
                f"Minimum: {((sample_opt + 1) / 2).astype(int)} Energy: {eng_opt:6.2f}\tGlobal minimum: {success} ({ever_found} [{where_found}])"
            )
            if self._verbose > 1:
                print("\n")
        # save results
        result = dict(opt_res)
        # update dict with the operator |=
        result |= {
            "sample_opt": sample_opt,
            "eng_opt": eng_opt,
            "success": success,
            "ever_found": ever_found,
            "history": self.history,
            "shots": self.shots,
        }
        return result

    def minimize(self, initial_point: np.ndarray) -> dict[str, Any]:
        # reset history
        # needed if the same VQE has been used for more than one job
        self._history: dict[list[float], list[float]] = {"min": [], "loss": []}
        # start initialization
        opt_res = optimize.minimize(
            self._minimize_func,
            initial_point,
            method=self.optimizer,
            options={"maxiter": self.maxiter} if self.maxiter is not None else None,
        )
        result = self._eval_result(opt_res)
        result["initial_point"] = initial_point
        return result
