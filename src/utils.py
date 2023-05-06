import itertools
import json
import math
import os

import numpy as np
import qiskit

from . import ising

OPT_T = 0.80


def get_init_points(
    initial_points: list[int],
    num_params: int,
    rng: np.random.Generator,
    opt_parameters: bool,
) -> list[np.ndarray]:
    thetas0: list[np.ndarray] = []
    for i in range(initial_points[1]):
        # skip initial points not needed
        # usefull for resuming jobs
        if i < initial_points[0]:
            _ = rng.uniform(0, 2 * math.pi, num_params)
            continue
        if opt_parameters:
            # Optimized initialization following
            # https://doi.org/10.22331/q-2021-07-01-491
            depth = num_params // 2
            gammas = [i * OPT_T / depth for i in range(1, depth + 1)]
            # betas
            opt_thetas0 = [(1 - i / depth) * OPT_T for i in range(1, depth + 1)]
            # [beta_1, ..., beta_n, gamma_1, ..., gamma_n]
            opt_thetas0.extend(gammas)
            thetas0.append(opt_thetas0)
        else:
            # generate initial points
            # uniform in [0,2*pi]
            thetas0.append(rng.uniform(0, 2 * math.pi, num_params))
    return thetas0


def get_ising_params(
    spins: int,
    h_field: float,
    type_ising: str,
    seed: int,
) -> np.ndarray:
    # hamiltonian is defined with +
    # following http://spinglass.uni-bonn.de/ notation
    if type_ising == "ferro":
        J = -np.ones(spins - 1)
        h = np.zeros(spins) - h_field
    elif type_ising == "binary":
        ising_rng = np.random.default_rng(seed=seed)
        J = (
            ising_rng.integers(
                low=0,
                high=2,
                size=spins - 1,
            )
            * 2
            - 1
        )
        h = np.zeros(spins)
    elif type_ising == "random":
        ising_rng = np.random.default_rng(seed=seed)
        J = ising_rng.normal(
            loc=0.0,
            scale=1.0,
            size=spins - 1,
        )
        h = ising_rng.normal(
            loc=0.0,
            scale=1.0,
            size=spins,
        )
    return J.astype(np.float64), h


def compute_ising_min(model: ising.Ising) -> float:
    energies = []
    for sample in itertools.product([0, 1], repeat=model.spins):
        energies.append(model.energy(np.asarray(sample) * 2 - 1))
    return np.asarray(energies).min()


def create_ising1d(
    spins: int,
    dim: int,
    type_ising: str,
    h_field: float,
    seed: int,
) -> tuple[ising.Ising, float]:
    # hamiltonian is defined with +
    # following http://spinglass.uni-bonn.de/ notation
    # and D-Wave https://docs.dwavesys.com/docs/latest/c_gs_2.html#underlying-quantum-physics
    J, h = get_ising_params(spins, h_field, type_ising, seed)
    adj_dict = {}
    for i in range(spins):
        if i == spins - 1:
            continue
        adj_dict[(i, i + 1)] = J[i]
    # class devoted to set the couplings and get the energy
    model = ising.Ising(spins, dim=dim, adj_dict=adj_dict, h_field=h)
    # exact enumeration would be too expensive for large sizes
    if spins < 22 and (type_ising == "binary" or type_ising == "random"):
        min_eng = compute_ising_min(model)
    elif type_ising == "ferro":
        min_eng = model.energy(-np.ones(model.spins))
    else:
        min_eng = -np.inf
    return model, min_eng


def create_qaoa_ansatz(
    num_qubits: int,
    circ_depth: int,
    hamiltonian: ising.Ising,
    increase_params: bool = False,
) -> qiskit.QuantumCircuit:
    """Create a quantum ansatz inspired by the problem Hamiltonian.
    Setting increase_params=False the ansatz becomes the typical one of QAOA, with 2*circ_depth parameters.
    Otherwise, increase_params=True increases the parameters to (2*num_qubits - 1)*circ_depth.
    See also https://qiskit.org/textbook/ch-applications/qaoa.html.

    Args:
        num_qubits (int): Number of qubits.
        circ_depth (int): Depth of the circuit, minimum 1.
        hamiltonian (ising.Ising): Problem Hamiltonian.
        increase_params (bool, optional): Allows betas and gammas to change in each layer. Defaults to False.

    Returns:
        qiskit.QuantumCircuit: QAOA or QAOA+ ansatz.
    """
    # define circuit
    name = f"QAOA+ p={circ_depth}" if increase_params else f"QAOA p={circ_depth}"
    qc = qiskit.QuantumCircuit(num_qubits, name=name)
    # set initial entalgled state
    for i in range(num_qubits):
        qc.h(i)
    if increase_params:
        # create the parameters for the circuit
        thetas = qiskit.circuit.ParameterVector("θ", (2 * num_qubits) * circ_depth)
    else:
        # calssical QAOA ansatz
        gammas = qiskit.circuit.ParameterVector("γ", circ_depth)
        betas = qiskit.circuit.ParameterVector("β", circ_depth)
    # add circ_depth layers
    for i in range(circ_depth):
        # add R_zz parametric gates
        for j, j_coupling in enumerate(hamiltonian.adj_dict.values()):
            if increase_params:
                qc.rzz(
                    j_coupling * 2 * thetas[(2 * num_qubits) * i + j],
                    j,
                    j + 1,
                )
            else:
                qc.rzz(j_coupling * 2 * gammas[i], j, j + 1)
        qc.barrier()
        # add Rz parametric gates
        for j, h_j in enumerate(hamiltonian.h_field):
            if increase_params:
                qc.rz(h_j * 2 * thetas[(2 * num_qubits) * i + num_qubits - 1 + j], j)
            else:
                qc.rz(h_j * 2 * gammas[i], j)
        # add Rx parametric gates
        for j in range(num_qubits):
            if increase_params:
                qc.rx(-2 * thetas[(2 * num_qubits) * (i + 1) - 1], j)
            else:
                qc.rx(-2 * betas[i], j)
        # do not put barrier in the last iteration
        if i == circ_depth - 1:
            continue
    return qc


def create_ansatz(
    qubits: int,
    circ_depth: int,
    ansatz_type: str,
    hamiltonian: ising.Ising,
    measure: bool,
    verbose: int = 0,
):
    if ansatz_type == "vqe":
        # standard VQE ansatz
        qc = qiskit.circuit.library.RealAmplitudes(
            qubits,
            reps=circ_depth,
            insert_barriers=True,
            entanglement="circular",
            name=f"VQE p={circ_depth}",
        )
    elif ansatz_type == "qaoa" or ansatz_type == "qaoa+":
        increase_params = True if ansatz_type == "qaoa+" else False
        qc = create_qaoa_ansatz(qubits, circ_depth, hamiltonian, increase_params)
    else:
        raise NotImplementedError(f"Ansatz type {ansatz_type} not found")
    if measure:
        qc.measure_all()
    if verbose > 0 and qubits < 8:
        print(qc)
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
                # data are a list with #initial_points elements
                data = json.load(file)
            ever_found = []
            nfevs = []
            # for each shot and iteration param
            # we randomized the initial point
            # to estimate the right probability
            for run in data:
                ever_found.append(run["ever_found"])
                nfevs.append(run["nfev"])
            # compute p('found minimum')
            p_everfound.append(
                np.mean(np.asarray(ever_found, dtype=np.float128))
            )  # float128 to avoid to many zeros
            nfev = int(math.ceil(np.asarray(nfevs).mean()))
            # maxiter*shots = actual number of iteration
            t.append(run["shots"] * nfev)
            s.append(run["shots"])
            it.append(nfev)
        # update list for each number of qubits
        ts.append(t)
        psucc.append(p_everfound)
        shots.append(s)
        nfevs.append(it)
    return psucc, ts, shots, nfevs


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
