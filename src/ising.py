from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numba import jit
from scipy import sparse


class IsingModel:
    def __init__(
        self,
        spin_side: int = None,
        dim: int = 2,
        adja_dict: Optional[Dict[tuple, float]] = None,
        ext_field: Optional[np.ndarray] = None,
    ):
        self.AdjaDict: Dict[Tuple(int, int), float] = {}
        self.ExtField = ext_field
        self.Dim = dim
        self.SpinSide = spin_side

        if adja_dict is not None:
            self.AdjaDict = adja_dict

        self.NeighboursCouplings: Optional[np.ndarray] = None
        self.Connectivity: Optional[Union[int, List[int]]] = None
        self.MaxNeighbours: Optional[int] = None
        self.AdjaMatrix: Optional[np.ndarray] = None

    def get_adjadict(self) -> Dict[Tuple[int, int], float]:
        return self.AdjaDict

    def get_neighbours(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.NeighboursCouplings is None:
            self._create_neighbours()
        len_neighbours = np.sum(self.NeighboursCouplings[..., 1] != 0, axis=-1)
        return (
            self.NeighboursCouplings[..., 0].astype(int),
            self.NeighboursCouplings[..., 1],
            len_neighbours,
        )

    def get_adjamatrix(self) -> np.ndarray:
        if self.AdjaMatrix is None:
            self._create_adjamatrix()
        return self.AdjaMatrix.toarray()

    def get_sparse(self) -> sparse.coo.coo_matrix:
        if self.AdjaMatrix is None:
            self._create_adjamatrix()
        return self.AdjaMatrix

    def energy(self, sample: np.ndarray) -> float:
        @jit(nopython=True)
        def compute_eng(
            sample: np.ndarray,
            neighbours: np.ndarray,
            couplings: np.ndarray,
            field: np.ndarray,
            len_neighbours: np.ndarray,
        ) -> float:
            energy = 0
            for i in range(neighbours.shape[0]):
                for j in range(len_neighbours[i]):
                    energy += sample[i] * (sample[neighbours[i, j]] * couplings[i, j])
                energy += sample[i] * field[i] * 2
            return energy / 2

        neighbours, couplings, len_neighbours = self.get_neighbours()
        energy = compute_eng(
            sample, neighbours, couplings, self.ExtField, len_neighbours
        )
        return energy

    def savetxt(self) -> None:
        assert bool(
            self.AdjaDict
        ), "The connectivity dictionary is empty, instantiate first."
        txtarr = []
        for (i, j), coupling in self.AdjaDict.items():
            # see http://mcsparse.uni-bonn.de/spinglass/ for the format style
            # see loadtxt as well
            txtarr.append([i + 1, j + 1, coupling])
        np.savetxt(f"couplings-{self.SpinSide}spins", txtarr)

    def loadtxt(self, txt_path: str) -> None:
        txt_file = np.loadtxt(txt_path)
        # see http://spinglass.uni-bonn.de/ for the input format
        # for 2D lattice the elements are numbered row-wise
        # for 3D lattice the elements are numbered sequentially
        # layer by layer starting with index 1
        # and so on
        adjadict = {}
        for i in range(txt_file.shape[0]):
            adjadict.update(
                {(int(txt_file[i, 0] - 1), int(txt_file[i, 1] - 1)): txt_file[i, 2]}
            )
        self.AdjaDict = adjadict

    def _create_neighbours(self) -> None:
        if self.MaxNeighbours is None:
            self.MaxNeighbours = self.SpinSide**self.Dim
        self.NeighboursCouplings = np.zeros(
            (self.SpinSide**self.Dim, self.MaxNeighbours, 2)
        )
        for spins, coupling in self.AdjaDict.items():
            idxs_nghb = np.where(self.NeighboursCouplings[spins[0], :, 1] == 0)[0]
            self.NeighboursCouplings[spins[0], idxs_nghb[0]] = spins[1], coupling
            idxs_nghb = np.where(self.NeighboursCouplings[spins[1], :, 1] == 0)[0]
            self.NeighboursCouplings[spins[1], idxs_nghb[0]] = spins[0], coupling
        mask = np.where((self.NeighboursCouplings == 0).all(0).all(1))
        # the max number of neighbours is always the minimum of zeros
        # elements in the matrix
        self.MaxNeighbours = mask[0]
        # reduce the dimension of the couplings
        self.NeighboursCouplings = np.delete(self.NeighboursCouplings, mask, axis=1)

    def _create_adjamatrix(self) -> None:
        i_vec, j_vec = [], []
        couplings = []
        for (i, j), coupling in self.AdjaDict.items():
            i_vec.extend([i, j])
            j_vec.extend([j, i])
            couplings.extend([coupling, coupling])
        self.AdjaMatrix = sparse.coo_matrix((couplings, (i_vec, j_vec)))
