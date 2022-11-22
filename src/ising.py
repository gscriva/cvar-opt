from typing import Optional, Tuple

import numpy as np
from numba import jit
from scipy import sparse


class Ising:
    """Class for dealing with a Ising problem and compute samples' energy.

    Attributes:
        adja_dict (dict[tuple[int, int], float]): Dictionary with the couplings.
        ext_field (Optional[np.ndarray], optional): Array with the external field. Defaults to None.
        dim (int): Dimensions of the lattice.
        spin_side (int): Spins per lattice's side.
        spins (int): Total numbers of spins.
        neighbours_repr (tuple[np.ndarray, np.ndarray, np.ndarray]): Tuple with:
                neighbours for each spin, couplings with each neighbour and connectivity of each spin.
        adja_matrix (sparse.coo.coo_matrix): Adjacency matrix in sparse representation.

    Methods:
        energy(sample=np.ndarray) -> float:
            Compute the energy of the input sample.
        savetxt() -> None:
            Save the couplings in a .txt file.
        loadtxt(spin_side=int, dim=int, txt_path=str, ext_field=Optional[np.ndarray]) -> None:
            Instantiate an Ising class problem loading the ajacency dictionary from a .txt file.
    """

    __MIN_DIM = 0

    def __init__(
        self,
        spin_side: int,
        dim: int,
        adja_dict: dict[tuple[int, int], float],
        ext_field: Optional[np.ndarray] = None,
    ):
        self._spin_side = spin_side
        if dim > self.__MIN_DIM:
            self._dim = dim
        else:
            raise ValueError("Non-positive dimension\n")
        self._max_neighbours = self._spin_side**self._dim
        self._adja_dict = adja_dict
        self._ext_field = ext_field if ext_field is not None else np.zeros(self.spins)
        assert (
            self.ext_field.shape[0] == self.spins
        ), f"ext_field shape {ext_field.shape[0]} does not match spins number {self.spins}"
        # save memory and define it
        # only when you need it
        self._neighbours_repr = None
        self._create_adja_matrix()

    def __str__(self) -> str:
        return f"""\nIsing Model 
        Spins N={self.spins}
        Dimension D={self.dim}
        Connectivity: z={self.neighbours_repr[2].max()}
        """

    @classmethod
    def loadtxt(
        cls,
        spin_side: int,
        dim: int,
        txt_path: str,
        ext_field: Optional[np.ndarray] = None,
    ) -> None:
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
        return cls(spin_side, dim, adjadict, ext_field)

    @property
    def adja_dict(self) -> dict[Tuple[int, int], float]:
        return self._adja_dict

    @property
    def ext_field(self) -> np.ndarray:
        return self._ext_field

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def spin_side(self) -> int:
        return self._spin_side

    @property
    def spins(self) -> int:
        return self.spin_side**self.dim

    @property
    def neighbours_repr(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._neighbours_repr is None:
            self._create_neighbours_repr()
        return self._neighbours_repr

    @property
    def adja_matrix(self) -> sparse.coo.coo_matrix:
        return self._adja_matrix

    def _create_neighbours_repr(self) -> None:
        neighbours_couplings = np.zeros((self._max_neighbours, self._max_neighbours, 2))
        for spins, coupling in self._adja_dict.items():
            idxs_nghb = np.where(neighbours_couplings[spins[0], :, 1] == 0)[0]
            neighbours_couplings[spins[0], idxs_nghb[0]] = spins[1], coupling
            idxs_nghb = np.where(neighbours_couplings[spins[1], :, 1] == 0)[0]
            neighbours_couplings[spins[1], idxs_nghb[0]] = spins[0], coupling
        mask = np.where((neighbours_couplings == 0).all(0).all(1))
        # the max number of neighbours is always
        # the minimum of zeros elements in the matrix
        self._max_neighbours = mask[0]
        # reduce the dimension of the couplings
        neighbours_couplings = np.delete(neighbours_couplings, mask, axis=1)
        len_neighbours = np.sum(neighbours_couplings[..., 1] != 0, axis=-1)
        self._neighbours_repr = (
            neighbours_couplings[..., 0].astype(int),
            neighbours_couplings[..., 1],
            len_neighbours,
        )

    def _create_adja_matrix(self) -> None:
        i_vec, j_vec = [], []
        couplings = []
        for (i, j), coupling in self._adja_dict.items():
            i_vec.extend([i, j])
            j_vec.extend([j, i])
            couplings.extend([coupling, coupling])
        self._adja_matrix = sparse.coo_matrix(
            (couplings, (i_vec, j_vec)), shape=(self.spins, self.spins)
        )

    # energy compouted with numba
    # it save memory but takes long
    # TODO Remove it when sure about time advantage of energy method.
    def numba_energy(self, sample: np.ndarray) -> float:
        # cache=True improves speed in next calls
        @jit(cache=True)
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

        neighbours, couplings, len_neighbours = self.neighbours_repr
        energy = compute_eng(
            sample, neighbours, couplings, self.ext_field, len_neighbours
        )
        return energy

    # TODO use einsum
    def energy(self, sample: np.ndarray) -> float:
        energy = (
            self.adja_matrix.toarray() * sample[..., None] * sample[None, ...]
        ).sum(-1)
        energy += 2 * self.ext_field * sample
        return float(energy.sum() / 2)

    def savetxt(self) -> None:
        assert bool(
            self._adja_dict
        ), "The connectivity dictionary is empty, instantiate first."
        txtarr = []
        for (i, j), coupling in self._adja_dict.items():
            # see http://mcsparse.uni-bonn.de/spinglass/ for the format style
            # see loadtxt as well
            txtarr.append([i + 1, j + 1, coupling])
        np.savetxt(f"couplings-{self.spins}spins", txtarr)
