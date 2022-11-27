from typing import Optional

import numpy as np
from numba import jit
from scipy import sparse


class Ising:
    """Class for dealing with a Ising problem and compute samples' energy.

    Attributes:
        adja_dict (dict[tuple[int, int], float]): Dictionary with the couplings J.
        ext_field (Optional[np.ndarray], optional): Array with the external field. Defaults to None.
        dim (int): Dimensions of the lattice.
        spin_side (int): Spins per lattice's side.
        spins (int): Total numbers of spins.
        adja_matrix (sparse.coo_array): Adjacency matrix in sparse representation.
        h_field (np.ndarray): External h field.

    Methods:
        energy(sample=np.ndarray) -> float:
            Compute the energy of the input sample.
        savetxt() -> None:
            Save the couplings in a .txt file.
        loadtxt(spin_side=int, dim=int, txt_path=str, h_field=Optional[np.ndarray]) -> None:
            Instantiate an Ising class problem loading the ajacency dictionary from a .txt file.
    """

    __MIN_DIM = 0

    def __init__(
        self,
        spin_side: int,
        dim: int,
        adja_dict: dict[tuple[int, int], float],
        h_field: Optional[np.ndarray] = None,
    ):
        self._spin_side = spin_side
        if dim > self.__MIN_DIM:
            self._dim = dim
        else:
            raise ValueError("Non-positive dimension\n")

        self._adja_dict = {}
        self._check(adja_dict)

        self._h_field = h_field if h_field is not None else np.zeros(self.spins)
        assert (
            self.h_field.shape[0] == self.spins
        ), f"External h field shape {h_field.shape[0]} does not match spins number {self.spins}"

        self._create_adja_matrix()

    def __str__(self) -> str:
        return f"""\nIsing Model 
        Spins N={self.spins}
        Dimension D={self.dim}
        Connectivity: z = {(self.adja_matrix.toarray() != 0).sum(-1).max()}

        """

    @classmethod
    def loadtxt(
        cls,
        spin_side: int,
        dim: int,
        txt_path: str,
        h_field: Optional[np.ndarray] = None,
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
        return cls(spin_side, dim, adjadict, h_field)

    @property
    def adja_dict(self) -> dict[tuple[int, int], float]:
        return self._adja_dict

    @property
    def h_field(self) -> np.ndarray:
        return self._h_field

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
    def adja_matrix(self) -> sparse.coo_array:
        return self._adja_matrix

    def _check(self, adja_dict: dict[tuple[int, int], float]) -> None:
        for key, val in adja_dict.items():
            # do not save J_ij if J_ji is present
            try:
                if self._adja_dict[(key[1], key[0])] == val:
                    continue
            except:
                pass
            self._adja_dict.update({key: val})

    def _create_adja_matrix(self) -> None:
        i_vec, j_vec = [], []
        couplings = []
        for (i, j), coupling in self._adja_dict.items():
            i_vec.extend([i, j])
            j_vec.extend([j, i])
            couplings.extend([coupling, coupling])
        self._adja_matrix = sparse.coo_array(
            (couplings, (i_vec, j_vec)), shape=(self.spins, self.spins)
        )

    def energy(self, sample: np.ndarray) -> float:
        eng = np.einsum("ij,i,j", self.adja_matrix.toarray(), sample, sample) / 2
        eng += np.einsum("i,i", self.h_field, sample)
        return eng

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
