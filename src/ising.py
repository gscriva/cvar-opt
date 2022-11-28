from typing import Optional

import numpy as np


class Ising:
    """Class for dealing with a Ising problem and compute samples' energy.

    Attributes:
        adj_dict (dict[tuple[int, int], float]): Dictionary with the couplings J.
        ext_field (Optional[np.ndarray], optional): Array with the external field. Defaults to None.
        dim (int): Dimensions of the lattice.
        spin_side (int): Spins per lattice's side.
        spins (int): Total numbers of spins.
        adj_matrix (np.ndarray): Adjacency matrix in sparse representation.
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
        adj_dict: dict[tuple[int, int], float],
        h_field: Optional[np.ndarray] = None,
    ):
        self._spin_side = spin_side
        if dim > self.__MIN_DIM:
            self._dim = dim
        else:
            raise ValueError("Non-positive dimension\n")

        self._adj_dict = {}
        self._check(adj_dict)

        self._h_field = h_field if h_field is not None else np.zeros(self.spins)
        assert (
            self.h_field.shape[0] == self.spins
        ), f"External h field shape {h_field.shape[0]} does not match spins number {self.spins}"

        self._create_adj_matrix()

    def __str__(self) -> str:
        return f"""\nIsing Model 
        Spins N={self.spins}
        Dimension D={self.dim}
        Connectivity: z = {(self.adj_matrix != 0).sum(-1).max()}

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
    def adj_dict(self) -> dict[tuple[int, int], float]:
        return self._adj_dict

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
    def adj_matrix(self) -> np.ndarray:
        return self._adj_matrix

    def _check(self, adj_dict: dict[tuple[int, int], float]) -> None:
        for key, val in adj_dict.items():
            # do not save J_ij if J_ji is present
            try:
                if self._adj_dict[(key[1], key[0])] == val:
                    continue
            except:
                pass
            self._adj_dict.update({key: val})

    def _create_adj_matrix(self) -> None:
        self._adj_matrix = np.zeros((self.spins, self.spins))
        for (i, j), coupling in self._adj_dict.items():
            self._adj_matrix[i, j] = coupling

    def energy(self, sample: np.ndarray) -> float:
        eng = np.einsum("ij,i,j", self.adj_matrix, sample, sample)
        eng += np.einsum("i,i", self.h_field, sample)
        return eng

    def savetxt(self) -> None:
        assert bool(
            self._adj_dict
        ), "The connectivity dictionary is empty, instantiate first."
        txtarr = []
        for (i, j), coupling in self._adj_dict.items():
            # see http://mcsparse.uni-bonn.de/spinglass/ for the format style
            # see loadtxt as well
            txtarr.append([i + 1, j + 1, coupling])
        np.savetxt(f"couplings-{self.spins}spins", txtarr)
