from typing import Tuple
import numpy as np
from spars import CSC, CSR, LIL


def easy2x2() -> LIL:
    A = LIL((2, 2))
    A[0, 0] = 4
    A[0, 1] = 2
    A[1, 0] = 2
    A[1, 1] = 3
    return A


def easy3x3() -> np.ndarray:
    return np.array([
        [1, 0, 2],
        [0, 3, 0],
        [4, 0, 5],
    ])


def easy3x4() -> np.ndarray:
    return np.array([
        [1, 0, 2, -1],
        [0, 3, 0, 4],
        [4, 0, 5, 0],
    ])


def random_lil(shape: Tuple[int, int], density: float, seed: int) -> LIL:
    """Create a random LIL sparse matrix with given shape and density."""
    rng = np.random.default_rng(seed)
    a = LIL(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.rand() < density:
                a[i, j] = rng.standard_normal()
    return a


def random_csr(shape: Tuple[int, int], density: float, seed: int) -> CSR:
    return random_lil(shape, density, seed).tocsr()


def random_csc(shape: Tuple[int, int], density: float, seed: int) -> CSC:
    return random_lil(shape, density, seed).tocsc()
