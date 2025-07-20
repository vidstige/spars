from typing import Tuple
import numpy as np
from sparsely import LIL, lil_array, csr_array


def random_lil(shape: Tuple[int, int], density: float, seed: int) -> LIL:
    """Create a random LIL sparse matrix with given shape and density."""
    rng = np.random.default_rng(123)
    a = lil_array(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.rand() < density:
                a[i, j] = rng.standard_normal()
    return a


def test_csr_mul_csr_sparse():
    A_lil = random_lil((10, 10), density=0.2, seed=42)
    A_csr = A_lil.tocsr()

    # A @ A
    C = A_csr @ A_csr

    # Dense ground truth
    A_dense = A_csr.todense()
    C_expected = A_dense @ A_dense

    np.testing.assert_allclose(C.todense(), C_expected, rtol=1e-6, atol=1e-12)


def test_csc_mul_dense():
    A_dense = np.array([
        [1, 0, 2],
        [0, 3, 0],
        [4, 0, 5]
    ])

    A = csr_array.from_dense(A_dense)
    AT = A.T
    x = np.array([1.0, 2.0, 3.0])

    result = AT @ x

    expected = A_dense.T @ x

    np.testing.assert_allclose(result, expected)