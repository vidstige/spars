from typing import Tuple
import numpy as np
from sparsely import CSC, CSR, LIL, lil_array, csr_array
from .matrices import easy3x3


def random_lil(shape: Tuple[int, int], density: float, seed: int) -> LIL:
    """Create a random LIL sparse matrix with given shape and density."""
    rng = np.random.default_rng(123)
    a = lil_array(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.rand() < density:
                a[i, j] = rng.standard_normal()
    return a


def random_csr(shape: Tuple[int, int], density: float, seed: int) -> CSR:
    return random_lil(shape, density, seed).tocsr()


def random_csc(shape: Tuple[int, int], density: float, seed: int) -> CSC:
    return random_lil(shape, density, seed).tocsc()


def test_csr_dot_csr():
    seed = 42
    lhs = random_csr(shape=(10, 10), density=0.1, seed=seed)
    rhs = random_csr(shape=(10, 10), density=0.1, seed=seed)

    actual = lhs @ rhs

    # Dense ground truth
    expected = lhs.todense() @ rhs.todense()
    np.testing.assert_allclose(actual.todense(), expected, rtol=1e-6, atol=1e-12)


def test_csc_mul_csr():
    seed = 42
    lhs = random_csc(shape=(10, 10), density=0.1, seed=seed)
    rhs = random_csr(shape=(10, 10), density=0.1, seed=seed)

    actual = lhs @ rhs

    # Dense ground truth
    expected = lhs.todense() @ rhs.todense()
    np.testing.assert_allclose(actual.todense(), expected, rtol=1e-6, atol=1e-12)


# matrix vector multiplication tests
def test_csr_mul_dense():
    A_dense = easy3x3()
    A = CSR.from_dense(A_dense)
    x = np.array([1.0, 2.0, 3.0])
    result = A @ x
    expected = A_dense @ x
    np.testing.assert_allclose(result, expected)


def test_csc_mul_dense():
    A_dense = easy3x3()
    A = CSC.from_dense(A_dense)
    x = np.array([1.0, 2.0, 3.0])
    result = A @ x
    expected = A_dense @ x
    np.testing.assert_allclose(result, expected)
