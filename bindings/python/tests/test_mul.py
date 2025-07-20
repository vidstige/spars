from typing import Tuple
import numpy as np
import pytest
from sparsely import CSR, LIL, lil_array, csr_array


def random_lil(shape: Tuple[int, int], density: float, seed: int) -> LIL:
    """Create a random LIL sparse matrix with given shape and density."""
    rng = np.random.default_rng(123)
    a = lil_array(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if np.random.rand() < density:
                a[i, j] = rng.standard_normal()
    return a


def as_csr(lil: LIL) -> CSR:
    return lil.tocsr()


def as_csc(lil: LIL) -> CSR:
    return lil.tocsc()


def as_lil(lil: LIL) -> LIL:
    return lil


@pytest.mark.parametrize("left_type,left_shape,left_seed", [
    (as_csr, (10, 10), 42),
])
@pytest.mark.parametrize("right_type,right_shape,right_seed", [
    (as_csr, (10, 10), 42),
])
def test_sparse_matrix_multiplication(
    left_type, left_shape, left_seed,
    right_type, right_shape, right_seed,
):
    lhs = left_type(random_lil(left_shape, density=0.1, seed=left_seed))
    rhs = right_type(random_lil(right_shape, density=0.1, seed=right_seed))

    actual = lhs @ rhs

    # Dense ground truth
    C_expected = lhs.todense() @ rhs.todense()

    np.testing.assert_allclose(actual.todense(), C_expected, rtol=1e-6, atol=1e-12)


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