from typing import Callable, Tuple

import numpy as np
import pytest
from spars import CSC, CSR

from .matrices import easy3x3, easy3x4, random_csc, random_csr


@pytest.mark.parametrize("shape,density", [
    ((3, 3), 1.0),
    ((10, 10), 0.1),
])
def test_csr_dot_csr(shape: Tuple[int, int], density: float):
    seed = 42
    lhs = random_csr(shape=shape, density=density, seed=seed)
    rhs = random_csr(shape=shape, density=density, seed=seed)

    actual = lhs @ rhs

    # Dense ground truth
    expected = lhs.todense() @ rhs.todense()
    np.testing.assert_allclose(actual.todense(), expected, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("shape,density", [
    ((3, 3), 1.0),
    ((10, 10), 0.1),
])
def test_csc_mul_csr(shape: Tuple[int, int], density: float):
    seed = 42
    lhs = random_csc(shape=shape, density=density, seed=seed)
    rhs = random_csr(shape=shape, density=density, seed=seed)

    actual = lhs @ rhs

    # Dense ground truth
    expected = lhs.todense() @ rhs.todense()
    np.testing.assert_allclose(actual.todense(), expected, rtol=1e-6, atol=1e-12)


# matrix vector multiplication tests
@pytest.mark.parametrize("create_matrix,x", [
    (easy3x3, np.array([1.0, 2.0, 3.0])),
    (easy3x4, np.array([1.0, 2.0, 3.0, 4.0])),
])
def test_csr_mul_dense(create_matrix: Callable[[], np.ndarray], x: np.ndarray):
    A_dense = create_matrix()
    A = CSR.from_dense(A_dense)
    result = A @ x
    expected = A_dense @ x
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("create_matrix,x", [
    (easy3x3, np.array([1.0, 2.0, 3.0])),
    (easy3x4, np.array([1.0, 2.0, 3.0, 4.0])),
])
def test_csc_mul_dense(create_matrix: Callable[[], np.ndarray], x: np.ndarray):
    A_dense = create_matrix()
    A = CSC.from_dense(A_dense)
    result = A @ x
    expected = A_dense @ x
    np.testing.assert_allclose(result, expected)
