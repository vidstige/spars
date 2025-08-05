import numpy as np
import pytest

from spars import CSC

from .matrices import easy2x2


def test_csc_raw():
    # Simple 2x2 example
    #
    # Matrix:
    # [4 2]
    # [2 3]
    #
    # CSC representation:
    # colptr: [0, 2, 4]
    # rowind: [0, 1, 0, 1]
    # values: [4, 2, 2, 3]

    colptr = np.array([0, 2, 4], dtype=np.int32)
    rowind = np.array([0, 1, 0, 1], dtype=np.int32)
    values = np.array([4.0, 2.0, 2.0, 3.0], dtype=np.float64)

    A = CSC(nrows=2, ncols=2, colptr=colptr, rowind=rowind, values=values)

    assert A.shape == (2, 2)

    # Check individual elements using __getitem__
    assert A[0, 0] == pytest.approx(4.0)
    assert A[1, 0] == pytest.approx(2.0)
    assert A[0, 1] == pytest.approx(2.0)
    assert A[1, 1] == pytest.approx(3.0)


def test_csc():
    colptr = np.array([0, 2, 4], dtype=np.int32)
    rowind = np.array([0, 1, 0, 1], dtype=np.int32)
    values = np.array([4.0, 2.0, 2.0, 3.0], dtype=np.float64)

    A = CSC(nrows=2, ncols=2, colptr=colptr, rowind=rowind, values=values)
    dense = A.todense()

    expected = np.array([
        [4.0, 2.0],
        [2.0, 3.0]
    ])

    np.testing.assert_allclose(dense, expected, rtol=1e-6, atol=1e-12)


def test_tocsr():
    csc = easy2x2().tocsc()
    csr = csc.tocsr()
    np.testing.assert_allclose(csr.todense(), csc.todense())