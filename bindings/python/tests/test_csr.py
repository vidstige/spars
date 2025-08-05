import numpy as np
import pytest
from spars import CSR


def test_csr_raw():
    # Matrix:
    # [4 2]
    # [2 3]
    #
    # CSR representation:
    # rowptr: [0, 2, 4]
    # colind: [0, 1, 0, 1]
    # values: [4, 2, 2, 3]

    rowptr = np.array([0, 2, 4], dtype=np.int32)
    colind = np.array([0, 1, 0, 1], dtype=np.int32)
    values = np.array([4.0, 2.0, 2.0, 3.0], dtype=np.float64)

    A = CSR(nrows=2, ncols=2, rowptr=rowptr, colind=colind, values=values)

    assert A.shape == (2, 2)

    # Check individual elements using __getitem__
    assert A[0, 0] == pytest.approx(4.0)
    assert A[0, 1] == pytest.approx(2.0)
    assert A[1, 0] == pytest.approx(2.0)
    assert A[1, 1] == pytest.approx(3.0)


def test_csr():
    rowptr = np.array([0, 2, 4], dtype=np.int32)
    colind = np.array([0, 1, 0, 1], dtype=np.int32)
    values = np.array([4.0, 2.0, 2.0, 3.0], dtype=np.float64)

    A = CSR(nrows=2, ncols=2, rowptr=rowptr, colind=colind, values=values)
    dense = A.todense()

    expected = np.array([
        [4.0, 2.0],
        [2.0, 3.0]
    ])

    np.testing.assert_allclose(dense, expected, rtol=1e-6, atol=1e-12)
