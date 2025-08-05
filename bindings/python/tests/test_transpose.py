import numpy as np
from spars import CSR, CSC, LIL
from .matrices import easy3x3


def test_csr_transpose():
    A_dense = easy3x3()
    A = CSR.from_dense(A_dense)
    np.testing.assert_allclose(A.T.todense(), A_dense.T)


def test_csc_transpose():
    A_dense = easy3x3()
    A = CSC.from_dense(A_dense)
    np.testing.assert_allclose(A.T.todense(), A_dense.T)


def test_lil_transpose():
    A_dense = easy3x3()
    A = LIL.from_dense(A_dense)
    np.testing.assert_allclose(A.T.todense(), A_dense.T)


def test_csr_transpose_type():
    A = CSR.from_dense(easy3x3())
    A_T = A.T
    assert isinstance(A_T, CSC), "Transpose of CSR should be CSC"


def test_csc_transpose_type():
    A = CSC.from_dense(easy3x3())
    A_T = A.T
    assert isinstance(A_T, CSR), "Transpose of CSC should be CSR"
