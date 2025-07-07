from typing import Tuple
import numpy as np
from sparsely import CSR


def matrix() -> Tuple[np.ndarray, CSR]:
    A_dense = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 5.0]
    ])
    return A_dense, CSR.from_dense(A_dense)


def test_csr_transpose_roundtrip():
    A_dense, A = matrix()
    A_T = A.T
    A_back = A_T.T

    np.testing.assert_allclose(A_back.todense(), A_dense)


def test_dense():
    A_dense, A = matrix()
    AT = A.T
    np.testing.assert_allclose(AT.todense(), A_dense.T)
