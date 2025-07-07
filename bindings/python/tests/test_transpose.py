import numpy as np
from sparsely import csr_array


def test_csr_transpose_roundtrip():
    dense = np.array([
        [1.0, 0.0, 2.0],
        [0.0, 3.0, 0.0],
        [4.0, 0.0, 5.0]
    ])

    A = csr_array.from_dense(dense)
    A_T = A.T
    A_back = A_T.T

    np.testing.assert_allclose(A_back.todense(), dense)
