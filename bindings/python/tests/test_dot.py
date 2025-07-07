import numpy as np

from sparsely import lil_array

def test_csr_dot_dense():
    A = lil_array((2, 2))
    A[0, 0] = 4
    A[0, 1] = 2
    A[1, 0] = 2
    A[1, 1] = 3
    A = A.tocsr()
    x = np.array([1.0, 2.0])
    b = A.dot(x)
    expected = np.array([8.0, 8.0])
    np.testing.assert_allclose(b, expected, rtol=1e-6)
