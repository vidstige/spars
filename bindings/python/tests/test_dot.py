import numpy as np

from .matrices import easy2x2


def test_csr_dot_dense():
    A = easy2x2().tocsr()
    x = np.array([1.0, 2.0])
    b = A.dot(x)
    expected = np.array([8.0, 8.0])
    np.testing.assert_allclose(b, expected, rtol=1e-6)
