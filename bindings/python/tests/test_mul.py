import numpy as np
from sparsely import lil_array


def test_csr_mul_csr_sparse():
    np.random.seed(42)

    size = 10
    density = 0.2

    # Create random sparse matrix A
    A_lil = lil_array((size, size))
    for i in range(size):
        for j in range(size):
            if np.random.rand() < density:
                A_lil[i, j] = np.random.randn()

    A_csr = A_lil.tocsr()

    # A @ A
    C = A_csr @ A_csr

    # Dense ground truth
    A_dense = A_csr.todense()
    C_expected = A_dense @ A_dense

    np.testing.assert_allclose(C.todense(), C_expected, rtol=1e-6, atol=1e-12)
