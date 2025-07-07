import numpy as np
from sparsely import CSR, cholesky, solve_cholesky


def xtest_cholesky_lsqr():
    # Example matrix A and vector b
    A = CSR.from_dense([[2.0, 1.0], [1.0, 2.0]])
    b = np.array([8.0, 8.0])

    # Compute the Cholesky factorization of A^T @ A
    ATA = A.T @ A
    ATb = A.T @ b
    factor = cholesky(ATA)

    # Solve for x using the factorization
    x = solve_cholesky(factor, ATb)

    # Expected solution
    expected = np.array([1.0, 2.0])

    np.testing.assert_allclose(x, expected, rtol=1e-6, atol=1e-12)