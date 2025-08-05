import numpy as np
from spars import CSR, cholesky, solve_cholesky


def test_cholesky_lsqr_against_numpy():
    # Overdetermined system: 3 equations, 2 unknowns
    dense_A = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
    ])
    b = np.array([6.0, 0.0, 0.0])

    A = CSR.from_dense(dense_A)

    ATA = A.T @ A
    ATb = A.T @ b
    L = cholesky(ATA.tocsc())
    L.sort_indices()  # sort indices to ensure diagonal is last
    actual = solve_cholesky(L, ATb)

    # Use NumPy to compute expected solution
    ATA = dense_A.T @ dense_A
    ATb = dense_A.T @ b
    L = np.linalg.cholesky(ATA)
    y = np.linalg.solve(L, ATb)
    expected = np.linalg.solve(L.T, y)

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12)
