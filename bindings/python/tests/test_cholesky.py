import numpy as np

from spars import cholesky, solve_cholesky

from .matrices import easy2x2


def test_factorization():
    A = easy2x2().tocsc()
    L = cholesky(A)
    L.sort_indices()

    expected = np.array([
        [2.0, 0.0],
        [1.0, np.sqrt(2.0)]
    ])
    np.testing.assert_allclose(L.todense(), expected, rtol=1e-6, atol=1e-12)


def test_solve_2x2():
    A_lil = easy2x2()
    A = A_lil.tocsc()
    
    # Expected solution
    expected = np.array([1.0, 2.0])
    b = A_lil.todense() @ expected

    L = cholesky(A)  # find factorization
    L.sort_indices()  # sort indices to ensure diagonal is last
    x = solve_cholesky(L, b)  # solve

    np.testing.assert_allclose(x, expected, rtol=1e-6, atol=1e-12)
