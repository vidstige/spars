import numpy as np
from sparsely import lil_array, cholesky, solve_cholesky


def test_factorization():
    A_lil = lil_array((2, 2))
    A_lil[0, 0] = 4
    A_lil[0, 1] = 2
    A_lil[1, 0] = 2
    A_lil[1, 1] = 3
    
    A = A_lil.tocsr()
    L = cholesky(A)

    expected = np.array([
        [2.0, 0.0],
        [1.0, np.sqrt(2.0)]
    ])
    assert np.allclose(L.todense(), expected, rtol=1e-6, atol=1e-12)


def test_solve_2x2():
    # Build A in LIL form
    A_lil = lil_array((2, 2))
    A_lil[0, 0] = 4
    A_lil[0, 1] = 2
    A_lil[1, 0] = 2
    A_lil[1, 1] = 3

    A = A_lil.tocsr()  # Convert to CSR
    b = np.array([8.0, 8.0])
    L = cholesky(A)  # find factorization
    L.sort_indices()  # sort indices to ensure diagonal is last
    x = solve_cholesky(L, b)  # solve

    # Expected solution
    expected = np.array([1.0, 2.0])

    np.testing.assert_allclose(x, expected, rtol=1e-6, atol=1e-12)
