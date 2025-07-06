import numpy as np
from sparsely import lil_array, cholesky


def test_solve():
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