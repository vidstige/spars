from sparsely import lil_array, cholesky


def test_solve():
    A_lil = lil_array((2, 2))
    A_lil[0, 0] = 4
    A_lil[0, 1] = 2
    A_lil[1, 0] = 2
    A_lil[1, 1] = 3
    
    A = A_lil.tocsr()
    factor = cholesky(A)

    print(factor)
