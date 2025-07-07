from sparsely import CSR, lil_array

def easy2x2() -> CSR:
    A = lil_array((2, 2))
    A[0, 0] = 4
    A[0, 1] = 2
    A[1, 0] = 2
    A[1, 1] = 3
    return A.tocsr()
