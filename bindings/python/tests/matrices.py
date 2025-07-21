from sparsely import LIL

def easy2x2() -> LIL:
    A = LIL((2, 2))
    A[0, 0] = 4
    A[0, 1] = 2
    A[1, 0] = 2
    A[1, 1] = 3
    return A
