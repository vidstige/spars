import numpy as np

from . import _sparse_c
from .csr import CSR


def cholesky(A: CSR) -> CSR:
    raw_result = _sparse_c.cholesky(A._c_obj)
    return CSR.from_c_obj(raw_result)


def solve_cholesky(L: CSR, b: np.ndarray) -> np.ndarray:
    return _sparse_c.solve_cholesky(L._c_obj, b)
