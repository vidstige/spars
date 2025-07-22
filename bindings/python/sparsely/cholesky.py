import numpy as np

from ._sparse_c import (
    cholesky as _c_cholesky,
    solve_cholesky as _c_solve_cholesky,
)
from .csr import CSR


def cholesky(A: CSR) -> CSR:
    raw_result = _c_cholesky(A._c_obj)
    return CSR.from_c_obj(raw_result)


def solve_cholesky(L: CSR, b: np.ndarray) -> np.ndarray:
    return _c_solve_cholesky(L._c_obj, b)
