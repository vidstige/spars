import numpy as np

from ._sparse_c import (
    cholesky as _c_cholesky,
    solve_cholesky as _c_solve_cholesky,
)
from .csc import CSC


def cholesky(A: CSC) -> CSC:
    raw_result = _c_cholesky(A._c_obj)
    return CSC.from_c_obj(raw_result)


def solve_cholesky(L: CSC, b: np.ndarray) -> np.ndarray:
    return _c_solve_cholesky(L._c_obj, b)
