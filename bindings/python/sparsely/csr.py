from typing import Union
import numpy as np
from . import _sparse_c
from ._sparse_c import CSR as _RawCSR, csr_mul_dense, csr_mul_csr


class CSR:
    def __init__(self, *args, **kwargs):
        self._c_obj = _RawCSR(*args, **kwargs)

    @classmethod
    def from_c_obj(cls, c_obj):
        instance = cls.__new__(cls)
        instance._c_obj = c_obj
        return instance

    @property
    def shape(self):
        return self._c_obj.shape

    def todense(self) -> np.ndarray:
        return self._c_obj.todense()

    def dot(self, rhs):
        if isinstance(rhs, CSR):
            return CSR.from_c_obj(_sparse_c.csr_mul_csr(self._c_obj, rhs._c_obj))
        if isinstance(rhs, np.ndarray):
            return _sparse_c.csr_mul_dense(self._c_obj, rhs)
        raise TypeError(f"Unsupported rhs type for dot: {type(rhs)}")

    def __matmul__(self, rhs):
        return self.dot(rhs)

    def __getitem__(self, key):
        return self._c_obj[key]

    def sort_indices(self):
        return self._c_obj.sort_indices()
