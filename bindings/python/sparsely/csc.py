from typing import List, Union
import numpy as np
from ._sparse_c import CSC as _RawCSC, csc_mul_csr, csc_mul_dense, csc_to_csr


class CSC:
    def __init__(self, *args, **kwargs):
        self._c_obj = _RawCSC(*args, **kwargs)
   
    @classmethod
    def from_dense(cls, array_like: np.ndarray) -> 'CSC':
        return CSC.from_c_obj(_RawCSC.fromdense(array_like))

    @classmethod
    def from_c_obj(cls, c_obj):
        instance = cls.__new__(cls)
        instance._c_obj = c_obj
        return instance

    @property
    def shape(self):
        return self._c_obj.shape

    @property
    def T(self):
        from .csr import CSR
        return CSR.from_c_obj(self._c_obj.T)

    def todense(self) -> np.ndarray:
        return self._c_obj.todense()

    def dot(self, rhs):
        from .csr import CSR
        import numpy as np

        if isinstance(rhs, CSR):
            return CSR.from_c_obj(csc_mul_csr(self._c_obj, rhs._c_obj))
        if isinstance(rhs, np.ndarray):
            return csc_mul_dense(self._c_obj, rhs)
        raise TypeError(f"Unsupported rhs type for dot: {type(rhs)}")

    def __matmul__(self, rhs):
        return self.dot(rhs)

    def __getitem__(self, key):
        return self._c_obj[key]

    def sort_indices(self):
        return self._c_obj.sort_indices()

    def tocsr(self):
        from .csr import CSR
        return CSR.from_c_obj(csc_to_csr(self._c_obj))
