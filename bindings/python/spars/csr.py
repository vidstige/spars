from typing import List, Union
import numpy as np

from ._sparse_c import CSR as _RawCSR, csr_mul_dense, csr_mul_csr, csr_add_csr, csr_to_csc


class CSR:
    def __init__(self, *args, **kwargs):
        self._c_obj = _RawCSR(*args, **kwargs)

    @classmethod
    def from_dense(cls, array_like: Union[np.ndarray, List[List[float]]]) -> 'CSR':
        from .lil import LIL
        lil = LIL.from_dense(array_like)
        return lil.tocsr()

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
        from .csc import CSC
        return CSC.from_c_obj(self._c_obj.T)

    def todense(self) -> np.ndarray:
        return self._c_obj.todense()

    def dot(self, rhs):
        if isinstance(rhs, CSR):
            return CSR.from_c_obj(csr_mul_csr(self._c_obj, rhs._c_obj))
        if isinstance(rhs, np.ndarray):
            return csr_mul_dense(self._c_obj, rhs)
        raise TypeError(f"Unsupported rhs type for dot: {type(rhs)}")

    def __matmul__(self, rhs):
        return self.dot(rhs)

    def add(self, other: 'CSR') -> 'CSR':
         return CSR.from_c_obj(csr_add_csr(self._c_obj, other._c_obj))

    def __add__(self, other: 'CSR') -> 'CSR':
        return self.add(other)

    def __getitem__(self, key):
        return self._c_obj[key]
    
    def tocsc(self):
        from .csc import CSC
        return CSC.from_c_obj(csr_to_csc(self._c_obj))
