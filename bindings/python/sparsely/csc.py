import numpy as np
from ._sparse_c import CSC as _RawCSC


class CSC:
    def __init__(self, *args, **kwargs):
        self._c_obj = _RawCSC(*args, **kwargs)

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

    def __getitem__(self, key):
        return self._c_obj[key]
