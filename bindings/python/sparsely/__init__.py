from ._sparse_c import CSR as CSR
from .lil import LIL as LIL

csr_array = CSR
lil_array = LIL

__all__ = ["csr_array", "lil_array", "CSR", "LIL"]
