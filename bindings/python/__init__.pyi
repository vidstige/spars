from typing import Tuple

import numpy as np

class CSR:
    shape: Tuple[int, int]

    def __getitem__(self, key: Tuple[int, int]) -> float: ...
    def todense(self) -> np.ndarray: ...

def cholesky(A: CSR) -> CSR: ...
