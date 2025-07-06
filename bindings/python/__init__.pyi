from typing import Tuple


class CSR:
    shape: Tuple[int, int]

    def __getitem__(self, key: Tuple[int, int]) -> float: ...


def cholesky(A: CSR) -> CSR: ...
