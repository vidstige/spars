from typing import List, Tuple, Union

import numpy as np

from .csr import CSR
from ._sparse_c import CSR as _RawCSR


class LIL:
    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape
        nrows, _ = shape
        self.rows: List[List[Tuple[int, float]]] = [[] for _ in range(nrows)]

    @classmethod
    def from_dense(cls, array_like: Union[np.ndarray, List[List[float]]]) -> 'LIL':
        dense = np.asarray(array_like, dtype=np.float64)
        if dense.ndim != 2:
            raise ValueError("from_dense expects a 2D array.")

        nrows, ncols = dense.shape
        lil = cls((nrows, ncols))

        for i in range(nrows):
            for j in range(ncols):
                val = dense[i, j]
                if val != 0.0:
                    lil[i, j] = val

        return lil

    def __setitem__(self, key, value):
        """Allow A[i, j] = val"""
        nrows, ncols = self.shape
        i, j = key
        if not (0 <= i < nrows) or not (0 <= j < ncols):
            raise IndexError("Index out of bounds")

        # Replace if exists
        row = self.rows[i]
        for idx, (col, _) in enumerate(row):
            if col == j:
                row[idx] = (j, value)
                return
        row.append((j, value))

    def __getitem__(self, key):
        i, j = key
        row = self.rows[i]
        for col, val in row:
            if col == j:
                return val
        return 0.0

    def __repr__(self) -> str:
        nrows, ncols = self.shape
        return f"LIL(nrows={nrows}, ncols={ncols}, rows={self.rows})"
    
    def transpose(self):
        nrows, ncols = self.shape
        new_rows = [[] for _ in range(ncols)]

        for row_idx, row in enumerate(self.rows):
            for col_idx, val in row:
                new_rows[col_idx].append((row_idx, val))

        result = LIL((ncols, nrows))
        result.rows = new_rows
        return result

    @property
    def T(self):
        return self.transpose()

    def todense(self) -> np.ndarray:
        dense = np.zeros(self.shape, dtype=np.float64)
        for i, row in enumerate(self.rows):
            for j, val in row:
                dense[i, j] = val
        return dense

    def tocsc(self):
        return self.T.tocsr().T

    def tocsr(self):
        rowptr = [0]
        colind = []
        values = []

        for row in self.rows:
            row = sorted(row, key=lambda x: x[0])  # optional: ensure sorted
            for col, val in row:
                colind.append(col)
                values.append(val)
            rowptr.append(len(colind))

        rowptr = np.asarray(rowptr, dtype=np.int32)
        colind = np.asarray(colind, dtype=np.int32)
        values = np.asarray(values, dtype=np.float64)

        assert rowptr.shape[0] == self.shape[0] + 1
        assert colind.shape[0] == values.shape[0]

        nrows, ncols = self.shape
        return CSR.from_c_obj(_RawCSR(
            nrows=nrows,
            ncols=ncols,
            rowptr=rowptr,
            colind=colind,
            values=values
        ))
