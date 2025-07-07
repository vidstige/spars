from typing import Tuple

import numpy as np

from . import _sparse_c


class LIL:
    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape
        nrows, _ = shape
        self.rows = [[] for _ in range(nrows)]

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
        return _sparse_c.CSR(
            nrows=nrows,
            ncols=ncols,
            rowptr=rowptr,
            colind=colind,
            values=values
        )
