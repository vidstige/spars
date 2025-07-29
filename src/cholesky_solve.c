#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h> // todo: don't exit here, return error code instead
#include "sparsely/mul.h"
#include "sparsely/cholesky_solve.h"

void csr_solve_lower(const csr_t *L, const dense_t *b, dense_t *x) {
    int n = L->nrows;

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        double diag = 0.0;

        for (int idx = L->rowptr[i]; idx < L->rowptr[i + 1]; ++idx) {
            int j = L->colind[idx];
            double val = L->values[idx];

            if (j < i) {
                sum += val * x->values[j];
            } else if (j == i) {
                diag = val;
            }
        }

        assert(diag != 0.0 && "Zero diagonal in lower-triangular matrix");
        x->values[i] = (b->values[i] - sum) / diag;
    }
}

void csr_solve_upper(const csr_t *U, const dense_t *b, dense_t *x) {
    assert(U->nrows == U->ncols);
    assert(U->nrows == b->n);
    assert(U->nrows == x->n);

    int n = U->nrows;
    for (int i = n - 1; i >= 0; --i) {
        double sum = b->values[i];
        double diag = 0.0;

        // First pass: compute the sum of upper triangle
        for (int idx = U->rowptr[i]; idx < U->rowptr[i + 1]; ++idx) {
            int j = U->colind[idx];
            double val = U->values[idx];

            if (j > i) {
                sum -= val * x->values[j];
            } else if (j == i) {
                diag = val;
            } else {
                // Sanity check: shouldn't happen in upper matrix
                printf("  Unexpected lower entry U[%d, %d] = %.6f (ignored)\n", i, j, val);
            }
        }

        assert(diag != 0.0 && "Zero diagonal in upper-triangular matrix");
        x->values[i] = sum / diag;
    }
}

void csc_solve_cholesky(const csc_t *L, const dense_t *b, dense_t *x) {
    int n = L->ncols;
    dense_t y = dense_empty(n);

    csr_t *L_csr = csc_to_csr(L);
    csr_solve_lower(L_csr, b, &y);

    csr_t *LT = csc_transpose_to_csr(L);
    csr_sort_indices(LT);
    csr_solve_upper(LT, &y, x);

    csr_destroy(LT);
    csr_destroy(L_csr);
    dense_destroy(&y);
}
