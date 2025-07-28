#include <stddef.h>
#include <stdio.h>
#include <stdlib.h> // todo: don't exit here, return error code instead
#include "sparsely/cholesky_solve.h"

void csc_solve_lower(const csc_t *L, const dense_t *b, dense_t *x) {
    int n = L->ncols;
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        for (int idx = L->colptr[i]; idx < L->colptr[i + 1]; ++idx) {
            int row = L->rowind[idx];
            if (row == i) {
                x->values[i] = (b->values[i] - sum) / L->values[idx];
                break;
            } else if (row < i) {
                sum += L->values[idx] * x->values[row];
            }
        }
    }
}
void csc_solve_upper(const csc_t *L, const dense_t *b, dense_t *x) {
    int n = L->ncols;
    for (int i = n - 1; i >= 0; --i) {
        double sum = 0.0;
        for (int idx = L->colptr[i]; idx < L->colptr[i + 1]; ++idx) {
            int row = L->rowind[idx];
            if (row == i) {
                x->values[i] = (b->values[i] - sum) / L->values[idx];
                break;
            } else if (row > i) {
                sum += L->values[idx] * x->values[row];
            }
        }
    }
}

void csc_solve_cholesky(const csc_t *L, const dense_t *b, dense_t *x) {
    int n = L->ncols;
    dense_t y = dense_empty(n);
    csc_solve_lower(L, b, &y);
    csc_solve_upper(L, &y, x);
    dense_destroy(&y);
}
