#include <stddef.h>
#include <stdio.h>
#include <stdlib.h> // todo: don't exit here, return error code instead
#include "sparsely/cholesky_solve.h"

void csr_solve_lower(const csr_t *L, const dense_t *b, dense_t *x)
{
    int n = L->nrows;

    for (int i = 0; i < n; i++) {
        int start = L->rowptr[i];
        int end = L->rowptr[i + 1];

        if (end <= start) {
            fprintf(stderr, "Empty row %d in lower solve.\n", i);
            exit(1);
        }

        int diag_idx = end - 1;
        double diag = L->values[diag_idx];
        if (L->colind[diag_idx] != i) {
            fprintf(stderr, "Diagonal not last in row %d.\n", i);
            exit(1);
        }

        double sum = b->values[i];

        for (int idx = start; idx < diag_idx; idx++) {
            sum -= L->values[idx] * x->values[L->colind[idx]];
        }

        x->values[i] = sum / diag;
    }
}

void csr_solve_upper(const csr_t *L, const dense_t *b, dense_t *x) {
    int n = L->nrows;

    for (int i = n - 1; i >= 0; i--) {
        double sum = b->values[i];

        for (int j = i + 1; j < n; j++) {
            int start_j = L->rowptr[j];
            int end_j = L->rowptr[j + 1];

            for (int idx = start_j; idx < end_j; idx++) {
                if (L->colind[idx] == i) {
                    sum -= L->values[idx] * x->values[j];
                    break;
                }
                if (L->colind[idx] > i) break;
            }
        }

        int diag_idx = L->rowptr[i + 1] - 1;
        double diag = L->values[diag_idx];
        x->values[i] = sum / diag;
    }
}

void csr_solve_cholesky(const csr_t *L, const dense_t *b, dense_t *x) {
    int n = L->nrows;
    dense_t y = dense_zeros(n);

    csr_solve_lower(L, b, &y);
    csr_solve_upper(L, &y, x);

    dense_destroy(&y);
}