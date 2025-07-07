#include <assert.h>
#include "sparsely/dense.h"
#include "sparsely/mul.h"

dense_t csr_mul_dense(const csr_t *A, const dense_t *x) {
    assert(A->ncols == x->n);

    dense_t y = dense_empty(A->nrows);

    for (int i = 0; i < A->nrows; i++) {
        double sum = 0.0;
        int start = A->rowptr[i];
        int end = A->rowptr[i + 1];
        for (int idx = start; idx < end; idx++) {
            sum += A->values[idx] * x->values[A->colind[idx]];
        }
        y.values[i] = sum;
    }

    return y;
}
