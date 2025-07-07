#include <assert.h>
#include <stdio.h>
#include "sparsely/mul.h"

dense_t *csr_dot_dense(const csr_t *A, const dense_t *x) {
    assert(A->ncols == x->n);

    // TODO: Use dense_empty to allocate y
    dense_t *y = dense_create(A->nrows, NULL);
    assert(y != NULL);

    for (int i = 0; i < A->nrows; i++) {
        double sum = 0.0;
        int start = A->rowptr[i];
        int end = A->rowptr[i + 1];
        for (int idx = start; idx < end; idx++) {
            sum += A->values[idx] * x->values[A->colind[idx]];
        }
        y->values[i] = sum;
    }

    return y;
}
