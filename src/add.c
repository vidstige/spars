#include "sparsely/add.h"

#include <stdlib.h>
#include <string.h>
#include <assert.h>

// assumes that the indices in A and B are sorted
csr_t *csr_add(const csr_t *A, const csr_t *B) {
    assert(A->nrows == B->nrows);
    assert(A->ncols == B->ncols);

    int nrows = A->nrows;
    int capacity = A->nnz + B->nnz;

    int *rowptr = malloc((nrows + 1) * sizeof(int));
    int *colind = malloc(capacity * sizeof(int));
    double *values = malloc(capacity * sizeof(double));
    
    int nnz = 0;
    rowptr[0] = 0;

    for (int i = 0; i < nrows; ++i) {
        int ap = A->rowptr[i], aq = A->rowptr[i + 1];
        int bp = B->rowptr[i], bq = B->rowptr[i + 1];

        while (ap < aq || bp < bq) {
            int acol = (ap < aq) ? A->colind[ap] : A->ncols;
            int bcol = (bp < bq) ? B->colind[bp] : B->ncols;

            if (acol == bcol) {
                double sum = A->values[ap++] + B->values[bp++];
                if (sum != 0.0) {
                    colind[nnz] = acol;
                    values[nnz++] = sum;
                }
            } else if (acol < bcol) {
                colind[nnz] = acol;
                values[nnz++] = A->values[ap++];
            } else {
                colind[nnz] = bcol;
                values[nnz++] = B->values[bp++];
            }
        }

        rowptr[i + 1] = nnz;
    }

    colind = realloc(colind, nnz * sizeof(int));
    values = realloc(values, nnz * sizeof(double));

    csr_t *C = malloc(sizeof(csr_t));
    C->nrows = A->nrows;
    C->ncols = A->ncols;
    C->nnz = nnz;
    C->rowptr = rowptr;
    C->colind = colind;
    C->values = values;
    return C;
}
