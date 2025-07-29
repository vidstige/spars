#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "sparsely/sort.h"
#include "sparsely/csr.h"

csr_t *csr_create(
    int nrows, int ncols, int nnz,
    int *rowptr, int *colind,
    double *values
) {
    if (!rowptr || !colind || !values) return NULL;

    csr_t *csr = malloc(sizeof(csr_t));
    if (!csr) return NULL;

    csr->nrows = nrows;
    csr->ncols = ncols;
    csr->nnz = nnz;
    csr->rowptr = rowptr;
    csr->colind = colind;
    csr->values = values;

    return csr;
}

void csr_destroy(csr_t *mat) {
    if (!mat) return;

    free(mat->values);
    free(mat->colind);
    free(mat->rowptr);
    free(mat);
}

// Compare A->colind[start + a] vs A->colind[start + b]
static SPARSELY_COMPARE_FUNCTION(compare_col_indices, a, b, thunk) {
    int *base_colind = (int *)thunk;
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return base_colind[ia] - base_colind[ib];
}

void csr_sort_indices(csr_t *A) {
    for (int i = 0; i < A->nrows; ++i) {
        int start = A->rowptr[i];
        int end = A->rowptr[i + 1];
        int len = end - start;
        if (len <= 1) continue;

        int *perm = malloc(len * sizeof(int));
        for (int k = 0; k < len; ++k)
            perm[k] = k;

        QSORT_R(perm, len, sizeof(int), compare_col_indices, A->colind + start);

        int *tmp_colind = malloc(len * sizeof(int));
        double *tmp_values = malloc(len * sizeof(double));
        for (int k = 0; k < len; ++k) {
            tmp_colind[k] = A->colind[start + perm[k]];
            tmp_values[k] = A->values[start + perm[k]];
        }

        memcpy(A->colind + start, tmp_colind, len * sizeof(int));
        memcpy(A->values + start, tmp_values, len * sizeof(double));
        
        // diaginal should be at the end of the row
        assert(A->colind[A->rowptr[i + 1] - 1] == i);

        free(perm);
        free(tmp_colind);
        free(tmp_values);
    }
}
