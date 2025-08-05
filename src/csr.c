#define _GNU_SOURCE // For qsort_r
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "spars/alloc.h"
#include "spars/sort.h"
#include "spars/csr.h"

csr_t *csr_create(
    int nrows, int ncols, int nnz,
    const int *rowptr, const int *colind,
    const double *values
) {
    if (!rowptr || !colind || !values) return NULL;

    csr_t *csr = malloc(sizeof(csr_t));
    if (!csr) return NULL;

    csr->nrows = nrows;
    csr->ncols = ncols;
    csr->nnz = nnz;

    // Allocate and copy rowptr and colind using plain malloc (alignment doesn't help here)
    csr->rowptr = malloc(sizeof(int) * (nrows + 1));
    csr->colind = malloc(sizeof(int) * nnz);
    csr->values = spars_alloc(32, sizeof(double) * nnz); // aligned allocation

    if (!csr->rowptr || !csr->colind || !csr->values) {
        csr_destroy(csr);
        return NULL;
    }

    memcpy(csr->rowptr, rowptr, sizeof(int) * (nrows + 1));
    memcpy(csr->colind, colind, sizeof(int) * nnz);
    memcpy(csr->values, values, sizeof(double) * nnz);

    return csr;
}

void csr_destroy(csr_t *csr) {
    if (!csr) return;

    spars_free(csr->values);
    free(csr->colind);
    free(csr->rowptr);
    free(csr);
}

// Compare A->colind[start + a] vs A->colind[start + b]
static SPARS_COMPARE_FUNCTION(compare_col_indices, a, b, thunk) {
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
