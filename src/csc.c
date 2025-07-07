#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include "sparsely/csc.h"

csc_t *csc_create(
    int nrows, int ncols, int nnz,
    const int *colptr,
    const int *rowind,
    const double *values
) {
    // Basic sanity checks
    assert(nrows >= 0);
    assert(ncols >= 0);
    assert(nnz >= 0);

    assert(colptr != NULL);
    assert(rowind != NULL);
    assert(values != NULL);

    // colptr must have length ncols + 1
    for (int i = 0; i < ncols; ++i) {
        assert(colptr[i] <= colptr[i + 1]);
    }

    // Final entry in colptr must match nnz
    assert(colptr[ncols] == nnz);

    csc_t *csc = (csc_t *)malloc(sizeof(csc_t));
    if (!csc) {
        fprintf(stderr, "csc_create: malloc failed for struct\n");
        return NULL;
    }

    csc->nrows = nrows;
    csc->ncols = ncols;
    csc->nnz = nnz;

    csc->colptr = (int *)malloc((ncols + 1) * sizeof(int));
    csc->rowind = (int *)malloc(nnz * sizeof(int));
    csc->values = (double *)malloc(nnz * sizeof(double));

    if (!csc->colptr || !csc->rowind || !csc->values) {
        fprintf(stderr, "csc_create: malloc failed for arrays\n");
        free(csc->colptr);
        free(csc->rowind);
        free(csc->values);
        free(csc);
        return NULL;
    }

    memcpy(csc->colptr, colptr, (ncols + 1) * sizeof(int));
    memcpy(csc->rowind, rowind, nnz * sizeof(int));
    memcpy(csc->values, values, nnz * sizeof(double));

    return csc;
}

void csc_destroy(csc_t *csc) {
    if (!csc) return;

    free(csc->colptr);
    free(csc->rowind);
    free(csc->values);
    free(csc);
}
