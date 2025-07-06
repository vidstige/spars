#include <stdlib.h>
#include <string.h>
#include "sparsely/csr.h"

csr_t *csr_create(
    int nrows, int ncols, int nnz,
    int *rowptr, int *colind,
    const double *values
) {
    if (!rowptr || !colind || !values) return NULL;
    if (nrows <= 0 || ncols <= 0 || nnz < 0) return NULL;

    csr_t *mat = (csr_t *)malloc(sizeof(csr_t));
    if (!mat) return NULL;

    mat->nrows = nrows;
    mat->ncols = ncols;
    mat->nnz = nnz;

    // Allocate and copy rowptr
    mat->rowptr = (int *)malloc((nrows + 1) * sizeof(int));
    if (!mat->rowptr) {
        free(mat);
        return NULL;
    }
    memcpy(mat->rowptr, rowptr, (nrows + 1) * sizeof(int));

    // Allocate and copy colind
    mat->colind = (int *)malloc(nnz * sizeof(int));
    if (!mat->colind) {
        free(mat->rowptr);
        free(mat);
        return NULL;
    }
    memcpy(mat->colind, colind, nnz * sizeof(int));

    // Allocate and copy values
    mat->values = (double *)malloc(nnz * sizeof(double));
    if (!mat->values) {
        free(mat->colind);
        free(mat->rowptr);
        free(mat);
        return NULL;
    }
    memcpy(mat->values, values, nnz * sizeof(double));

    return mat;
}

void csr_destroy(csr_t *mat)
{
    if (!mat) return;

    free(mat->values);
    free(mat->colind);
    free(mat->rowptr);
    free(mat);
}