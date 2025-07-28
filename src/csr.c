#include <stdlib.h>
#include <string.h>
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
