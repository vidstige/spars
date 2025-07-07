#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "sparsely/csc.h"

csc_t *csc_create(int nrows, int ncols, int nnz,
                   int *colptr,
                   int *rowind,
                   double *values)
{
    assert(colptr && rowind && values);

    csc_t *csc = malloc(sizeof(csc_t));
    if (!csc) return NULL;

    csc->nrows = nrows;
    csc->ncols = ncols;
    csc->nnz = nnz;
    csc->colptr = colptr;
    csc->rowind = rowind;
    csc->values = values;

    return csc;
}

void csc_destroy(csc_t *csc) {
    if (!csc) return;

    free(csc->colptr);
    free(csc->rowind);
    free(csc->values);
    free(csc);
}
