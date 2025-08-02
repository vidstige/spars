#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparsely/alloc.h"
#include "sparsely/csc.h"
#include "sparsely/sort.h"

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

    // Allocate and copy rowptr and colind using plain malloc (alignment doesn't help here)
    csc->colptr = malloc(sizeof(int) * (ncols + 1));
    csc->rowind = malloc(sizeof(int) * nnz);
    csc->values = sparsely_alloc(32, sizeof(double) * nnz); // aligned allocation
    
    if (!csc->colptr || !csc->rowind || !csc->values) {
        csc_destroy(csc);
        return NULL;
    }

    memcpy(csc->colptr, colptr, sizeof(int) * (ncols + 1));
    memcpy(csc->rowind, rowind, sizeof(int) * nnz);
    memcpy(csc->values, values, sizeof(double) * nnz);

    return csc;
}

void csc_destroy(csc_t *csc) {
    if (!csc) return;

    free(csc->colptr);
    free(csc->rowind);
    sparsely_free(csc->values);
    free(csc);
}

static SPARSELY_COMPARE_FUNCTION(compare_row_indices, a, b, thunk) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    const int *rows = (const int *)thunk;
    return (rows[ia] - rows[ib]);
}

void csc_sort_indices(csc_t *A) {
    for (int col = 0; col < A->ncols; ++col) {
        int start = A->colptr[col];
        int end = A->colptr[col + 1];
        int len = end - start;

        // Skip empty or singleton columns
        if (len <= 1) continue;

        // Create permutation array
        int *perm = malloc(len * sizeof(int));
        for (int i = 0; i < len; ++i)
            perm[i] = i;

        // Sort by rowind[perm[i]]
        QSORT_R(perm, len, sizeof(int), compare_row_indices, A->rowind + start);

        // Apply sorted permutation to rowind and values
        int *tmp_rows = malloc(len * sizeof(int));
        double *tmp_vals = malloc(len * sizeof(double));
        for (int i = 0; i < len; ++i) {
            tmp_rows[i] = A->rowind[start + perm[i]];
            tmp_vals[i] = A->values[start + perm[i]];
        }

        memcpy(A->rowind + start, tmp_rows, len * sizeof(int));
        memcpy(A->values + start, tmp_vals, len * sizeof(double));

        free(tmp_rows);
        free(tmp_vals);
        free(perm);
    }
}
