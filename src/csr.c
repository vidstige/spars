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

void csr_destroy(csr_t *mat) {
    if (!mat) return;

    free(mat->values);
    free(mat->colind);
    free(mat->rowptr);
    free(mat);
}

static void sort_row(int *cols, double *vals, int length)
{
    for (int i = 1; i < length; i++) {
        int ckey = cols[i];
        double vkey = vals[i];
        int j = i - 1;

        while (j >= 0 && cols[j] > ckey) {
            cols[j + 1] = cols[j];
            vals[j + 1] = vals[j];
            j--;
        }

        cols[j + 1] = ckey;
        vals[j + 1] = vkey;
    }
}

void csr_sort_indices(csr_t *A) {
    for (int i = 0; i < A->nrows; i++) {
        int start = A->rowptr[i];
        int end = A->rowptr[i + 1];
        int length = end - start;

        if (length > 1) {
            sort_row(&A->colind[start], &A->values[start], length);
        }

        // Ensure diagonal last if present
        int diag_pos = -1;
        for (int k = start; k < end; k++) {
            if (A->colind[k] == i) {
                diag_pos = k;
                break;
            }
        }
        if (diag_pos >= 0 && diag_pos != end - 1) {
            // Swap diagonal to last
            int tmp_col = A->colind[diag_pos];
            A->colind[diag_pos] = A->colind[end - 1];
            A->colind[end - 1] = tmp_col;

            double tmp_val = A->values[diag_pos];
            A->values[diag_pos] = A->values[end - 1];
            A->values[end - 1] = tmp_val;
        }
    }
}