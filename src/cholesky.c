#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "sparsely/alloc.h"
#include "sparsely/csc.h"

csc_t *cholesky_factor(const csc_t *A) {
    int n = A->ncols;
    if (A->nrows != n) {
        fprintf(stderr, "Matrix is not square.\n");
        return NULL;
    }

    int *colptr = calloc(n + 1, sizeof(int));
    int *rowind = malloc(A->nnz * sizeof(int));  // overallocate
    double *values = sparsely_alloc(SPARSELY_ALIGNMENT, A->nnz * sizeof(double));
    if (!colptr || !rowind || !values) return NULL;

    double *work = sparsely_alloc(SPARSELY_ALIGNMENT, n * sizeof(double));
    memset(work, 0, n * sizeof(double));

    int *pattern = malloc(n * sizeof(int));
    if (!work || !pattern) return NULL;

    int nz = 0;

    for (int j = 0; j < n; ++j) {
        int pattern_len = 0;

        // Step 1: copy A[j:n, j] into work
        for (int idx = A->colptr[j]; idx < A->colptr[j + 1]; ++idx) {
            int i = A->rowind[idx];
            if (i >= j) {
                work[i] = A->values[idx];
                pattern[pattern_len++] = i;
            }
        }

        // Step 2: compute L[i,j] for i > j using previously computed columns
        for (int k = 0; k < j; ++k) {
            // find L[j,k] if it exists (it's at colptr[k] to colptr[k+1])
            double Ljk = 0.0;
            for (int idx = colptr[k]; idx < colptr[k + 1]; ++idx) {
                if (rowind[idx] == j) {
                    Ljk = values[idx];
                    break;
                }
            }
            if (Ljk == 0.0) continue;

            // subtract Ljk * L[i,k] for i >= j
            for (int idx = colptr[k]; idx < colptr[k + 1]; ++idx) {
                int i = rowind[idx];
                if (i >= j) {
                    work[i] -= Ljk * values[idx];
                }
            }
        }

        // Step 3: compute and validate diagonal
        double diag = work[j];
        if (diag <= 0.0) {
            fprintf(stderr, "Matrix not positive definite at column %d\n", j);
            free(colptr); free(rowind); free(values); free(work); free(pattern);
            return NULL;
        }

        double Ljj = sqrt(diag);
        colptr[j] = nz;

        // Step 4: store column j
        for (int i = j; i < n; ++i) {
            if (work[i] != 0.0) {
                rowind[nz] = i;
                values[nz++] = (i == j) ? Ljj : work[i] / Ljj;
            }
        }

        colptr[j + 1] = nz;

        // Step 5: clear work
        for (int k = 0; k < pattern_len; ++k)
            work[pattern[k]] = 0.0;
    }

    free(work);
    free(pattern);

    csc_t *L = malloc(sizeof(csc_t));
    L->nrows = n;
    L->ncols = n;
    L->nnz = nz;
    L->colptr = colptr;
    L->rowind = realloc(rowind, nz * sizeof(int));
    L->values = realloc(values, nz * sizeof(double));
    L->values = sparsely_realloc(
        L->values, A->nnz * sizeof(double),
        nz * sizeof(double),
        SPARSELY_ALIGNMENT
    );
    return L;
}
