#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "sparsely/csc.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "sparsely/csr.h"
#include "sparsely/csc.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "sparsely/csc.h"

csc_t *cholesky_factor(const csc_t *A) {
    int n = A->ncols;
    if (A->nrows != n) {
        fprintf(stderr, "Matrix is not square.\n");
        return NULL;
    }

    int *colptr = calloc(n + 1, sizeof(int));
    int *rowind = malloc(A->nnz * sizeof(int));    // Overallocate
    double *values = malloc(A->nnz * sizeof(double));
    if (!colptr || !rowind || !values) return NULL;

    double *work = calloc(n, sizeof(double));
    int *pattern = malloc(n * sizeof(int));
    if (!work || !pattern) return NULL;

    int nz = 0;

    for (int j = 0; j < n; ++j) {
        int pattern_len = 0;

        // Copy A[0..j, j] into work
        for (int idx = A->colptr[j]; idx < A->colptr[j + 1]; ++idx) {
            int i = A->rowind[idx];
            if (i <= j) {
                work[i] = A->values[idx];
                pattern[pattern_len++] = i;
            }
        }

        // For all previous columns i < j
        for (int i = 0; i < j; ++i) {
            // Look for L[i,j] = work[i]
            int diag_idx = -1;
            for (int k = colptr[i]; k < colptr[i + 1]; ++k) {
                if (rowind[k] == i) {
                    diag_idx = k;
                    break;
                }
            }
            if (diag_idx == -1) continue; // skip column i if diagonal not found

            double Lii = values[diag_idx];
            double Lij = work[i] / Lii;

            // Subtract outer product: work[k] -= Lij * L[i,k]
            for (int k = colptr[i]; k < colptr[i + 1]; ++k) {
                int row = rowind[k];
                if (row >= i && row != i) {
                    work[row] -= Lij * values[k];
                }
            }

            work[i] = Lij;
        }

        // Compute and validate diagonal
        double diag = work[j];
        for (int k = 0; k < j; ++k) {
            double Lij = work[k];
            diag -= Lij * Lij;
        }

        if (diag <= 0.0) {
            fprintf(stderr, "Matrix not positive definite at column %d\n", j);
            free(colptr); free(rowind); free(values); free(work); free(pattern);
            return NULL;
        }

        colptr[j] = nz;
        for (int k = 0; k < j; ++k) {
            if (work[k] != 0.0) {
                rowind[nz] = k;
                values[nz++] = work[k];
            }
        }

        rowind[nz] = j;
        values[nz++] = sqrt(diag);
        colptr[j + 1] = nz;

        // Clear work
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
    return L;
}
