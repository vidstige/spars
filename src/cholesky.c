#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "sparsely/alloc.h"
#include "sparsely/csc.h"

static inline csc_t *cholesky(
    int nrows, int ncols, int nnz,
    const int *restrict a_colptr,
    const int *restrict a_rowind,
    const double *restrict a_pointer_values)
{
    const double *a_values = SPARSELY_ASSUME_ALIGNED(a_pointer_values);

    int n = ncols;
    if (nrows != n) {
        fprintf(stderr, "Matrix is not square.\n");
        return NULL;
    }

    int *colptr = calloc(n + 1, sizeof(int));
    int *rowind = malloc(nnz * sizeof(int));  // overallocate
    double *values = SPARSELY_ASSUME_ALIGNED(
        sparsely_alloc(SPARSELY_ALIGNMENT, nnz * sizeof(double))
    );
    if (!colptr || !rowind || !values) return NULL;

    double *work = SPARSELY_ASSUME_ALIGNED(
        sparsely_alloc(SPARSELY_ALIGNMENT, n * sizeof(double))
    );
    int *marker = malloc(n * sizeof(int));
    if (!work || !marker) return NULL;

    memset(marker, 0, n * sizeof(int));
    int gen = 1;

    int nz = 0;
    for (int j = 0; j < n; ++j) {
        gen++;

        // Fill work[] with column j of A
        for (int idx = a_colptr[j]; idx < a_colptr[j + 1]; ++idx) {
            int i = a_rowind[idx];
            if (i >= j) {
                if (marker[i] != gen) {
                    work[i] = a_values[idx];
                    marker[i] = gen;
                }
            }
        }

        // Subtract L * L^T contributions
        for (int k = 0; k < j; ++k) {
            double Ljk = 0.0;
            int idx0 = colptr[k];
            int idx1 = colptr[k + 1];

            // Find Ljk and do update in a single pass
            for (int idx = idx0; idx < idx1; ++idx) {
                int i = rowind[idx];
                double Lik = values[idx];

                if (i == j) {
                    Ljk = Lik;
                    break; // Since i is sorted, i == j will appear before i > j
                } else if (i > j) {
                    break; // No match will be found beyond this
                }
            }

            if (Ljk == 0.0)
                continue;

            // Now apply the update
            for (int idx = idx0; idx < idx1; ++idx) {
                int i = rowind[idx];
                if (i >= j) {
                    if (marker[i] != gen) {
                        work[i] = 0.0;
                        marker[i] = gen;
                    }
                    work[i] -= Ljk * values[idx];
                }
            }
        }

        // Compute diagonal
        double diag = work[j];
        if (diag <= 0.0) {
            fprintf(stderr, "Matrix not positive definite at column %d\n", j);
            free(colptr); free(rowind); free(values); free(work); free(marker);
            return NULL;
        }

        double Ljj = sqrt(diag);
        colptr[j] = nz;

        for (int i = j; i < n; ++i) {
            if (marker[i] == gen) {
                rowind[nz] = i;
                values[nz++] = (i == j) ? Ljj : work[i] / Ljj;
            }
        }

        colptr[j + 1] = nz;
    }

    free(work);
    free(marker);

    csc_t *L = malloc(sizeof(csc_t));
    L->nrows = n;
    L->ncols = n;
    L->nnz = nz;
    L->colptr = colptr;
    L->rowind = realloc(rowind, nz * sizeof(int));
    L->values = sparsely_realloc(
        values, nnz * sizeof(double),
        nz * sizeof(double),
        SPARSELY_ALIGNMENT
    );
    return L;
}

csc_t *cholesky_factor(const csc_t *A) {
    return cholesky(
        A->nrows, A->ncols, A->nnz,
        A->colptr, A->rowind, A->values);
}
