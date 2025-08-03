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

    int *work_rows = malloc(n * sizeof(int));
    double *work_values =  SPARSELY_ASSUME_ALIGNED(
        sparsely_alloc(SPARSELY_ALIGNMENT, n * sizeof(double))
    );
    int *imap = malloc(n * sizeof(int));  // -1 means unused
    if (!work_rows || !work_values || !imap) return NULL;

    int nz = 0;
    for (int j = 0; j < n; ++j) {
        int count = 0;
        for (int i = 0; i < n; ++i) imap[i] = -1;

        // Fill work arrays with column j of A, where i >= j
        for (int idx = a_colptr[j]; idx < a_colptr[j + 1]; ++idx) {
            int i = a_rowind[idx];
            if (i >= j) {
                imap[i] = count;
                work_rows[count] = i;
                work_values[count] = a_values[idx];
                count++;
            }
        }

        // Subtract previous column contributions
        for (int k = 0; k < j; ++k) {
            double Ljk = 0.0;
            for (int idx = colptr[k]; idx < colptr[k + 1]; ++idx) {
                if (rowind[idx] == j) {
                    Ljk = values[idx];
                    break;
                } else if (rowind[idx] > j) {
                    break;
                }
            }

            if (Ljk == 0.0) continue;

            for (int idx = colptr[k]; idx < colptr[k + 1]; ++idx) {
                int i = rowind[idx];
                if (i >= j) {
                    double Lik = values[idx];

                    int entry_idx = imap[i];
                    if (entry_idx == -1) {
                        imap[i] = count;
                        work_rows[count] = i;
                        work_values[count] = 0.0;
                        entry_idx = count;
                        count++;
                    }
                    work_values[entry_idx] -= Ljk * Lik;
                }
            }
        }

        // Find and validate diagonal
        double diag = 0.0;
        int found_diag = 0;
        for (int k = 0; k < count; ++k) {
            if (work_rows[k] == j) {
                diag = work_values[k];
                found_diag = 1;
                break;
            }
        }

        if (!found_diag || diag <= 0.0) {
            fprintf(stderr, "Matrix not positive definite at column %d\n", j);
            free(colptr); free(rowind); free(values);
            free(work_rows); free(imap);
            sparsely_free(work_values);
            return NULL;
        }

        double Ljj = sqrt(diag);
        colptr[j] = nz;

        for (int k = 0; k < count; ++k) {
            int i = work_rows[k];
            double val = (i == j) ? Ljj : work_values[k] / Ljj;
            rowind[nz] = i;
            values[nz++] = val;
        }

        colptr[j + 1] = nz;
    }

    free(work_rows);
    sparsely_free(work_values);
    free(imap);

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
