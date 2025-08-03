#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "sparsely/alloc.h"
#include "sparsely/csc.h"

typedef struct {
    int i;
    double val;
} entry_t;

static inline int find_or_insert(entry_t *entries, int *count, int i) {
    for (int k = 0; k < *count; ++k) {
        if (entries[k].i == i)
            return k;
    }
    entries[*count].i = i;
    entries[*count].val = 0.0;
    return (*count)++;
}

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

    entry_t *entries = malloc(n * sizeof(entry_t));
    if (!entries) return NULL;

    int nz = 0;
    for (int j = 0; j < n; ++j) {
        int count = 0;

        // Fill entries with column j of A, i >= j
        for (int idx = a_colptr[j]; idx < a_colptr[j + 1]; ++idx) {
            int i = a_rowind[idx];
            if (i >= j) {
                entries[count].i = i;
                entries[count].val = a_values[idx];
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

            if (Ljk == 0.0)
                continue;

            // Apply update: work[i] -= Ljk * Lik for i >= j
            for (int idx = colptr[k]; idx < colptr[k + 1]; ++idx) {
                int i = rowind[idx];
                if (i >= j) {
                    double Lik = values[idx];
                    int k_idx = find_or_insert(entries, &count, i);
                    entries[k_idx].val -= Ljk * Lik;
                }
            }
        }

        // Compute diagonal
        double diag = 0.0;
        int found_diag = 0;
        for (int k = 0; k < count; ++k) {
            if (entries[k].i == j) {
                diag = entries[k].val;
                found_diag = 1;
                break;
            }
        }

        if (!found_diag || diag <= 0.0) {
            fprintf(stderr, "Matrix not positive definite at column %d\n", j);
            free(colptr); free(rowind); free(values); free(entries);
            return NULL;
        }

        double Ljj = sqrt(diag);
        colptr[j] = nz;

        for (int k = 0; k < count; ++k) {
            int i = entries[k].i;
            double val = (i == j) ? Ljj : entries[k].val / Ljj;
            rowind[nz] = i;
            values[nz++] = val;
        }

        colptr[j + 1] = nz;
    }

    free(entries);

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
