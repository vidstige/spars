#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "spars/alloc.h"
#include "spars/csc.h"

static inline csc_t *cholesky(
    int nrows, int ncols, int nnz,
    const int *restrict a_colptr,
    const int *restrict a_rowind,
    const double *restrict a_pointer_values)
{
    const double *a_values = SPARS_ASSUME_ALIGNED(a_pointer_values);

    int n = ncols;
    if (nrows != n) {
        fprintf(stderr, "Matrix is not square.\n");
        return NULL;
    }

    int *restrict colptr = calloc(n + 1, sizeof(int));
    int *restrict rowind = malloc(nnz * sizeof(int));  // overallocate
    double *restrict values = SPARS_ASSUME_ALIGNED(
        spars_alloc(SPARS_ALIGNMENT, nnz * sizeof(double))
    );
    if (!colptr || !rowind || !values) return NULL;

    int *restrict work_rows = malloc(n * sizeof(int));
    double *restrict work_values =  SPARS_ASSUME_ALIGNED(
        spars_alloc(SPARS_ALIGNMENT, n * sizeof(double))
    );
    int *restrict imap = malloc(n * sizeof(int));  // -1 means unused
    if (!work_rows || !work_values || !imap) return NULL;
    for (int i = 0; i < n; ++i) imap[i] = -1;

    // Keep track of how far we've worked in each column
    int *restrict ptr = malloc(n * sizeof(int));
    if (!ptr) return NULL;
    for (int k = 0; k < n; ++k)
        ptr[k] = colptr[k];

    int diag_idx = -1; // keep track of diagonal index
    int nz = 0;
    for (int j = 0; j < n; ++j) {
        int count = 0;

        // Fill work arrays with column j of A, where i >= j
        for (int idx = a_colptr[j]; idx < a_colptr[j + 1]; ++idx) {
            int i = a_rowind[idx];
            if (i >= j) {
                imap[i] = count;
                work_rows[count] = i;
                work_values[count] = a_values[idx];
                if (i == j) diag_idx = count; // save diagonal index
                count++;
            }
        }

        // Subtract previous column contributions
        for (int k = 0; k < j; ++k) {
            // First find j in column k
            double Ljk = 0.0;
            int idx = ptr[k];
            const int end = colptr[k + 1];
            while (idx < end && rowind[idx] < j)
                ++idx;

            ptr[k] = idx;
            if (idx < end && rowind[idx] == j)
                Ljk = values[idx];

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
                        if (i == j) diag_idx = count; // update diagonal index in case diagonal entry was inserted
                        entry_idx = count;
                        count++;
                    }
                    work_values[entry_idx] -= Ljk * Lik;
                }
            }
        }

        // validate diagonal entry 
        if (diag_idx == -1 || work_values[diag_idx] <= 0.0) {
        //if (!found_diag || diag <= 0.0) {
            fprintf(stderr, "Matrix not positive definite at column %d\n", j);
            free(colptr); free(rowind); free(values);
            free(work_rows); free(imap);
            spars_free(work_values);
            return NULL;
        }
        
        double diag = work_values[diag_idx];
        double Ljj = sqrt(diag);
        colptr[j] = nz;

        for (int k = 0; k < count; ++k) {
            int i = work_rows[k];
            double val = (i == j) ? Ljj : work_values[k] / Ljj;
            rowind[nz] = i;
            values[nz++] = val;
        }

        colptr[j + 1] = nz;

        // clear i lookup
        for (int k = 0; k < count; ++k)
            imap[work_rows[k]] = -1;
    }

    free(work_rows);
    spars_free(work_values);
    free(imap);
    free(ptr);

    csc_t *L = malloc(sizeof(csc_t));
    L->nrows = n;
    L->ncols = n;
    L->nnz = nz;
    L->colptr = colptr;
    L->rowind = realloc(rowind, nz * sizeof(int));
    L->values = spars_realloc(
        values, nnz * sizeof(double),
        nz * sizeof(double),
        SPARS_ALIGNMENT
    );
    return L;
}

csc_t *cholesky_factor(const csc_t *A) {
    return cholesky(
        A->nrows, A->ncols, A->nnz,
        A->colptr, A->rowind, A->values);
}
