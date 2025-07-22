#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "sparsely/csr.h"

csr_t *cholesky_factor(const csr_t *A) {
    int n = A->nrows;
    if (A->ncols != n) {
        fprintf(stderr, "Matrix is not square.\n");
        return NULL;
    }

    csr_t *L = malloc(sizeof(csr_t));
    if (!L) return NULL;

    L->nrows = n;
    L->ncols = n;
    L->rowptr = calloc(n + 1, sizeof(int));
    L->colind = malloc(A->nnz * sizeof(int));    // Overallocate for simplicity
    L->values = malloc(A->nnz * sizeof(double));
    if (!L->rowptr || !L->colind || !L->values) {
        free(L->rowptr); free(L->colind); free(L->values); free(L);
        return NULL;
    }

    int *work_cols = malloc(n * sizeof(int));       // Max needed
    double *work_values = malloc(n * sizeof(double)); // Same here
    if (!work_cols || !work_values) {
        free(work_cols); free(work_values);
        free(L->rowptr); free(L->colind); free(L->values); free(L);
        return NULL;
    }

    int nz = 0;
    L->rowptr[0] = 0;

    for (int i = 0; i < n; i++) {
        int work_len = 0;

        // 1. Scatter A[i,:] into sparse work arrays (only lower triangle)
        for (int idx = A->rowptr[i]; idx < A->rowptr[i+1]; idx++) {
            int j = A->colind[idx];
            if (j <= i) {
                work_cols[work_len] = j;
                work_values[work_len] = A->values[idx];
                work_len++;
            }
        }

        // 2. Compute L[i, j] for j < i
        for (int j = 0; j < i; j++) {
            // Find work[j]
            double sum = 0.0;
            for (int k = 0; k < work_len; k++) {
                if (work_cols[k] == j) {
                    sum = work_values[k];
                    break;
                }
            }

            // Sparse dot-product: sum -= L[i,k] * L[j,k] for k < j
            int p_i = L->rowptr[i];
            int p_j = L->rowptr[j];
            while (p_i < nz && L->colind[p_i] < j && p_j < L->rowptr[j+1]) {
                int col_i = L->colind[p_i];
                int col_j = L->colind[p_j];

                if (col_i == col_j) {
                    sum -= L->values[p_i] * L->values[p_j];
                    p_i++;
                    p_j++;
                } else if (col_i < col_j) {
                    p_i++;
                } else {
                    p_j++;
                }
            }

            // Divide by L[j,j]
            int diag_idx = L->rowptr[j+1] - 1;
            double Lj_diag = L->values[diag_idx];
            double Lij = sum / Lj_diag;

            // Update work[j]
            for (int k = 0; k < work_len; k++) {
                if (work_cols[k] == j) {
                    work_values[k] = Lij;
                    goto store_lij;
                }
            }
            // If j wasn't already in work, append it
            work_cols[work_len] = j;
            work_values[work_len] = Lij;
            work_len++;

        store_lij:
            L->colind[nz] = j;
            L->values[nz] = Lij;
            nz++;
        }

        // 3. Compute diagonal
        double diag = 0.0;
        int found_diag = 0;

        for (int k = 0; k < work_len; k++) {
            if (work_cols[k] == i) {
                diag = work_values[k];
                found_diag = 1;
                break;
            }
        }

        if (!found_diag) {
            fprintf(stderr, "Missing diagonal entry at row %d\n", i);
            free(work_cols); free(work_values);
            free(L->rowptr); free(L->colind); free(L->values); free(L);
            return NULL;
        }

        for (int k = 0; k < work_len; k++) {
            int col = work_cols[k];
            double val = work_values[k];
            if (col < i) {
                diag -= val * val;
            }
        }

        if (diag <= 0.0) {
            fprintf(stderr, "Matrix not positive definite at row %d\n", i);
            free(work_cols); free(work_values);
            free(L->rowptr); free(L->colind); free(L->values); free(L);
            return NULL;
        }

        double Lii = sqrt(diag);
        L->colind[nz] = i;
        L->values[nz] = Lii;
        nz++;

        L->rowptr[i+1] = nz;
    }

    free(work_cols);
    free(work_values);
    return L;
}
