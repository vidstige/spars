#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "sparsely/csr.h"

csr_t *cholesky_factor(const csr_t *A)
{
    int n = A->nrows;
    if (A->ncols != n) {
        fprintf(stderr, "Matrix is not square.\n");
        return NULL;
    }

    // Allocate result
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

    // Temporary workspace
    double *work = calloc(n, sizeof(double));
    if (!work) {
        free(L->rowptr); free(L->colind); free(L->values); free(L);
        return NULL;
    }

    int nz = 0;
    L->rowptr[0] = 0;

    for (int i = 0; i < n; i++) {
        // 1. Clear work vector
        for (int k = 0; k < n; k++) work[k] = 0.0;

        // 2. Scatter A[i,:] into work (only lower triangle)
        for (int idx = A->rowptr[i]; idx < A->rowptr[i+1]; idx++) {
            int j = A->colind[idx];
            if (j <= i) {
                work[j] = A->values[idx];
            }
        }

        // 3. Compute L[i, j] for j < i
        for (int j = 0; j < i; j++) {
            double sum = work[j];

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

            // Store in work
            work[j] = Lij;

            // Store in CSR
            L->colind[nz] = j;
            L->values[nz] = Lij;
            nz++;
        }

        // 4. Compute diagonal
        double diag = work[i];
        for (int k = 0; k < i; k++) {
            diag -= work[k] * work[k];
        }

        if (diag <= 0.0) {
            fprintf(stderr, "Matrix not positive definite at row %d\n", i);
            free(work);
            free(L->rowptr); free(L->colind); free(L->values); free(L);
            return NULL;
        }

        double Lii = sqrt(diag);

        L->colind[nz] = i;
        L->values[nz] = Lii;
        nz++;

        L->rowptr[i+1] = nz;
    }

    free(work);
    return L;
}
