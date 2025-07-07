#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "sparsely/dense.h"
#include "sparsely/mul.h"

dense_t csr_mul_dense(const csr_t *A, const dense_t *x) {
    assert(A->ncols == x->n);

    dense_t y = dense_empty(A->nrows);

    for (int i = 0; i < A->nrows; i++) {
        double sum = 0.0;
        int start = A->rowptr[i];
        int end = A->rowptr[i + 1];
        for (int idx = start; idx < end; idx++) {
            sum += A->values[idx] * x->values[A->colind[idx]];
        }
        y.values[i] = sum;
    }

    return y;
}

csc_t *csr_transpose_to_csc(const csr_t *A) {
    int m = A->nrows;
    int n = A->ncols;
    int nnz = A->nnz;

    // In transpose, rows of A become columns of Aᵗ
    // Count non-zeros per row in A
    int *row_counts = (int *)calloc(m, sizeof(int));
    if (!row_counts) return NULL;

    for (int i = 0; i < m; i++)
        row_counts[i] = A->rowptr[i + 1] - A->rowptr[i];

    // Allocate colptr for CSC of shape (nrows=n, ncols=m)
    // Since transpose shape is (n, m)
    int *colptr = (int *)malloc((m + 1) * sizeof(int));
    if (!colptr) {
        free(row_counts);
        return NULL;
    }

    colptr[0] = 0;
    for (int i = 0; i < m; i++)
        colptr[i + 1] = colptr[i] + row_counts[i];

    // Prepare rowind and values
    int *rowind = (int *)malloc(nnz * sizeof(int));
    double *values = (double *)malloc(nnz * sizeof(double));
    if (!rowind || !values) {
        free(row_counts);
        free(colptr);
        free(rowind);
        free(values);
        return NULL;
    }

    // Reset positions for scatter
    memcpy(row_counts, colptr, m * sizeof(int));

    // Scatter
    for (int i = 0; i < m; i++) {
        for (int idx = A->rowptr[i]; idx < A->rowptr[i + 1]; idx++) {
            int j = A->colind[idx];
            int dest = row_counts[i]++;

            // Transpose: (i, j) -> (j, i)
            rowind[dest] = j;
            values[dest] = A->values[idx];
        }
    }

    free(row_counts);

    // Return CSC of shape (n, m) = transpose of (m, n)
    return csc_create(n, m, nnz, colptr, rowind, values);
}

csr_t *csr_mul_csr(const csr_t *A, const csr_t *B) {
    assert(A->ncols == B->nrows);

    int m = A->nrows;
    int n = B->ncols;

    int *rowptr = (int *)malloc((m + 1) * sizeof(int));
    int alloc_nnz = 4 * (A->nnz + B->nnz);  // initial guess
    int *colind = (int *)malloc(alloc_nnz * sizeof(int));
    double *values = (double *)malloc(alloc_nnz * sizeof(double));
    if (!rowptr || !colind || !values) {
        free(rowptr); free(colind); free(values);
        return NULL;
    }

    // Temporary accumulators
    double *accum = (double *)calloc(n, sizeof(double));
    int *marker = (int *)malloc(n * sizeof(int));
    if (!accum || !marker) {
        free(rowptr); free(colind); free(values);
        free(accum); free(marker);
        return NULL;
    }

    int nnz = 0;
    rowptr[0] = 0;

    for (int i = 0; i < m; i++) {
        int count = 0;

        // Reset marker
        for (int j = 0; j < n; j++) marker[j] = -1;

        // Accumulate A's row i * B
        for (int a_idx = A->rowptr[i]; a_idx < A->rowptr[i + 1]; a_idx++) {
            int k = A->colind[a_idx];
            double Aval = A->values[a_idx];

            // Row k of B
            for (int b_idx = B->rowptr[k]; b_idx < B->rowptr[k + 1]; b_idx++) {
                int j = B->colind[b_idx];
                double Bval = B->values[b_idx];

                if (marker[j] < rowptr[i]) {
                    marker[j] = rowptr[i] + count;
                    accum[j] = Aval * Bval;
                    count++;
                } else {
                    accum[j] += Aval * Bval;
                }
            }
        }

        // Store results
        for (int j = 0; j < n; j++) {
            if (marker[j] >= rowptr[i] && accum[j] != 0.0) {
                if (nnz >= alloc_nnz) {
                    alloc_nnz *= 2;
                    colind = (int *)realloc(colind, alloc_nnz * sizeof(int));
                    values = (double *)realloc(values, alloc_nnz * sizeof(double));
                    if (!colind || !values) {
                        free(rowptr); free(colind); free(values);
                        free(accum); free(marker);
                        return NULL;
                    }
                }
                colind[nnz] = j;
                values[nnz] = accum[j];
                nnz++;
            }
        }
        rowptr[i + 1] = nnz;
    }

    free(accum);
    free(marker);

    return csr_create(m, n, nnz, rowptr, colind, values);
}

csr_t *csc_transpose_to_csr(const csc_t *A) {
    int m = A->nrows;
    int n = A->ncols;
    int nnz = A->nnz;

    // Output shape is (n x m)
    int *row_counts = (int *)calloc(n, sizeof(int));
    if (!row_counts) return NULL;

    // Count entries per row in transpose
    for (int j = 0; j < n; j++) {
        for (int idx = A->colptr[j]; idx < A->colptr[j + 1]; idx++) {
            row_counts[j]++;
        }
    }

    int *rowptr = (int *)malloc((n + 1) * sizeof(int));
    if (!rowptr) { free(row_counts); return NULL; }

    rowptr[0] = 0;
    for (int i = 0; i < n; i++)
        rowptr[i + 1] = rowptr[i] + row_counts[i];

    int *colind = (int *)malloc(nnz * sizeof(int));
    double *values = (double *)malloc(nnz * sizeof(double));
    if (!colind || !values) {
        free(row_counts);
        free(rowptr);
        free(colind);
        free(values);
        return NULL;
    }

    // Temp positions
    memcpy(row_counts, rowptr, n * sizeof(int));

    // Scatter phase
    for (int j = 0; j < n; j++) {
        for (int idx = A->colptr[j]; idx < A->colptr[j + 1]; idx++) {
            int i = A->rowind[idx];
            int dest = row_counts[j]++;
            colind[dest] = i;
            values[dest] = A->values[idx];
        }
    }

    free(row_counts);
    return csr_create(n, m, nnz, rowptr, colind, values);
}

csr_t *csc_mul_csr(const csc_t *A, const csr_t *B) {
    assert(A->ncols == B->nrows);

    int m = A->nrows;
    int n = B->ncols;

    int *rowptr = (int *)calloc(m + 1, sizeof(int));
    int capacity = 16;
    int *colind = (int *)malloc(capacity * sizeof(int));
    double *values = (double *)malloc(capacity * sizeof(double));
    if (!rowptr || !colind || !values) {
        free(rowptr); free(colind); free(values);
        return NULL;
    }

    int nnz = 0;
    for (int i = 0; i < m; i++) {
        double *accum = (double *)calloc(n, sizeof(double));
        char *mask = (char *)calloc(n, sizeof(char));

        // For row i in A (CSC)
        for (int j = 0; j < A->ncols; j++) {
            // A(i, j) = ?
            for (int idx = A->colptr[j]; idx < A->colptr[j + 1]; idx++) {
                if (A->rowind[idx] == i) {
                    double a_ij = A->values[idx];

                    // Multiply this column of A with row of B
                    for (int k = B->rowptr[j]; k < B->rowptr[j + 1]; k++) {
                        int col = B->colind[k];
                        double val = a_ij * B->values[k];

                        accum[col] += val;
                        mask[col] = 1;
                    }
                }
            }
        }

        rowptr[i + 1] = rowptr[i];
        for (int col = 0; col < n; col++) {
            if (mask[col]) {
                if (nnz >= capacity) {
                    capacity *= 2;
                    colind = (int *)realloc(colind, capacity * sizeof(int));
                    values = (double *)realloc(values, capacity * sizeof(double));
                    if (!colind || !values) {
                        free(rowptr); free(colind); free(values);
                        free(accum); free(mask);
                        return NULL;
                    }
                }
                colind[nnz] = col;
                values[nnz] = accum[col];
                nnz++;
                rowptr[i + 1]++;
            }
        }

        free(accum);
        free(mask);
    }

    // Trim arrays
    colind = (int *)realloc(colind, nnz * sizeof(int));
    values = (double *)realloc(values, nnz * sizeof(double));
    return csr_create(m, n, nnz, rowptr, colind, values);
}

dense_t csc_mul_dense(const csc_t *A, const dense_t *x) {
    assert(A->ncols == x->n);

    dense_t y = dense_zeros(A->nrows);

    for (int j = 0; j < A->ncols; j++) {
        double xj = x->values[j];
        for (int idx = A->colptr[j]; idx < A->colptr[j + 1]; idx++) {
            int i = A->rowind[idx];
            y.values[i] += A->values[idx] * xj;
        }
    }

    return y;
}
