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

    int *col_counts = (int *)calloc(n, sizeof(int));
    if (!col_counts) return NULL;

    // Count entries per column
    for (int idx = 0; idx < nnz; idx++)
        col_counts[A->colind[idx]]++;

    int *colptr = (int *)malloc((n + 1) * sizeof(int));
    if (!colptr) { free(col_counts); return NULL; }

    colptr[0] = 0;
    for (int i = 0; i < n; i++)
        colptr[i + 1] = colptr[i] + col_counts[i];

    int *rowind = (int *)malloc(nnz * sizeof(int));
    double *values = (double *)malloc(nnz * sizeof(double));
    if (!rowind || !values) {
        free(col_counts);
        free(colptr);
        free(rowind);
        free(values);
        return NULL;
    }

    // Temp positions
    memcpy(col_counts, colptr, n * sizeof(int));

    for (int i = 0; i < m; i++) {
        for (int idx = A->rowptr[i]; idx < A->rowptr[i + 1]; idx++) {
            int j = A->colind[idx];
            int dest = col_counts[j]++;
            rowind[dest] = i;
            values[dest] = A->values[idx];
        }
    }

    free(col_counts);
    
    return csc_create(m, n, nnz, colptr, rowind, values);
}

csr_t *csr_mul_csr(const csr_t *A, const csr_t *B) {
    assert(A->ncols == B->nrows);

    int m = A->nrows;
    int n = B->ncols;

    csc_t *B_T = csr_transpose_to_csc(B);
    if (!B_T) return NULL;

    int *rowptr = (int *)malloc((m + 1) * sizeof(int));
    int alloc_nnz = 4 * A->nnz;
    int *colind = (int *)malloc(alloc_nnz * sizeof(int));
    double *values = (double *)malloc(alloc_nnz * sizeof(double));

    if (!rowptr || !colind || !values) {
        free(rowptr);
        free(colind);
        free(values);
        csc_destroy(B_T);
        return NULL;
    }

    int nnz = 0;
    rowptr[0] = 0;

    for (int i = 0; i < m; i++) {
        // Markers for the i-th row of A
        int *marker = (int *)calloc(n, sizeof(int));
        if (!marker) {
            free(rowptr);
            free(colind);
            free(values);
            csc_destroy(B_T);
            return NULL;
        }

        int row_nnz = 0;

        for (int j = 0; j < n; j++) {
            double sum = 0.0;

            // Dot product of row i in A and column j in B
            int a_start = A->rowptr[i];
            int a_end = A->rowptr[i + 1];

            int b_start = B_T->colptr[j];
            int b_end = B_T->colptr[j + 1];

            int a_idx = a_start;
            int b_idx = b_start;

            while (a_idx < a_end && b_idx < b_end) {
                int a_col = A->colind[a_idx];
                int b_row = B_T->rowind[b_idx];
                if (a_col == b_row) {
                    sum += A->values[a_idx] * B_T->values[b_idx];
                    a_idx++;
                    b_idx++;
                } else if (a_col < b_row) {
                    a_idx++;
                } else {
                    b_idx++;
                }
            }

            if (sum != 0.0) {
                if (nnz >= alloc_nnz) {
                    alloc_nnz *= 2;
                    colind = (int *)realloc(colind, alloc_nnz * sizeof(int));
                    values = (double *)realloc(values, alloc_nnz * sizeof(double));
                    if (!colind || !values) {
                        free(rowptr);
                        free(colind);
                        free(values);
                        csc_destroy(B_T);
                        free(marker);
                        return NULL;
                    }
                }
                colind[nnz] = j;
                values[nnz] = sum;
                nnz++;
                row_nnz++;
            }
        }
        rowptr[i + 1] = nnz;
        free(marker);
    }

    csc_destroy(B_T);

    return csr_create(m, n, nnz, rowptr, colind, values);
}

csr_t *csc_transpose_to_csr(const csc_t *A) {
    int m = A->nrows;
    int n = A->ncols;
    int nnz = A->nnz;

    int *row_counts = (int *)calloc(m, sizeof(int));
    if (!row_counts) return NULL;

    // Count entries per row
    for (int idx = 0; idx < nnz; idx++)
        row_counts[A->rowind[idx]]++;

    int *rowptr = (int *)malloc((m + 1) * sizeof(int));
    if (!rowptr) { free(row_counts); return NULL; }

    rowptr[0] = 0;
    for (int i = 0; i < m; i++)
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
    memcpy(row_counts, rowptr, m * sizeof(int));

    for (int j = 0; j < n; j++) {
        for (int idx = A->colptr[j]; idx < A->colptr[j + 1]; idx++) {
            int i = A->rowind[idx];
            int dest = row_counts[i]++;
            colind[dest] = j;
            values[dest] = A->values[idx];
        }
    }

    free(row_counts);
    return csr_create(m, n, nnz, rowptr, colind, values);
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
