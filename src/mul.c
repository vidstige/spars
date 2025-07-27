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

dense_t csr_transposed_mul_dense(const csr_t *A, const dense_t *x) {
    assert(A->nrows == x->n);
    dense_t y = dense_zeros(A->ncols); // Initialize result to zero

    for (int i = 0; i < A->nrows; i++) {
        double xi = x->values[i];
        int start = A->rowptr[i];
        int end = A->rowptr[i + 1];
        for (int idx = start; idx < end; idx++) {
            int j = A->colind[idx];
            y.values[j] += A->values[idx] * xi;
        }
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

// Fast matrix multiplication using Gustavson's algorithm
csr_t *csr_mul_csr(const csr_t *A, const csr_t *B) {
    assert(A->ncols == B->nrows);

    int m = A->nrows;
    int n = B->ncols;

    int *rowptr = (int *)malloc((m + 1) * sizeof(int));
    int alloc_nnz = 4 * (A->nnz + B->nnz);  // initial guess
    int *colind = (int *)malloc(alloc_nnz * sizeof(int));
    double *values = (double *)malloc(alloc_nnz * sizeof(double));
    if (!rowptr || !colind || !values) goto fail1;

    // Sparse accumulator setup
    double *accum = (double *)calloc(n, sizeof(double));
    int *next = (int *)malloc(n * sizeof(int));
    if (!accum || !next) goto fail2;

    // Initialize next array to -1
    for (int j = 0; j < n; j++) next[j] = -1;

    int nnz = 0;
    rowptr[0] = 0;

    for (int i = 0; i < m; i++) {
        int head = -2;

        // Accumulate row i of A times B
        for (int a_idx = A->rowptr[i]; a_idx < A->rowptr[i + 1]; a_idx++) {
            int k = A->colind[a_idx];
            assert(k >= 0 && k < B->nrows);
            double Aval = A->values[a_idx];

            for (int b_idx = B->rowptr[k]; b_idx < B->rowptr[k + 1]; b_idx++) {
                int j = B->colind[b_idx];
                assert(j >= 0 && j < n);
                double Bval = B->values[b_idx];

                if (next[j] == -1) {
                    next[j] = head;
                    head = j;
                    accum[j] = Aval * Bval;
                } else {
                    accum[j] += Aval * Bval;
                }
            }
        }

        rowptr[i + 1] = rowptr[i];
        int j = head;
        while (j != -2) {
            int next_j = next[j];

            if (nnz >= alloc_nnz) {
                alloc_nnz *= 2;
                int *new_colind = realloc(colind, alloc_nnz * sizeof(int));
                double *new_values = realloc(values, alloc_nnz * sizeof(double));
                if (!new_colind || !new_values) {
                    free(new_colind);
                    free(new_values);
                    goto fail2;
                }
                colind = new_colind;
                values = new_values;
            }

            colind[nnz] = j;
            values[nnz] = accum[j];
            nnz++;
            rowptr[i + 1]++;

            accum[j] = 0.0;
            next[j] = -1;
            j = next_j;
        }
    }

    free(accum);
    free(next);

    colind = realloc(colind, nnz * sizeof(int));
    values = realloc(values, nnz * sizeof(double));
    return csr_create(m, n, nnz, rowptr, colind, values);

fail2:
    free(accum);
    free(next);
fail1:
    free(rowptr);
    free(colind);
    free(values);
    return NULL;
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
    // Convert A from CSC to CSR
    csr_t *A_csr = csc_to_csr(A);
    if (!A_csr) return NULL;
    csr_t *C = csr_mul_csr(A_csr, B);
    csr_destroy(A_csr);    
    return C;
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

csr_t* csc_to_csr(const csc_t* csc) {
    int nrows = csc->nrows;
    int ncols = csc->ncols;
    int nnz = csc->nnz;

    // Allocate csr_t struct
    csr_t* csr = (csr_t*)malloc(sizeof(csr_t));
    if (!csr) return NULL;

    csr->nrows = nrows;
    csr->ncols = ncols;
    csr->nnz = nnz;
    csr->rowptr = (int*)calloc(nrows + 1, sizeof(int));
    csr->colind = (int*)malloc(nnz * sizeof(int));
    csr->values = (double*)malloc(nnz * sizeof(double));

    if (!csr->rowptr || !csr->colind || !csr->values) {
        // Cleanup if any allocation fails
        free(csr->rowptr);
        free(csr->colind);
        free(csr->values);
        free(csr);
        return NULL;
    }

    // Step 1: Count number of entries in each row
    for (int col = 0; col < ncols; ++col) {
        for (int idx = csc->colptr[col]; idx < csc->colptr[col + 1]; ++idx) {
            int row = csc->rowind[idx];
            assert(0 <= row && row < nrows);
            csr->rowptr[row + 1]++;
        }
    }

    // Step 2: Cumulative sum for rowptr
    for (int i = 0; i < nrows; ++i) {
        csr->rowptr[i + 1] += csr->rowptr[i];
    }

    // Step 3: Fill colind and values using a row-wise offset
    int* offset = (int*)malloc(nrows * sizeof(int));
    if (!offset) {
        free(csr->rowptr);
        free(csr->colind);
        free(csr->values);
        free(csr);
        return NULL;
    }
    memcpy(offset, csr->rowptr, nrows * sizeof(int));

    for (int col = 0; col < ncols; ++col) {
        for (int idx = csc->colptr[col]; idx < csc->colptr[col + 1]; ++idx) {
            int row = csc->rowind[idx];
            int dest = offset[row]++;
            csr->colind[dest] = col;
            csr->values[dest] = csc->values[idx];
        }
    }

    free(offset);
    return csr;
}