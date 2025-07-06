#pragma once

typedef struct {
    int nrows;
    int ncols;
    int nnz;
    int *rowptr;    // nrows + 1
    int *colind;    // nnz
    double *values; // nnz
} csr_t;

csr_t *csr_create(
    int nrows, int ncols, int nnz,
    int *rowptr, int *colind,
    const double *values
);

void csr_destroy(csr_t *mat);

