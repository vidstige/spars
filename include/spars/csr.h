#pragma once
#include <stdbool.h>

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
    const int *rowptr, const int *colind,
    const double *values
);
void csr_destroy(csr_t *csr);
bool csr_ok(csr_t *csr);

void csr_sort_indices(csr_t *A);
