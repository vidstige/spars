#pragma once

typedef struct {
    int nrows;
    int ncols;
    int nnz;
    int *colptr;
    int *rowind;
    double *values;
} csc_t;

csc_t *csc_create(int nrows, int ncols, int nnz,
                   int *colptr,
                   int *rowind,
                   double *values);

void csc_destroy(csc_t *csc);

void csc_sort_indices(csc_t *A);