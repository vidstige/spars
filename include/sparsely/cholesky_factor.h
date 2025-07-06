#pragma once

#include "csr.h"

typedef struct {
    int n;
    int *rowptr;
    int *colind;
    double *values;
} cholesky_factor_t;

cholesky_factor_t *cholesky_factor_create(const csr_t *A);

void cholesky_factor_destroy(cholesky_factor_t *factor);

void cholesky_factor_solve(const cholesky_factor_t *factor, const double *b, double *x);