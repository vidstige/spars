#pragma once
#include "sparsely/csc.h"
#include "sparsely/dense.h"

void csc_solve_lower(const csc_t *L, const dense_t *b, dense_t *x);
void csc_solve_upper(const csc_t *L, const dense_t *b, dense_t *x);
void csc_solve_cholesky(const csc_t *L, const dense_t *b, dense_t *x);