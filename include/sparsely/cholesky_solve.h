#pragma once
#include "sparsely/csr.h"
#include "sparsely/dense.h"

void csr_solve_lower(const csr_t *L, const dense_t *b, dense_t *x);
void csr_solve_upper(const csr_t *L, const dense_t *b, dense_t *x);
void csr_solve_cholesky(const csr_t *L, const dense_t *b, dense_t *x);
