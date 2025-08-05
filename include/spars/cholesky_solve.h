#pragma once
#include "spars/csc.h"
#include "spars/csr.h"
#include "spars/dense.h"

void csr_solve_lower(const csr_t *L, const dense_t *b, dense_t *x);
void csr_solve_upper(const csr_t *U, const dense_t *b, dense_t *x);
void csc_solve_cholesky(const csc_t *L, const dense_t *b, dense_t *x);
