#pragma once
#include "sparsely/csr.h"
#include "sparsely/dense.h"

dense_t csr_mul_dense(const csr_t *A, const dense_t *x);
csr_t *csr_mul_csr(const csr_t *A, const csr_t *B);
