#pragma once
#include "sparsely/csr.h"
#include "sparsely/csc.h"
#include "sparsely/dense.h"

dense_t csr_mul_dense(const csr_t *A, const dense_t *x);
csc_t *csr_transpose_to_csc(const csr_t *A); // todo: move to transpose.h
csr_t *csr_mul_csr(const csr_t *A, const csr_t *B);
csr_t *csc_transpose_to_csr(const csc_t *A);
csr_t *csc_mul_csr(const csc_t *A, const csr_t *B);
