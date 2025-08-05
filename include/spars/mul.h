#pragma once
#include "spars/csr.h"
#include "spars/csc.h"
#include "spars/dense.h"

dense_t csr_mul_dense(const csr_t *A, const dense_t *x);
dense_t csr_transposed_mul_dense(const csr_t *A, const dense_t *x);

csc_t *csr_transpose_to_csc(const csr_t *A); // todo: move to transpose.h
csr_t *csr_mul_csr(const csr_t *A, const csr_t *B);
csr_t *csc_transpose_to_csr(const csc_t *A);
csr_t *csc_mul_csr(const csc_t *A, const csr_t *B);
dense_t csc_mul_dense(const csc_t *A, const dense_t *x);

csr_t *csc_to_csr(const csc_t* csc);
csc_t *csr_to_csc(const csr_t *csr);

