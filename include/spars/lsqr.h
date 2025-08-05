#pragma once

#include "spars/csr.h"
#include "spars/dense.h"

// Solve Ax = b in least-squares sense using LSQR
// If x0 is not NULL, it will be used as initial guess; otherwise zero vector is used
dense_t lsqr(const csr_t *A, const dense_t *b, const dense_t *x0);
