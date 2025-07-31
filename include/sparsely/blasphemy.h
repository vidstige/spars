// Internal header file for the Sparsely library
// Contains a simplified BLAS-like implementation
#pragma once
void blasphemy_daxpy(int n, double *restrict y, const double *restrict x, double alpha);