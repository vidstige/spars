#include "spars/alloc.h"
void blasphemy_daxpy(int n, double *restrict y, const double *restrict x, double alpha) {
    double *y_aligned = SPARS_ASSUME_ALIGNED(y);
    const double *x_aligned = SPARS_ASSUME_ALIGNED(x);
    for (int i = 0; i < n; ++i)
        y_aligned[i] += alpha * x_aligned[i];
}
