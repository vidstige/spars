void blasphemy_daxpy(int n, double *restrict y, const double *restrict x, double alpha) {
    for (int i = 0; i < n; ++i)
        y[i] += alpha * x[i];
}
