#pragma once

typedef struct {
    int n;
    double *values;
} dense_t;

dense_t dense_empty(int n); // uninitiaized values
dense_t dense_zeros(int n); // zero initialized values
dense_t dense_copy(int n, const double *data); // copies values
void dense_destroy(const dense_t *v);
