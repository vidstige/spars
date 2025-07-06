#pragma once

typedef struct {
    int n;
    double *values;
} dense_t;

dense_t *dense_create(int n, const double *data);
void dense_destroy(const dense_t *v);
