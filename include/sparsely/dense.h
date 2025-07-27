#pragma once

typedef struct {
    int n;
    double *values;
} dense_t;

dense_t dense_empty(int n); // uninitiaized values
dense_t dense_zeros(int n); // zero initialized values
dense_t dense_copy(int n, const double *data); // copies values
void dense_copy_to(dense_t *dst, const dense_t *src);
dense_t *dense_clone(const dense_t *src);
void dense_destroy(const dense_t *v);

double dense_dot(const dense_t *a, const dense_t *b);
double dense_norm(const dense_t *v);
void dense_scale(dense_t *v, double alpha);
void dense_add_scaled(dense_t *y, double alpha, const dense_t *x);  // y ← y + αx
