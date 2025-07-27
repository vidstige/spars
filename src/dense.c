#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "sparsely/dense.h"

dense_t dense_empty(int n) {
    dense_t v;
    v.n = n;
    v.values = malloc(n * sizeof(double));
    if (!v.values) {
        v.n = -1; // indicate failure
    }
    return v;
}

dense_t dense_zeros(int n) {
    dense_t v;
    v.n = n;
    v.values = malloc(n * sizeof(double));
    if (!v.values) {
        v.n = -1; // indicate failure
    } else {
        memset(v.values, 0, n * sizeof(double)); // zero initialize
    }
    return v;
}

dense_t dense_copy(int n, const double *data) {
    assert(data != NULL); // data must not be NULL
    dense_t v;
    v.n = n;
    v.values = malloc(n * sizeof(double));
    if (!v.values) {
        v.n = -1; // indicate failure
    } else {
        memcpy(v.values, data, n * sizeof(double)); // copy values
    }
    return v;
}
void dense_copy_to(dense_t *dst, const dense_t *src) {
    assert(dst->n == src->n);
    for (int i = 0; i < dst->n; ++i)
        dst->values[i] = src->values[i];
}

dense_t dense_clone(const dense_t *src) {
    dense_t copy;
    copy.n = src->n;
    copy.values = malloc(sizeof(double) * src->n);
    for (int i = 0; i < src->n; ++i)
        copy.values[i] = src->values[i];
    return copy;
}

void dense_destroy(const dense_t *v) {
    if (v) free(v->values);
}

// operations
double dense_dot(const dense_t *a, const dense_t *b) {
    assert(a->n == b->n);
    double sum = 0.0;
    for (int i = 0; i < a->n; ++i)
        sum += a->values[i] * b->values[i];
    return sum;
}

double dense_norm(const dense_t *v) {
    return sqrt(dense_dot(v, v));
}

void dense_scale(dense_t *v, double alpha) {
    for (int i = 0; i < v->n; ++i)
        v->values[i] *= alpha;
}

void dense_add_scaled(dense_t *y, double alpha, const dense_t *x) {
    assert(y->n == x->n);
    for (int i = 0; i < y->n; ++i)
        y->values[i] += alpha * x->values[i];
}
