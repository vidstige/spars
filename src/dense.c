#include <assert.h>
#include <stdlib.h>
#include <string.h>
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

void dense_destroy(const dense_t *v) {
    if (v) free(v->values);
}
