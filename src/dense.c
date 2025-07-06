#include <stdlib.h>
#include <string.h>
#include "sparsely/dense.h"

dense_t *dense_create(int n, const double *data)
{
    dense_t *v = malloc(sizeof(dense_t));
    if (!v) return NULL;
    v->n = n;
    v->values = malloc(n * sizeof(double));
    if (!v->values) {
        free(v);
        return NULL;
    }
    memcpy(v->values, data, n * sizeof(double));
    return v;
}

void dense_destroy(const dense_t *v)
{
    if (v) {
        free(v->values);
        free(v);
    }
}