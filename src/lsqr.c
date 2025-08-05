#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "spars/lsqr.h"
#include "spars/dense.h"
#include "spars/csr.h"
#include "spars/mul.h"   

#define MAX_ITER 1000
#define TOLERANCE 1e-8

dense_t lsqr(const csr_t *A, const dense_t *b, const dense_t *x0) {
    int m = A->nrows;
    int n = A->ncols;

    //dense_t *x = x0 ? dense_clone(x0) : malloc(sizeof(dense_t));
    //if (!x0) *x = dense_zeros(n);
    dense_t x = x0 ? dense_clone(x0) : dense_zeros(n);

    // u = b - A * x
    dense_t Ax = csr_mul_dense(A, &x);
    dense_t u = dense_zeros(m);
    for (int i = 0; i < m; ++i)
        u.values[i] = b->values[i] - Ax.values[i];
    free(Ax.values);

    double beta = dense_norm(&u);
    if (beta == 0.0) return x;
    dense_scale(&u, 1.0 / beta);

    // v = Aᵗ * u
    dense_t v = csr_transposed_mul_dense(A, &u);
    double alpha = dense_norm(&v);
    dense_scale(&v, 1.0 / alpha);

    dense_t w = dense_clone(&v);
    double phi_bar = beta;
    double rho_bar = alpha;

    dense_t dx = dense_zeros(n);

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // u = A * v - alpha * u
        dense_t Av = csr_mul_dense(A, &v);
        dense_add_scaled(&Av, -alpha, &u);
        u = Av;

        beta = dense_norm(&u);
        if (beta != 0.0)
            dense_scale(&u, 1.0 / beta);

        // v = Aᵗ * u - beta * v
        dense_t Atu = csr_transposed_mul_dense(A, &u);
        dense_add_scaled(&Atu, -beta, &v);
        v = Atu;

        alpha = dense_norm(&v);
        if (alpha != 0.0)
            dense_scale(&v, 1.0 / alpha);

        // plane rotation
        double rho = hypot(rho_bar, beta);
        double c = rho_bar / rho;
        double s = beta / rho;
        double theta = s * alpha;
        rho_bar = -c * alpha;
        double phi = c * phi_bar;
        phi_bar = s * phi_bar;

        // x += (phi / rho) * w
        dense_add_scaled(&dx, phi / rho, &w);

        // w = v - (theta / rho) * w
        dense_scale(&w, -theta / rho);
        dense_add_scaled(&w, 1.0, &v);

        // check convergence
        if (fabs(phi_bar) < TOLERANCE * beta)
            break;
    }

    // x += dx
    dense_add_scaled(&x, 1.0, &dx);

    // clean up
    free(u.values);
    free(v.values);
    free(w.values);
    free(dx.values);

    return x;
}
