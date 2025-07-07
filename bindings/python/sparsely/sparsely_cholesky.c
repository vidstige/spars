#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "sparsely/cholesky.h"
#include "sparsely/cholesky_solve.h"
#include "sparsely/dense.h"

#include "sparsely_csr.h"
#include "sparsely_cholesky.h"

// Extern type from CSR
extern PyTypeObject PyCSRType;

// -------------------- cholesky --------------------

static PyObject *
cholesky_func(PyObject *self, PyObject *args)
{
    PyObject *csr_arg;
    if (!PyArg_ParseTuple(args, "O", &csr_arg))
        return NULL;

    if (!PyObject_TypeCheck(csr_arg, &PyCSRType)) {
        PyErr_SetString(PyExc_TypeError, "Expected CSR object.");
        return NULL;
    }

    csr_t *L = cholesky_factor(((PyCSR *)csr_arg)->csr);
    if (!L) {
        PyErr_SetString(PyExc_RuntimeError, "Factorization failed.");
        return NULL;
    }

    PyCSR *result = PyObject_New(PyCSR, &PyCSRType);
    if (!result) {
        csr_destroy(L);
        return NULL;
    }

    result->csr = L;
    return (PyObject *)result;
}

// -------------------- solve --------------------

static PyObject *
sparse_solve_cholesky(PyObject *self, PyObject *args)
{
    PyObject *L_obj;
    PyObject *b_obj;

    if (!PyArg_ParseTuple(args, "OO", &L_obj, &b_obj))
        return NULL;

    if (!PyObject_TypeCheck(L_obj, &PyCSRType)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be CSR.");
        return NULL;
    }

    PyArrayObject *b_array = (PyArrayObject *)PyArray_FROM_OTF(b_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (!b_array) return NULL;

    if (PyArray_NDIM(b_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "RHS b must be 1D.");
        Py_DECREF(b_array);
        return NULL;
    }

    int n = (int)PyArray_DIM(b_array, 0);
    if (n != ((PyCSR *)L_obj)->csr->nrows) {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch between matrix and RHS.");
        Py_DECREF(b_array);
        return NULL;
    }

    npy_intp dims[1] = {n};
    PyObject *x_array = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!x_array) {
        Py_DECREF(b_array);
        return NULL;
    }

    dense_t b_dense = { n, (double *)PyArray_DATA(b_array) };
    dense_t x_dense = { n, (double *)PyArray_DATA((PyArrayObject *)x_array) };

    csr_solve_cholesky(((PyCSR *)L_obj)->csr, &b_dense, &x_dense);

    Py_DECREF(b_array);
    return x_array;
}

// -------------------- registration --------------------

static PyMethodDef cholesky_methods[] = {
    {"cholesky", cholesky_func, METH_VARARGS, "Compute Cholesky factorization of a CSR matrix."},
    {"solve_cholesky", sparse_solve_cholesky, METH_VARARGS, "Solve LLᵗ x = b for x."},
    {NULL, NULL, 0, NULL}
};

int register_cholesky_functions(PyObject *module)
{
    import_array();
#if PY_VERSION_HEX >= 0x030C0000
    return PyModule_AddFunctions(module, cholesky_methods);
#else
    for (PyMethodDef *def = cholesky_methods; def && def->ml_name; ++def) {
        PyModule_AddObject(module, def->ml_name, PyCFunction_New(def, NULL));
    }
    return 0;
#endif
}
