#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "sparsely_csr.h"
#include "sparsely/mul.h"
#include "sparsely_mul.h"

// Extern type from CSR
extern PyTypeObject PyCSRType;

// -------- csr_mul_dense wrapper --------
static PyObject *py_csr_mul_dense(PyObject *self, PyObject *args) {
    PyObject *csr_obj;
    PyObject *x_obj;

    if (!PyArg_ParseTuple(args, "OO", &csr_obj, &x_obj))
        return NULL;

    if (!PyObject_TypeCheck(csr_obj, &PyCSRType)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be CSR.");
        return NULL;
    }

    PyArrayObject *x_array = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (!x_array)
        return NULL;

    if (PyArray_NDIM(x_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "RHS must be 1D.");
        Py_DECREF(x_array);
        return NULL;
    }

    int n = (int)PyArray_DIM(x_array, 0);
    if (n != ((PyCSR *)csr_obj)->csr->ncols) {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch in csr_mul_dense.");
        Py_DECREF(x_array);
        return NULL;
    }

    dense_t x_dense = { n, (double *)PyArray_DATA(x_array) };
    dense_t y_dense = csr_mul_dense(((PyCSR *)csr_obj)->csr, &x_dense);

    npy_intp dims[1] = { y_dense.n };
    PyObject *result = PyArray_SimpleNew(1, dims, NPY_FLOAT64);
    if (!result) {
        dense_destroy(&y_dense);
        Py_DECREF(x_array);
        return NULL;
    }

    memcpy(PyArray_DATA((PyArrayObject *)result), y_dense.values, y_dense.n * sizeof(double));
    dense_destroy(&y_dense);
    Py_DECREF(x_array);

    return result;
}

// -------- csr_mul_csr wrapper --------
static PyObject *py_csr_mul_csr(PyObject *self, PyObject *args) {
    PyObject *lhs_obj;
    PyObject *rhs_obj;

    if (!PyArg_ParseTuple(args, "OO", &lhs_obj, &rhs_obj))
        return NULL;

    if (!PyObject_TypeCheck(lhs_obj, &PyCSRType) || !PyObject_TypeCheck(rhs_obj, &PyCSRType)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be CSR.");
        return NULL;
    }

    csr_t *result = csr_mul_csr(
        ((PyCSR *)lhs_obj)->csr,
        ((PyCSR *)rhs_obj)->csr
    );

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "csr_mul_csr failed (Not implemented).");
        return NULL;
    }

    PyCSR *py_result = PyObject_New(PyCSR, &PyCSRType);
    if (!py_result) {
        csr_destroy(result);
        return NULL;
    }

    py_result->csr = result;
    return (PyObject *)py_result;
}

// -------- Method table --------
static PyMethodDef mul_methods[] = {
    {"csr_mul_dense", py_csr_mul_dense, METH_VARARGS, "Multiply CSR with dense vector."},
    {"csr_mul_csr", py_csr_mul_csr, METH_VARARGS, "Multiply CSR with CSR (matrix multiplication)."},
    {NULL, NULL, 0, NULL}
};

// -------- Register function --------
int register_mul_functions(PyObject *module) {
    import_array();

#if PY_VERSION_HEX >= 0x030C0000
    PyModule_AddFunctions(module, mul_methods);
#else
    for (PyMethodDef *def = mul_methods; def && def->ml_name; ++def) {
        PyModule_AddObject(module, def->ml_name, PyCFunction_New(def, NULL));
    }
#endif

    return 0;
}
