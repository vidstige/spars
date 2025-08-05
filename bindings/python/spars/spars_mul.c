#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "spars_csr.h"
#include "spars_csc.h"

#include "spars/mul.h"
#include "spars_mul.h"

// Extern type from CSR
extern PyTypeObject PyCSRType;
extern PyTypeObject PyCSCType;

// -------- csr_mul_dense wrapper --------
static PyObject *
py_csr_mul_dense(PyObject *self, PyObject *args)
{
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
static PyObject *
py_csr_mul_csr(PyObject *self, PyObject *args)
{
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
        PyErr_SetString(PyExc_RuntimeError, "csr_mul_csr failed (Out of memory).");
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

static PyObject *
wrap_csc_mul_csr(PyObject *self, PyObject *args)
{
    PyObject *A_obj;
    PyObject *B_obj;

    if (!PyArg_ParseTuple(args, "OO", &A_obj, &B_obj))
        return NULL;

    if (!PyObject_TypeCheck(A_obj, &PyCSCType) || !PyObject_TypeCheck(B_obj, &PyCSRType)) {
        PyErr_SetString(PyExc_TypeError, "Expected (CSC, CSR).");
        return NULL;
    }

    csr_t *result = csc_mul_csr(((PyCSC *)A_obj)->csc, ((PyCSR *)B_obj)->csr);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Multiplication failed.");
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

static PyObject *
wrap_csc_mul_dense(PyObject *self, PyObject *args)
{
    PyObject *A_obj;
    PyObject *x_obj;

    if (!PyArg_ParseTuple(args, "OO", &A_obj, &x_obj))
        return NULL;

    if (!PyObject_TypeCheck(A_obj, &PyCSCType)) {
        PyErr_SetString(PyExc_TypeError, "First argument must be CSC.");
        return NULL;
    }

    PyArrayObject *x_array = (PyArrayObject *)PyArray_FROM_OTF(x_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (!x_array) return NULL;

    if (PyArray_NDIM(x_array) != 1) {
        PyErr_SetString(PyExc_ValueError, "Dense vector must be 1D.");
        Py_DECREF(x_array);
        return NULL;
    }

    int n = (int)PyArray_DIM(x_array, 0);
    if (n != ((PyCSC *)A_obj)->csc->ncols) {
        PyErr_SetString(PyExc_ValueError, "Dimension mismatch.");
        Py_DECREF(x_array);
        return NULL;
    }

    dense_t x_dense = { n, (double *)PyArray_DATA(x_array) };

    dense_t y_dense = csc_mul_dense(((PyCSC *)A_obj)->csc, &x_dense);

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

// -------- csc_to_csr wrapper --------
static PyObject *
py_csc_to_csr(PyObject *self, PyObject *args)
{
    PyObject *csc_obj;

    if (!PyArg_ParseTuple(args, "O", &csc_obj))
        return NULL;

    if (!PyObject_TypeCheck(csc_obj, &PyCSCType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a CSC matrix.");
        return NULL;
    }

    PyCSC *py_csc = (PyCSC *)csc_obj;
    csr_t *csr = csc_to_csr(py_csc->csc);
    if (!csr) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to convert CSC to CSR.");
        return NULL;
    }

    PyObject *py_csr = PyObject_New(PyCSR, &PyCSRType);
    if (!py_csr) {
        csr_destroy(csr);  // free struct and buffers
        return NULL;
    }

    ((PyCSR *)py_csr)->csr = csr;
    return py_csr;
}

static PyObject *
py_csr_to_csc(PyObject *self, PyObject *args)
{
    PyObject *csr_obj;

    if (!PyArg_ParseTuple(args, "O", &csr_obj))
        return NULL;

    if (!PyObject_TypeCheck(csr_obj, &PyCSRType)) {
        PyErr_SetString(PyExc_TypeError, "Argument must be a CSR matrix.");
        return NULL;
    }

    PyCSR *py_csr = (PyCSC *)csr_obj;
    csc_t *csc = csr_to_csc(py_csr->csr);
    if (!csc) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to convert CSR to CSC.");
        return NULL;
    }

    PyObject *py_csc = PyObject_New(PyCSR, &PyCSCType);
    if (!py_csc) {
        csr_destroy(csc);  // free struct and buffers
        return NULL;
    }

    ((PyCSC *)py_csc)->csc = csc;
    return py_csc;
}

// -------- Method table --------
static PyMethodDef mul_methods[] = {
    {"csr_mul_dense", py_csr_mul_dense, METH_VARARGS, "Multiply CSR with dense vector."},
    {"csr_mul_csr", py_csr_mul_csr, METH_VARARGS, "Multiply CSR with CSR (matrix multiplication)."},
    {"csc_mul_csr", wrap_csc_mul_csr, METH_VARARGS, "Multiply CSC * CSR → CSR."},
    {"csc_mul_dense", wrap_csc_mul_dense, METH_VARARGS, "CSC * dense → dense."},
    {"csc_to_csr", py_csc_to_csr, METH_VARARGS, "Convert CSC matrix to CSR format."},
    {"csr_to_csc", py_csr_to_csc, METH_VARARGS, "Convert CSR matrix to CSC format."},    
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
