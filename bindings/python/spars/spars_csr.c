#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "spars_csr.h"

#include "spars/csr.h"
#include "spars/dense.h"
#include "spars/mul.h"
#include "spars_csc.h"

// Extern type from CSC
extern PyTypeObject PyCSCType;


// ---------- Init and dealloc ----------
static int
PyCSR_init(PyCSR *self, PyObject *args, PyObject *kwds)
{
    int nrows, ncols;
    PyObject *rowptr_obj = NULL;
    PyObject *colind_obj = NULL;
    PyObject *values_obj = NULL;

    static char *kwlist[] = {"nrows", "ncols", "rowptr", "colind", "values", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiOOO", kwlist,
                                     &nrows, &ncols,
                                     &rowptr_obj, &colind_obj, &values_obj)) {
        fprintf(stderr, "[PyCSR_init] ERROR: PyArg_ParseTupleAndKeywords failed!\n");
        return -1;
    }

    PyArrayObject *rowptr_array = (PyArrayObject *)PyArray_FROM_OTF(rowptr_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *colind_array = (PyArrayObject *)PyArray_FROM_OTF(colind_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_array = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    if (!rowptr_array || !colind_array || !values_array) {
        fprintf(stderr, "[PyCSR_init] ERROR: One or more arrays failed conversion!\n");
        if (!rowptr_array) fprintf(stderr, "  -> rowptr_array is NULL\n");
        if (!colind_array) fprintf(stderr, "  -> colind_array is NULL\n");
        if (!values_array) fprintf(stderr, "  -> values_array is NULL\n");

        Py_XDECREF(rowptr_array);
        Py_XDECREF(colind_array);
        Py_XDECREF(values_array);
        PyErr_SetString(PyExc_ValueError, "Failed to convert inputs to NumPy arrays.");
        return -1;
    }

    int nnz = (int)PyArray_DIM(colind_array, 0);

    if (PyArray_DIM(values_array, 0) != nnz) {
        fprintf(stderr, "[PyCSR_init] ERROR: colind and values length mismatch! values=%ld, nnz=%d\n",
                PyArray_DIM(values_array, 0), nnz);
        PyErr_SetString(PyExc_ValueError, "colind and values must have same length.");
        goto fail;
    }

    if (PyArray_DIM(rowptr_array, 0) != nrows + 1) {
        fprintf(stderr, "[PyCSR_init] ERROR: rowptr length wrong! got=%ld, expected=%d\n",
                PyArray_DIM(rowptr_array, 0), nrows + 1);
        PyErr_SetString(PyExc_ValueError, "rowptr must have length nrows + 1.");
        goto fail;
    }

    self->csr = csr_create(
        nrows, ncols, nnz,
        PyArray_DATA(rowptr_array),
        PyArray_DATA(colind_array),
        PyArray_DATA(values_array)
    );

    if (!self->csr) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create CSR matrix.");
        goto fail;
    }

    Py_DECREF(rowptr_array);
    Py_DECREF(colind_array);
    Py_DECREF(values_array);
    return 0;

fail:
    Py_XDECREF(rowptr_array);
    Py_XDECREF(colind_array);
    Py_XDECREF(values_array);
    return -1;
}


static void
PyCSR_dealloc(PyCSR *self)
{
    if (self->csr) {
        csr_destroy(self->csr);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}


// ---------- Methods ----------
static PyObject *
PyCSR_todense(PyCSR *self, PyObject *Py_UNUSED(ignored))
{
    npy_intp dims[2] = {self->csr->nrows, self->csr->ncols};
    PyObject *result = PyArray_ZEROS(2, dims, NPY_FLOAT64, 0);
    if (!result) return NULL;

    double *data = (double *)PyArray_DATA((PyArrayObject *)result);

    for (int i = 0; i < self->csr->nrows; i++) {
        for (int idx = self->csr->rowptr[i]; idx < self->csr->rowptr[i+1]; idx++) {
            int j = self->csr->colind[idx];
            double val = self->csr->values[idx];
            data[i * self->csr->ncols + j] = val;
        }
    }

    return result;
}

// ---------- Properties ----------

static PyObject *
PyCSR_get_shape(PyCSR *self, void *closure)
{
    return Py_BuildValue("(ii)", self->csr->nrows, self->csr->ncols);
}

// transpose
static PyObject *
PyCSR_T(PyCSR *self, void *closure)
{
    csc_t *result = csr_transpose_to_csc(self->csr);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Transpose failed.");
        return NULL;
    }

    PyCSC *py_result = PyObject_New(PyCSC, &PyCSCType);
    if (!py_result) {
        csc_destroy(result);
        return NULL;
    }

    py_result->csc = result;
    return (PyObject *)py_result;
}


static PyMethodDef PyCSR_methods[] = {
    {"todense", (PyCFunction)PyCSR_todense, METH_NOARGS, "Convert to dense NumPy array."},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef PyCSR_getsetters[] = {
    {"shape", (getter)PyCSR_get_shape, NULL, "matrix dimensions", NULL},
    {"T", (getter)PyCSR_T, NULL, "Transpose (returns CSC)", NULL},
    {NULL}  /* Sentinel */
};

static PyObject *
PyCSR_subscript(PyCSR *self, PyObject *key)
{
    // Expect key as tuple (i, j)
    if (!PyTuple_Check(key) || PyTuple_Size(key) != 2) {
        PyErr_SetString(PyExc_TypeError, "CSR indices must be a 2-tuple");
        return NULL;
    }

    PyObject *i_obj = PyTuple_GetItem(key, 0);
    PyObject *j_obj = PyTuple_GetItem(key, 1);

    int i = (int)PyLong_AsLong(i_obj);
    int j = (int)PyLong_AsLong(j_obj);

    if (i < 0 || i >= self->csr->nrows || j < 0 || j >= self->csr->ncols) {
        PyErr_SetString(PyExc_IndexError, "index out of bounds");
        return NULL;
    }

    // Look in row i
    int start = self->csr->rowptr[i];
    int end = self->csr->rowptr[i + 1];

    for (int idx = start; idx < end; ++idx) {
        if (self->csr->colind[idx] == j) {
            return PyFloat_FromDouble(self->csr->values[idx]);
        }
    }

    return PyFloat_FromDouble(0.0);
}

static PyMappingMethods PyCSR_mappingmethods = {
    .mp_length = NULL,
    .mp_subscript = (binaryfunc)PyCSR_subscript,
    .mp_ass_subscript = NULL
};

PyTypeObject PyCSRType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_sparse_c.CSR",
    .tp_basicsize = sizeof(PyCSR),
    .tp_dealloc = (destructor)PyCSR_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Compressed Sparse Row Matrix",
    .tp_init = (initproc)PyCSR_init,
    .tp_new = PyType_GenericNew,
    .tp_methods = PyCSR_methods,
    .tp_getset = PyCSR_getsetters,
    .tp_as_mapping = &PyCSR_mappingmethods,
};

// ---------- Register ----------

int register_csr_type(PyObject *module)
{
    import_array();

    if (PyType_Ready(&PyCSRType) < 0)
        return -1;

    Py_INCREF(&PyCSRType);
    if (PyModule_AddObject(module, "CSR", (PyObject *)&PyCSRType) < 0)
        return -1;

    return 0;
}
