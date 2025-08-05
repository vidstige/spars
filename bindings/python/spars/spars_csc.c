#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "spars_csc.h"
#include "spars_csr.h"

#include "spars/alloc.h"
#include "spars/csr.h"
#include "spars/csc.h"
#include "spars/mul.h"

// Extern type from CSR
extern PyTypeObject PyCSRType;

// ----- INIT -----
static int PyCSC_init(PyCSC *self, PyObject *args, PyObject *kwds) {
    int nrows, ncols;
    PyObject *colptr_obj = NULL;
    PyObject *rowind_obj = NULL;
    PyObject *values_obj = NULL;

    static char *kwlist[] = {"nrows", "ncols", "colptr", "rowind", "values", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "iiOOO", kwlist,
                                     &nrows, &ncols,
                                     &colptr_obj, &rowind_obj, &values_obj)) {
        PyErr_SetString(PyExc_TypeError, "CSC requires nrows, ncols, colptr, rowind, values.");
        return -1;
    }

    // Convert to NumPy arrays
    PyArrayObject *colptr_array = (PyArrayObject *)PyArray_FROM_OTF(colptr_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *rowind_array = (PyArrayObject *)PyArray_FROM_OTF(rowind_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_array = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    if (!colptr_array || !rowind_array || !values_array) {
        PyErr_SetString(PyExc_ValueError, "Failed to convert inputs to NumPy arrays.");
        Py_XDECREF(colptr_array);
        Py_XDECREF(rowind_array);
        Py_XDECREF(values_array);
        return -1;
    }

    int nnz = (int)PyArray_DIM(rowind_array, 0);
    if (PyArray_DIM(values_array, 0) != nnz) {
        PyErr_SetString(PyExc_ValueError, "rowind and values must have same length.");
        goto fail;
    }

    if (PyArray_DIM(colptr_array, 0) != ncols + 1) {
        PyErr_SetString(PyExc_ValueError, "colptr must have length ncols + 1.");
        goto fail;
    }

    self->csc = csc_create(
        nrows, ncols, nnz,
        PyArray_DATA(colptr_array),
        PyArray_DATA(rowind_array),
        PyArray_DATA(values_array)
    );

    if (!self->csc) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create CSC matrix.");
        goto fail;
    }

    Py_DECREF(colptr_array);
    Py_DECREF(rowind_array);
    Py_DECREF(values_array);

    return 0;

fail:
    Py_XDECREF(colptr_array);
    Py_XDECREF(rowind_array);
    Py_XDECREF(values_array);
    return -1;
}

static PyObject *
PyCSC_fromdense(PyTypeObject *type, PyObject *args)
{
    PyObject *input;
    if (!PyArg_ParseTuple(args, "O", &input)) {
        return NULL;
    }

    PyArrayObject *array = (PyArrayObject *)PyArray_FROM_OTF(input, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);
    if (!array) return NULL;

    if (PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Input must be a 2D array");
        Py_DECREF(array);
        return NULL;
    }

    int nrows = (int)PyArray_DIM(array, 0);
    int ncols = (int)PyArray_DIM(array, 1);
    double *data = (double *)PyArray_DATA(array);

    // Count nnz per column
    int *col_counts = calloc(ncols, sizeof(int));
    if (!col_counts) goto fail;

    int nnz = 0;
    for (int j = 0; j < ncols; ++j) {
        for (int i = 0; i < nrows; ++i) {
            double val = data[i * ncols + j];
            if (val != 0.0) {
                col_counts[j]++;
                nnz++;
            }
        }
    }

    // Allocate CSC arrays
    int *colptr = malloc((ncols + 1) * sizeof(int));
    int *rowind = malloc(nnz * sizeof(int));
    double *values = spars_alloc(SPARS_ALIGNMENT, nnz * sizeof(double));
    if (!colptr || !rowind || !values) goto fail;

    // Fill colptr
    colptr[0] = 0;
    for (int j = 0; j < ncols; ++j)
        colptr[j + 1] = colptr[j] + col_counts[j];

    // Fill rowind and values
    int *offset = calloc(ncols, sizeof(int));
    if (!offset) goto fail;

    for (int j = 0; j < ncols; ++j) {
        int base = colptr[j];
        for (int i = 0; i < nrows; ++i) {
            double val = data[i * ncols + j];
            if (val != 0.0) {
                int idx = base + offset[j];
                rowind[idx] = i;
                values[idx] = val;
                offset[j]++;
            }
        }
    }

    // Construct CSC object
    csc_t *csc = malloc(sizeof(csc_t));
    if (!csc) goto fail;

    csc->nrows = nrows;
    csc->ncols = ncols;
    csc->nnz = nnz;
    csc->colptr = colptr;
    csc->rowind = rowind;
    csc->values = values;

    PyCSC *self = (PyCSC *)type->tp_alloc(type, 0);
    if (!self) {
        free(csc); // Assuming freeing here is fine — no deep free needed as fallback
        return NULL;
    }

    self->csc = csc;

    free(offset);
    free(col_counts);
    Py_DECREF(array);
    return (PyObject *)self;

fail:
    PyErr_SetString(PyExc_MemoryError, "Failed to allocate CSC matrix from dense");
    Py_XDECREF(array);
    free(col_counts);
    free(offset);
    free(colptr);
    free(rowind);
    spars_free(values);
    return NULL;
}

// ----- DEALLOC -----
static void PyCSC_dealloc(PyCSC *self) {
    if (self->csc) {
        csc_destroy(self->csc);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// ----- .shape property -----
static PyObject *
PyCSC_get_shape(PyCSC *self, void *closure)
{
    return Py_BuildValue("(ii)", self->csc->nrows, self->csc->ncols);
}

// transpose
static PyObject *
PyCSC_T(PyCSC *self, void *closure)
{
    csr_t *result = csc_transpose_to_csr(self->csc);
    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "Transpose failed.");
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

static PyGetSetDef PyCSC_getsetters[] = {
    {"shape", (getter)PyCSC_get_shape, NULL, "matrix dimensions", NULL},
    {"T", (getter)PyCSC_T, NULL, "Transpose (returns CSR)", NULL},
    {NULL}
};

// ----- .todense() method -----
static PyObject *
PyCSC_sort_indices(PyCSC *self, PyObject *Py_UNUSED(ignored))
{
    csc_sort_indices(self->csc);
    Py_RETURN_NONE;
}

static PyObject *
PyCSC_todense(PyCSC *self, PyObject *Py_UNUSED(ignored))
{
    npy_intp dims[2] = {self->csc->nrows, self->csc->ncols};
    PyObject *result = PyArray_ZEROS(2, dims, NPY_FLOAT64, 0);
    if (!result) return NULL;

    double *data = (double *)PyArray_DATA((PyArrayObject *)result);

    // Fill dense
    for (int j = 0; j < self->csc->ncols; j++) {
        for (int idx = self->csc->colptr[j]; idx < self->csc->colptr[j + 1]; idx++) {
            int i = self->csc->rowind[idx];
            double val = self->csc->values[idx];
            data[i * self->csc->ncols + j] = val;
        }
    }

    return result;
}

// index operator
static PyObject *
PyCSC_subscript(PyCSC *self, PyObject *key)
{
    // Expect key as tuple (i, j)
    if (!PyTuple_Check(key) || PyTuple_Size(key) != 2) {
        PyErr_SetString(PyExc_TypeError, "CSC indices must be a 2-tuple");
        return NULL;
    }

    PyObject *i_obj = PyTuple_GetItem(key, 0);
    PyObject *j_obj = PyTuple_GetItem(key, 1);

    int i = (int)PyLong_AsLong(i_obj);
    int j = (int)PyLong_AsLong(j_obj);

    if (i < 0 || i >= self->csc->nrows || j < 0 || j >= self->csc->ncols) {
        PyErr_SetString(PyExc_IndexError, "index out of bounds");
        return NULL;
    }

    // Look in column j
    int start = self->csc->colptr[j];
    int end = self->csc->colptr[j + 1];

    for (int idx = start; idx < end; ++idx) {
        if (self->csc->rowind[idx] == i) {
            return PyFloat_FromDouble(self->csc->values[idx]);
        }
    }

    return PyFloat_FromDouble(0.0);
}

static PyMethodDef PyCSC_methods[] = {
    {"fromdense", (PyCFunction)PyCSC_fromdense, METH_VARARGS | METH_CLASS, "Create CSC matrix from dense"},
    {"sort_indices", (PyCFunction)PyCSC_sort_indices, METH_NOARGS, "Sort colind within rows and move diagonal to last."},
    {"todense", (PyCFunction)PyCSC_todense, METH_NOARGS, "Convert to dense NumPy array."},
    {NULL, NULL, 0, NULL}
};

static PyMappingMethods PyCSC_mappingmethods = {
    .mp_length = NULL,
    .mp_subscript = (binaryfunc)PyCSC_subscript,
    .mp_ass_subscript = NULL
};

// ----- PyTypeObject -----
PyTypeObject PyCSCType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_sparse_c.CSC",
    .tp_basicsize = sizeof(PyCSC),
    .tp_dealloc = (destructor)PyCSC_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Compressed Sparse Column Matrix",
    .tp_init = (initproc)PyCSC_init,
    .tp_new = PyType_GenericNew,
    .tp_methods = PyCSC_methods,
    .tp_getset = PyCSC_getsetters,
    .tp_as_mapping = &PyCSC_mappingmethods,
};

// ----- Registration -----
int register_csc_type(PyObject *module) {
    import_array();

    if (PyType_Ready(&PyCSCType) < 0)
        return -1;

    Py_INCREF(&PyCSCType);
    if (PyModule_AddObject(module, "CSC", (PyObject *)&PyCSCType) < 0)
        return -1;

    return 0;
}
