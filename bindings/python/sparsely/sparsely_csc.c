#include <Python.h>
#include <numpy/arrayobject.h>
#include "sparsely/csc.h"
#include "sparsely_csc.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// ----- PyCSC struct -----
typedef struct {
    PyObject_HEAD
    csc_t *csc;
} PyCSC;

// helper functions
static int *copy_int_array(PyArrayObject *arr, int count) {
    int *out = (int *)malloc(count * sizeof(int));
    if (!out) return NULL;
    memcpy(out, PyArray_DATA(arr), count * sizeof(int));
    return out;
}

static double *copy_double_array(PyArrayObject *arr, int count) {
    double *out = (double *)malloc(count * sizeof(double));
    if (!out) return NULL;
    memcpy(out, PyArray_DATA(arr), count * sizeof(double));
    return out;
}


// ----- INIT -----
static int PyCSC_init(PyCSC *self, PyObject *args, PyObject *kwds) {
    fprintf(stderr, "[PyCSC_init] ENTER\n");

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

    int *colptr_copy = copy_int_array(colptr_array, ncols + 1);
    int *rowind_copy = copy_int_array(rowind_array, nnz);
    double *values_copy = copy_double_array(values_array, nnz);

    if (!colptr_copy || !rowind_copy || !values_copy) {
        free(colptr_copy);
        free(rowind_copy);
        free(values_copy);
        PyErr_SetString(PyExc_RuntimeError, "Memory allocation failed in copy.");
        goto fail;
    }

    self->csc = csc_create(
        nrows, ncols, nnz,
        colptr_copy, rowind_copy, values_copy
    );

    if (!self->csc) {
        free(colptr_copy);
        free(rowind_copy);
        free(values_copy);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create CSC matrix.");
        goto fail;
    }

    if (!self->csc) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create CSC matrix.");
        goto fail;
    }

    Py_DECREF(colptr_array);
    Py_DECREF(rowind_array);
    Py_DECREF(values_array);

    fprintf(stderr, "[PyCSC_init] SUCCESS\n");
    return 0;

fail:
    Py_XDECREF(colptr_array);
    Py_XDECREF(rowind_array);
    Py_XDECREF(values_array);
    return -1;
}

// ----- DEALLOC -----
static void PyCSC_dealloc(PyCSC *self) {
    if (self->csc) {
        csc_destroy(self->csc);
    }
    Py_TYPE(self)->tp_free((PyObject *)self);
}

// ----- .shape property -----
static PyObject *PyCSC_get_shape(PyCSC *self, void *closure) {
    return Py_BuildValue("(ii)", self->csc->nrows, self->csc->ncols);
}

static PyGetSetDef PyCSC_getsetters[] = {
    {"shape", (getter)PyCSC_get_shape, NULL, "matrix dimensions", NULL},
    {NULL}
};

// ----- .todense() method -----
static PyObject *PyCSC_todense(PyCSC *self, PyObject *Py_UNUSED(ignored)) {
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
static PyObject *PyCSC_subscript(PyCSC *self, PyObject *key) {
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
