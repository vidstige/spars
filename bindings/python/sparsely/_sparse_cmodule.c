#include <Python.h>
#include <numpy/arrayobject.h>
#include "sparsely/csr.h"

// PyCSR
typedef struct {
    PyObject_HEAD
    csr_t *csr;
} PyCSR;

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
        return -1;
    }
    PyArrayObject *rowptr_array = (PyArrayObject *)PyArray_FROM_OTF(rowptr_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *colind_array = (PyArrayObject *)PyArray_FROM_OTF(colind_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *values_array = (PyArrayObject *)PyArray_FROM_OTF(values_obj, NPY_FLOAT64, NPY_ARRAY_IN_ARRAY);

    if (!rowptr_array || !colind_array || !values_array) {
        Py_XDECREF(rowptr_array);
        Py_XDECREF(colind_array);
        Py_XDECREF(values_array);
        PyErr_SetString(PyExc_ValueError, "Failed to convert inputs to NumPy arrays.");
        return -1;
    }
    
    // validate shapes  
    int nnz = (int)PyArray_DIM(colind_array, 0);
    if (PyArray_DIM(values_array, 0) != nnz) {
        PyErr_SetString(PyExc_ValueError, "colind and values must have same length.");
        goto fail;
    }

    if (PyArray_DIM(rowptr_array, 0) != nrows + 1) {
        PyErr_SetString(PyExc_ValueError, "rowptr must have length nrows + 1.");
        goto fail;
    }

    self->csr = csr_create(
        nrows, ncols, nnz,
        (int *)PyArray_DATA(rowptr_array),
        (int *)PyArray_DATA(colind_array),
        (double *)PyArray_DATA(values_array)
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

static PyTypeObject PyCSRType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_sparse_c.CSR",
    .tp_basicsize = sizeof(PyCSR),
    .tp_dealloc = (destructor)PyCSR_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Compressed Sparse Row Matrix",
    .tp_init = (initproc)PyCSR_init,
    .tp_new = PyType_GenericNew,
};

// _sparse_c module definition
static PyModuleDef _sparse_c_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_sparse_c",
    .m_doc = "Light weight sparse matrix library",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit__sparse_c(void)
{
    PyObject *m;
    if (PyType_Ready(&PyCSRType) < 0)
        return NULL;

    import_array();

    m = PyModule_Create(&_sparse_c_module);
    if (!m)
        return NULL;

    Py_INCREF(&PyCSRType);
    PyModule_AddObject(m, "CSR", (PyObject *)&PyCSRType);

    return m;
}