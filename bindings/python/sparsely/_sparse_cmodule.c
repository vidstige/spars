#include <Python.h>
#include <numpy/arrayobject.h>

#include "sparsely/csr.h"
#include "sparsely/cholesky.h"

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

static PyObject *
PyCSR_get_shape(PyCSR *self, void *closure)
{
    return Py_BuildValue("(ii)", self->csr->nrows, self->csr->ncols);
}

static PyGetSetDef PyCSR_getsetters[] = {
    {"shape", (getter)PyCSR_get_shape, NULL, "matrix dimensions", NULL},
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

static PyTypeObject PyCSRType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "_sparse_c.CSR",
    .tp_basicsize = sizeof(PyCSR),
    .tp_dealloc = (destructor)PyCSR_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "Compressed Sparse Row Matrix",
    .tp_init = (initproc)PyCSR_init,
    .tp_new = PyType_GenericNew,
    .tp_getset = PyCSR_getsetters,
    .tp_as_mapping = &PyCSR_mappingmethods,
};

// cholesky function
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


// _sparse_c module definition
static PyModuleDef _sparse_c_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_sparse_c",
    .m_doc = "Light weight sparse matrix library",
    .m_size = -1,
};

static PyMethodDef module_methods[] = {
    {"cholesky", cholesky_func, METH_VARARGS, "Compute Cholesky factorization of a CSR matrix."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
PyInit__sparse_c(void)
{
    PyObject *m;
    if (PyType_Ready(&PyCSRType) < 0)
        return NULL;

    import_array(); // Initialize NumPy C API

    m = PyModule_Create(&_sparse_c_module);
    if (!m)
        return NULL;

    Py_INCREF(&PyCSRType);
    PyModule_AddObject(m, "CSR", (PyObject *)&PyCSRType);

#if PY_VERSION_HEX >= 0x030C0000
    PyModule_AddFunctions(m, module_methods);
#else
    for (PyMethodDef *def = module_methods; def && def->ml_name; ++def) {
        PyModule_AddObject(m, def->ml_name,
                           PyCFunction_New(def, NULL));
    }
#endif

    return m;
}