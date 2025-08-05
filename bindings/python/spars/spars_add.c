#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "spars_csr.h"

#include "spars/add.h"
#include "spars_add.h"

// Extern type from CSR
extern PyTypeObject PyCSRType;

// -------- csr_add wrapper --------
static PyObject *
py_csr_add_csr(PyObject *self, PyObject *args)
{
    PyObject *lhs_obj;
    PyObject *rhs_obj;

    if (!PyArg_ParseTuple(args, "OO", &lhs_obj, &rhs_obj))
        return NULL;

    if (!PyObject_TypeCheck(lhs_obj, &PyCSRType) || !PyObject_TypeCheck(rhs_obj, &PyCSRType)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be CSR.");
        return NULL;
    }

    csr_t *result = csr_add_csr(
        ((PyCSR *)lhs_obj)->csr,
        ((PyCSR *)rhs_obj)->csr
    );

    if (!result) {
        PyErr_SetString(PyExc_RuntimeError, "csr_add failed.");
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
static PyMethodDef add_methods[] = {
    {"csr_add_csr", py_csr_add_csr, METH_VARARGS, "Multiply CSR with dense vector."},
    {NULL, NULL, 0, NULL}
};

// -------- Register function --------
int register_add_functions(PyObject *module) {
    import_array();

#if PY_VERSION_HEX >= 0x030C0000
    PyModule_AddFunctions(module, mul_methods);
#else
    for (PyMethodDef *def = add_methods; def && def->ml_name; ++def) {
        PyModule_AddObject(module, def->ml_name, PyCFunction_New(def, NULL));
    }
#endif

    return 0;
}
