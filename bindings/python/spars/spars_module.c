#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "spars_csr.h"
#include "spars_csc.h"
#include "spars_add.h"
#include "spars_mul.h"
#include "spars_cholesky.h"

static PyModuleDef spars_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "_sparse_c",
    .m_doc = "Light weight sparse matrix library",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit__sparse_c(void)
{
    PyObject *m;

    import_array();

    m = PyModule_Create(&spars_module);
    if (!m)
        return NULL;

    if (register_csr_type(m) < 0)
        return NULL;
    
    if (register_csc_type(m) < 0)
        return NULL;

    if (register_mul_functions(m) < 0)
        return NULL;

    if (register_add_functions(m) < 0)
        return NULL;

    if (register_cholesky_functions(m) < 0)
        return NULL;

    return m;
}
