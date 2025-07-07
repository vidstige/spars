#pragma once
#include <Python.h>
#include "sparsely/csr.h"

typedef struct {
    PyObject_HEAD
    csr_t *csr;
} PyCSR;

int register_csr_type(PyObject *module);
