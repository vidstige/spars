#pragma once
#include <Python.h>
#include "sparsely/csc.h"

typedef struct {
    PyObject_HEAD
    csc_t *csc;
} PyCSC;

int register_csc_type(PyObject *module);
