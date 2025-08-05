#pragma once
#include <stdlib.h>

#if defined(__APPLE__) || defined(__MACH__)
    // BSD-style qsort_r: thunk first
    #define QSORT_R(base, nmemb, size, compar, thunk) \
        qsort_r((base), (nmemb), (size), (thunk), (compar))

    #define SPARS_COMPARE_FUNCTION(function_name, a, b, thunk) \
        int function_name(void *thunk, const void *a, const void *b)

#else
    // GNU-style qsort_r: thunk last
    #define QSORT_R(base, nmemb, size, compar, thunk) \
        qsort_r((base), (nmemb), (size), (compar), (thunk))

    #define SPARS_COMPARE_FUNCTION(function_name, a, b, thunk) \
        int function_name(const void *a, const void *b, void *thunk)

#endif