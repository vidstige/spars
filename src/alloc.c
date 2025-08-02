#include <assert.h>
#include <string.h>
#include "sparsely/alloc.h"

void *sparsely_alloc(size_t alignment, size_t size) {
    assert((alignment & (alignment - 1)) == 0); // must be power of two
    assert(alignment >= sizeof(void *));

    void *ptr = NULL;
    int err = posix_memalign(&ptr, alignment, size);
    if (err != 0) {
        return NULL;
    }
    return ptr;
}

void *sparsely_realloc(const void *old_ptr, size_t old_size, size_t new_size, size_t alignment) {
    void *new_ptr = sparsely_alloc(alignment, new_size);
    if (!new_ptr)
        return NULL;

    size_t min_size = old_size < new_size ? old_size : new_size;
    memcpy(new_ptr, old_ptr, min_size);
    sparsely_free(old_ptr);
    return new_ptr;
}

void sparsely_free(void *ptr) {
    free(ptr);
}
