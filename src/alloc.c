#include <assert.h>
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
void sparsely_free(void *ptr) {
    free(ptr);
}
