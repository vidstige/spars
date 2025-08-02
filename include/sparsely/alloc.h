#pragma once
#include <stdlib.h>
#define SPARSELY_ALIGNMENT 32
#define SPARSELY_ALLOC(type, count) ((type *)sparsely_alloc(SPARSELY_ALIGNMENT, sizeof(type) * (count)))
#define SPARSELY_ASSUME_ALIGNED(ptr) ((typeof(ptr))__builtin_assume_aligned((ptr), SPARSELY_ALIGNMENT))
void *sparsely_alloc(size_t alignment, size_t size);
void *sparsely_realloc(const void *old_ptr, size_t old_size, size_t new_size, size_t alignment);
void sparsely_free(void *ptr);
