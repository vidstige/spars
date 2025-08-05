#pragma once
#include <stdlib.h>
#define SPARS_ALIGNMENT 32
#define SPARS_ALLOC(type, count) ((type *)spars_alloc(SPARS_ALIGNMENT, sizeof(type) * (count)))
#define SPARS_ASSUME_ALIGNED(ptr) ((typeof(ptr))__builtin_assume_aligned((ptr), SPARS_ALIGNMENT))
void *spars_alloc(size_t alignment, size_t size);
void *spars_realloc(const void *old_ptr, size_t old_size, size_t new_size, size_t alignment);
void spars_free(void *ptr);
