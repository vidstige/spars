#pragma once
#include <stdlib.h>
#define SPARSELY_ALIGNMENT 32
#define SPARSELY_ALLOC(type, count) ((type *)sparsely_alloc(SPARSELY_ALIGNMENT, sizeof(type) * (count)))
void *sparsely_alloc(size_t alignment, size_t size);
void sparsely_free(void *ptr);
