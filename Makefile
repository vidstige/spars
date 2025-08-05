# Compiler and flags
CC = clang
CFLAGS = -Wall -Wextra -O2 -g -Iinclude

# Directories
SRC_DIR = src
INC_DIR = include
LIB_DIR = lib

# Detect architecture
UNAME_S := $(shell uname -s)
CPU_BRAND := $(shell sysctl -n machdep.cpu.brand_string 2>/dev/null)

# Choose CPU-specific flag
ifeq ($(UNAME_S),Darwin)
    ifneq (,$(findstring Apple M1,$(CPU_BRAND)))
        CPUFLAG = -mcpu=apple-m1
    else
        CPUFLAG = -march=native
    endif
else
    CPUFLAG = -march=native
endif

# Build configuration: default to release
BUILD ?= release

ifeq ($(BUILD),debug)
    CFLAGS = -Wall -Wextra -O0 -g -I$(INC_DIR)
else ifeq ($(BUILD),release)
    CFLAGS = -Wall -Wextra -O3 -mcpu=apple-m1 -ffast-math -fno-math-errno -funroll-loops -flto -I$(INC_DIR) \
             -Rpass=loop-vectorize -Rpass-missed=loop-vectorize -Rpass-analysis=loop-vectorize
else
    $(error Unknown BUILD type: $(BUILD))
endif
OBJ_DIR = obj/$(BUILD)

# Sources and objects
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

# Static library name
LIB = $(LIB_DIR)/libspars.a

# Default target
all: $(LIB)

# Rule to make .a from .o
$(LIB): $(OBJS)
	@mkdir -p $(LIB_DIR)
	ar rcs $@ $^

# Compile .c to .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Ensure obj dir exists
$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

# Clean
clean:
	rm -rf $(OBJ_DIR) $(LIB_DIR)

.PHONY: all clean
