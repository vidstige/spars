# Compiler and flags
CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -Iinclude

# Directories
SRC_DIR = src
OBJ_DIR = obj
LIB_DIR = lib

# Sources and objects
SRCS = $(wildcard $(SRC_DIR)/*.c)
OBJS = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))

# Static library name
LIB = $(LIB_DIR)/libsparsely.a

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
