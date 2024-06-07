CC=gcc
CFLAGS=-Wall -Iinclude
OBJ_DIR=obj
BIN_DIR=bin

SRC_FILES := $(wildcard src/*.c)
OBJ_FILES := $(patsubst src/%.c, $(OBJ_DIR)/%.o, $(SRC_FILES))

.PHONY: all clean

all: $(BIN_DIR)/3-batches-layers

$(BIN_DIR)/3-batches-layers: $(OBJ_DIR)/3-batches-layers.o $(OBJ_FILES)
	$(CC) $(CFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: src/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR)/3-batches-layers.o: examples/3-batches-layers.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

