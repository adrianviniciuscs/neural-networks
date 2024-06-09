CC = gcc
CFLAGS = -Wall -Iinclude
SRCDIR = src
BINDIR = bin

# List of examples
EXAMPLES = $(wildcard examples/*.c)
BINS = $(patsubst examples/%.c, $(BINDIR)/%, $(EXAMPLES))

# Default target
all: $(BINS)

# Rule for compiling each example
$(BINDIR)/%: examples/%.c $(SRCDIR)/nn.c $(SRCDIR)/spiral.c $(SRCDIR)/losscce.c
	@mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ -lm

# Clean target
clean:
	rm -rf $(BINDIR)

.PHONY: all clean

