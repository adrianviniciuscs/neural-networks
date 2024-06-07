CC=gcc
CFLAGS=-Wall -Iinclude
LDFLAGS=-lm

# Directories
SRCDIR=src
OBJDIR=obj
BINDIR=bin
INCDIR=include

# Files
SRC=$(wildcard $(SRCDIR)/*.c)
OBJ=$(SRC:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
EXEC=$(BINDIR)/3-batches-layers

# Targets
all: $(EXEC)

$(EXEC): $(OBJ)
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

