CC=gcc
CFLAGS=-Wall -Iinclude
LDFLAGS=-lm

# Directories
SRCDIR=src
OBJDIR=obj
BINDIR=bin
INCDIR=include
EXDIR=examples

# Files
SRC=$(wildcard $(SRCDIR)/*.c)
OBJ=$(SRC:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
EXEC=$(BINDIR)/3-batches-layers
EXECS=$(BINDIR)/3-batches-layers $(BINDIR)/4-relu

# Targets
all: $(EXEC) $(EXECS)

$(EXEC): $(OBJ)
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $(OBJDIR)/3-batches-layers.o $(OBJ) -o $@ $(LDFLAGS)

$(BINDIR)/4-relu: $(OBJDIR)/4-relu.o $(OBJ)
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $< $(OBJ) -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/%.o: $(EXDIR)/%.c
	mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR)

