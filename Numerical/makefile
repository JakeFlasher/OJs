# Makefile for multiple .cc C++ source files with tunable optimizations

# Default optimization: can be overridden by command-line or by the o2/o3 targets.
OPT ?= -O2

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 $(OPT) -Wall

# Automatically detect all .cc files in the directory.
SOURCES := $(wildcard *.cc)
# Create executables by removing the .cc suffix from each source file.
TARGETS := $(SOURCES:.cc=)

.PHONY: all run clean o2 o3

# Default target: Build all executables.
all: $(TARGETS)

# Pattern rule: For each executable, compile its corresponding .cc file.
%: %.cc
	$(CXX) $(CXXFLAGS) -o $@ $<

# run target: build all executables and run each one.
run: all
	@for exe in $(TARGETS); do \
		echo "Running $$exe:"; \
		./$$exe; \
		echo ""; \
	done

# clean target: remove all executables.
clean:
	rm -f $(TARGETS)

# o2 target: Clean and rebuild with -O2 optimization.
o2:
	$(MAKE) clean OPT=-O2
	$(MAKE) OPT=-O2

# o3 target: Clean and rebuild with -O3 optimization.
o3:
	$(MAKE) clean OPT=-O3
	$(MAKE) OPT=-O3
