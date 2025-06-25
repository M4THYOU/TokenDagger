# Simple Makefile for tiktoken library
# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -fPIC
INCLUDES = -I.

# Source files
SOURCES = tiktoken.cpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = libtiktoken.a

# Default target
all: $(TARGET)

# Build static library
$(TARGET): $(OBJECTS)
	ar rcs $@ $^

# Compile source files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install target (optional)
install: $(TARGET)
	cp $(TARGET) /usr/local/lib/
	cp tiktoken.h /usr/local/include/

# Phony targets
.PHONY: all clean install

