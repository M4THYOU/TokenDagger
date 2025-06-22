# Makefile for C++ project with tiktoken

# Compiler settings
CXX = g++

# Compiler flags for release build
CXX_FLAGS_RELEASE = -std=c++17 -O2 -fPIC -w

# Compiler flags for debug build (with debug symbols for GDB)
CXX_FLAGS_DEBUG = -std=c++17 -O0 -g -fPIC -w

# Compiler flags for profiling build
CXX_FLAGS_PROFILE = -std=c++17 -O1 -g -fno-omit-frame-pointer -fno-inline-small-functions -fPIC -w

# Default to release build
CXX_FLAGS = $(CXX_FLAGS_RELEASE)

# Include directories
INCLUDES = -I./tiktoken

# Libraries
LIBS = -lpcre2-8
TIKTOKEN_LIB = tiktoken/libtiktoken.a

# Source files
CPP_SOURCES = main.cpp

# Output executable
TARGET = main

# Default target (release build)
all: $(TARGET)

# Release build
release: $(TARGET)

# Debug build
debug: CXX_FLAGS = $(CXX_FLAGS_DEBUG)
debug: $(TARGET)

# Profile build
profile: CXX_FLAGS = $(CXX_FLAGS_PROFILE)
profile: $(TARGET)

# Build the tiktoken library first
$(TIKTOKEN_LIB):
	$(MAKE) -C tiktoken

# Build the C++ executable (depends on tiktoken library)
$(TARGET): $(CPP_SOURCES) $(TIKTOKEN_LIB)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -o $(TARGET) $(CPP_SOURCES) $(TIKTOKEN_LIB) $(LIBS)

# Alternative: Build with separate compilation (if you need more complex builds)
$(TARGET)-alt: main.o $(TIKTOKEN_LIB)
	$(CXX) -o $(TARGET) main.o $(TIKTOKEN_LIB) $(LIBS)

main.o: $(CPP_SOURCES)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c -o main.o $(CPP_SOURCES)

# Clean build artifacts (including tiktoken)
clean:
	rm -f $(TARGET) *.o
	$(MAKE) -C tiktoken clean

# Clean only main project
clean-main:
	rm -f $(TARGET) *.o

# Clean only tiktoken
clean-tiktoken:
	$(MAKE) -C tiktoken clean

# Test the executable
test: $(TARGET)
	./$(TARGET)

# Debug with GDB
gdb: debug
	gdb ./$(TARGET)

.PHONY: all release debug profile clean clean-main clean-tiktoken test gdb