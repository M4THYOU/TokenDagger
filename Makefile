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
INCLUDES = -I./src/tiktoken -I./src

# Libraries
LIBS = -lpcre2-8
TIKTOKEN_LIB = src/tiktoken/libtiktoken.a

# Python binding settings
PYTHON_CONFIG = python3-config
PYTHON_INCLUDES = $(shell $(PYTHON_CONFIG) --includes)
PYTHON_LIBS = $(shell $(PYTHON_CONFIG) --ldflags)

# Use local pybind11 from extern directory
PYBIND11_INCLUDES = -I./extern/pybind11/include

# Source files
CPP_SOURCES = src/main.cpp
PYTHON_SOURCES = src/py_binding.cpp

# Output targets
TARGET = main
PYTHON_MODULE_NAME = _tokendagger_core$(shell python3-config --extension-suffix)
PYTHON_MODULE_PATH = tokendagger/$(PYTHON_MODULE_NAME)

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

# Python module build
python: $(PYTHON_MODULE_PATH)

# Create tokendagger directory if it doesn't exist
tokendagger:
	mkdir -p tokendagger

# Build the tiktoken library first
$(TIKTOKEN_LIB):
	$(MAKE) -C src/tiktoken

# Build the C++ executable (depends on tiktoken library)
$(TARGET): $(CPP_SOURCES) $(TIKTOKEN_LIB)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -o $(TARGET) $(CPP_SOURCES) $(TIKTOKEN_LIB) $(LIBS)

# Build the Python module (shared library) and place it in the package directory
$(PYTHON_MODULE_PATH): $(PYTHON_SOURCES) $(TIKTOKEN_LIB) tokendagger
	$(CXX) $(CXX_FLAGS) $(INCLUDES) $(PYBIND11_INCLUDES) $(PYTHON_INCLUDES) \
		-shared -o $(PYTHON_MODULE_PATH) $(PYTHON_SOURCES) $(TIKTOKEN_LIB) $(LIBS)

# Alternative: Build with separate compilation (if you need more complex builds)
$(TARGET)-alt: main.o $(TIKTOKEN_LIB)
	$(CXX) -o $(TARGET) main.o $(TIKTOKEN_LIB) $(LIBS)

main.o: $(CPP_SOURCES)
	$(CXX) $(CXX_FLAGS) $(INCLUDES) -c -o main.o $(CPP_SOURCES)

# Clean build artifacts (including tiktoken)
clean:
	rm -f $(TARGET) tokendagger/*.so *.o
	$(MAKE) -C src/tiktoken clean

# Clean only main project
clean-main:
	rm -f $(TARGET) *.o

# Clean only tiktoken
clean-tiktoken:
	$(MAKE) -C src/tiktoken clean

# Clean only Python module
clean-python:
	rm -f tokendagger/*.so

# Test the executable
test: $(TARGET)
	./$(TARGET)

# Test the Python module
test-python: python
	python3 -c "import tokendagger; print('TokenDagger package loaded successfully')"

# Test the high-level wrapper
test-wrapper: python
	python3 -c "from tokendagger import Tokenizer; print('TokenDagger wrapper loaded successfully')"

# Debug with GDB
gdb: debug
	gdb ./$(TARGET)

.PHONY: all release debug profile python tokendagger clean clean-main clean-tiktoken clean-python test test-python test-wrapper gdb