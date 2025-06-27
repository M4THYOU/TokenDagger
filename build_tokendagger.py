#!/usr/bin/env python3
"""
Build script for TokenDagger that handles the C++ compilation.
"""

import subprocess
import sys
import os
from pathlib import Path

def build_tiktoken_lib():
    """Build the tiktoken static library."""
    try:
        subprocess.check_call(["make", "-C", "src/tiktoken"], 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE)
        print("✓ tiktoken library built successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to build tiktoken library: {e}")
        return False
    except FileNotFoundError:
        print("✗ make command not found. Please install build tools.")
        return False

def main():
    """Main build function."""
    print("Building TokenDagger...")
    
    # Check if we're in the right directory
    if not Path("src/tiktoken").exists():
        print("✗ tiktoken source directory not found")
        sys.exit(1)
    
    # Build tiktoken library
    if not build_tiktoken_lib():
        sys.exit(1)
    
    print("✓ TokenDagger build completed")

if __name__ == "__main__":
    main() 