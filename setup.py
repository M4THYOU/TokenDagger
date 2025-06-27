import os
import subprocess
import sys
from pathlib import Path

from pybind11 import get_cmake_dir
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import Extension, setup

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "tokendagger._tokendagger_core",
        [
            "src/py_binding.cpp",
        ],
        include_dirs=[
            "src/tiktoken",
            "src",
            "extern/pybind11/include",
        ],
        libraries=["pcre2-8"],
        library_dirs=[],
        extra_objects=["src/tiktoken/libtiktoken.a"],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"dev"')],
    ),
]


class CustomBuildExt(build_ext):
    """Custom build extension to build tiktoken library first."""
    
    def build_extensions(self):
        # Build the tiktoken library first
        tiktoken_lib_path = Path("src/tiktoken/libtiktoken.a")
        
        print("Building tiktoken library...")
        try:
            result = subprocess.run(
                ["make", "-C", "src/tiktoken"], 
                check=True, 
                capture_output=True, 
                text=True
            )
            print("✓ tiktoken library built successfully")
        except subprocess.CalledProcessError as e:
            print(f"Failed to build tiktoken library: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            sys.exit(1)
        except FileNotFoundError:
            print("make command not found. Please install build tools.")
            sys.exit(1)
        
        # Check if library was created
        if not tiktoken_lib_path.exists():
            print(f"tiktoken library not found at {tiktoken_lib_path}")
            sys.exit(1)
        
        # Then build the Python extension
        super().build_extensions()


# Read the version from __init__.py
def get_version():
    init_file = Path("tokendagger/__init__.py")
    if init_file.exists():
        with open(init_file) as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"\'')
    return "0.1.0"


setup(
    name="tokendagger",
    version=get_version(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": CustomBuildExt},
    zip_safe=False,
    python_requires=">=3.8",
) 