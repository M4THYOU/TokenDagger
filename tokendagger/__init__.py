"""
TokenDagger: Fast tokenization library with Python bindings.
"""

from .wrapper import (
    Tokenizer,
    TokenDaggerError,
    load_tokenizer,
    create_tokenizer,
    Encoding,
)

try:
    from . import _tokendagger_core as core
except ImportError:
    core = None

__version__ = "0.1.1"
__all__ = [
    "Tokenizer",
    "TokenDaggerError", 
    "load_tokenizer",
    "create_tokenizer",
    "Encoding",
    "core",
] 