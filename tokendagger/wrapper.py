"""
TokenDagger: A high-level wrapper around the tokendagger C++ library.

This module provides a tiktoken-compatible interface for the tokendagger tokenizer.
"""

from __future__ import annotations

import functools
from typing import AbstractSet, Collection, Literal, Sequence
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor

try:
    from . import _tokendagger_core as tokendagger
except ImportError:
    raise ImportError(
        "tokendagger C++ extension not found. Make sure to build it with 'make python'"
    )


class TokenDaggerError(Exception):
    """Base exception for TokenDagger errors."""
    pass


class Tokenizer:
    """High-level interface for the tokendagger tokenizer.
    
    This class provides a tiktoken-compatible API around the tokendagger.CoreBPE class.
    """
    
    def __init__(
        self,
        name: str,
        *,
        pattern: str,
        vocab: list[dict] | None = None,
        special_tokens: dict[str, int] | None = None,
        vocab_file: str | Path | None = None,
        special_tokens_file: str | Path | None = None,
    ):
        """Initialize the tokenizer.
        
        Args:
            name: Name of the tokenizer
            pattern: Regex pattern for text splitting
            vocab: List of vocabulary items as dicts with 'rank', 'token_bytes', 'token_string'
            special_tokens: Dict mapping special token strings to their IDs
            vocab_file: Path to vocabulary file (JSON format)
            special_tokens_file: Path to special tokens file (JSON format)
        """
        self.name = name
        self.pattern = pattern
        
        # Load vocabulary
        if vocab_file:
            vocab = self._load_vocab_file(vocab_file)
        elif vocab is None:
            raise ValueError("Either 'vocab' or 'vocab_file' must be provided")
            
        # Load special tokens
        if special_tokens_file:
            special_tokens = self._load_special_tokens_file(special_tokens_file)
        elif special_tokens is None:
            special_tokens = {}
            
        # Convert to VocabItem objects
        vocab_items = []
        for item in vocab:
            vocab_item = tokendagger.VocabItem()
            vocab_item.rank = item['rank']
            vocab_item.token_bytes = item['token_bytes']
            vocab_item.token_string = item.get('token_string', '')
            vocab_items.append(vocab_item)
            
        special_vocab_items = []
        for token_str, rank in special_tokens.items():
            special_item = tokendagger.VocabItem()
            special_item.rank = rank
            special_item.token_bytes = list(token_str.encode('utf-8'))
            special_item.token_string = token_str
            special_vocab_items.append(special_item)
            
        # Store for later use
        self._special_tokens = special_tokens
        self.max_token_value = max(
            max(item['rank'] for item in vocab),
            max(special_tokens.values()) if special_tokens else 0
        )
        
        # Initialize the core tokenizer
        try:
            self._core_bpe = tokendagger.CoreBPE(pattern, vocab_items, special_vocab_items)
        except Exception as e:
            raise TokenDaggerError(f"Failed to initialize CoreBPE: {e}")
    
    def _load_vocab_file(self, vocab_file: str | Path) -> list[dict]:
        """Load vocabulary from JSON file."""
        path = Path(vocab_file)
        if not path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_special_tokens_file(self, special_tokens_file: str | Path) -> dict[str, int]:
        """Load special tokens from JSON file."""
        path = Path(special_tokens_file)
        if not path.exists():
            raise FileNotFoundError(f"Special tokens file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def __repr__(self) -> str:
        return f"<TokenDagger {self.name!r}>"
    
    # ====================
    # Encoding methods
    # ====================
    
    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text using ordinary tokens only (no special tokens).
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        try:
            return self._core_bpe.encode_ordinary(text)
        except Exception as e:
            raise TokenDaggerError(f"Encoding failed: {e}")
    
    def encode(
        self,
        text: str,
        *,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),
        disallowed_special: Literal["all"] | Collection[str] = set(),
    ) -> list[int]:
        """Encode text with special token handling.
        
        Args:
            text: Text to encode
            allowed_special: Special tokens that are allowed in the text
            disallowed_special: Special tokens that should raise an error if found
            
        Returns:
            List of token IDs
        """
        if allowed_special == "all":
            allowed_special = set(self._special_tokens.keys())
        if disallowed_special == "all":
            disallowed_special = set(self._special_tokens.keys()) - set(allowed_special)
            
        # Check for disallowed special tokens
        if disallowed_special:
            for token in disallowed_special:
                if token in text:
                    raise ValueError(
                        f"Encountered disallowed special token {token!r}. "
                        f"Pass it to allowed_special to encode it as a special token."
                    )
        
        try:
            # Convert allowed_special to the format expected by C++
            allowed_set = set(allowed_special) if allowed_special != "all" else set(self._special_tokens.keys())
            tokens, _ = self._core_bpe.encode(text, allowed_set)
            return tokens
        except Exception as e:
            raise TokenDaggerError(f"Encoding failed: {e}")
    
    def encode_with_special_tokens(self, text: str) -> list[int]:
        """Encode text including all special tokens.
        
        Args:
            text: Text to encode
            
        Returns:
            List of token IDs
        """
        try:
            return self._core_bpe.encode_with_special_tokens(text)
        except Exception as e:
            raise TokenDaggerError(f"Encoding failed: {e}")
    
    def encode_batch(
        self,
        text: Sequence[str],
        *,
        num_threads: int = 8,
        allowed_special: Literal["all"] | AbstractSet[str] = set(),
        disallowed_special: Literal["all"] | Collection[str] = set(),
    ) -> list[list[int]]:
        """Encode multiple texts in parallel.
        
        Args:
            text: Sequence of texts to encode
            num_threads: Number of threads to use for parallel processing
            allowed_special: Special tokens that are allowed in the text
            disallowed_special: Special tokens that should raise an error if found
            
        Returns:
            List of lists of token IDs
        """
        encoder = functools.partial(
            self.encode, allowed_special=allowed_special, disallowed_special=disallowed_special
        )
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(encoder, text))
        
    def decode_batch(
        self,
        tokens: Sequence[Sequence[int]],
        *,
        num_threads: int = 8,
        errors: str = "replace",
    ) -> list[str]:
        """Decode multiple token sequences in parallel.
        
        Args:
            tokens: Sequence of token sequences to decode
            num_threads: Number of threads to use for parallel processing
            errors: How to handle decode errors ('replace', 'ignore', 'strict')
            
        Returns:
            List of decoded strings
        """
        decoder = functools.partial(self.decode, errors=errors)
        with ThreadPoolExecutor(num_threads) as e:
            return list(e.map(decoder, tokens))
    
    # ====================
    # Decoding methods
    # ====================
    
    def decode_bytes(self, tokens: Sequence[int]) -> bytes:
        """Decode tokens to bytes.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Decoded bytes
        """
        try:
            return bytes(self._core_bpe.decode_bytes(list(tokens)))
        except Exception as e:
            raise TokenDaggerError(f"Decoding failed: {e}")
    
    def decode(self, tokens: Sequence[int], errors: str = "replace") -> str:
        """Decode tokens to string.
        
        Args:
            tokens: List of token IDs
            errors: How to handle decode errors ('replace', 'ignore', 'strict')
            
        Returns:
            Decoded string
        """
        try:
            decoded_bytes = self.decode_bytes(tokens)
            return decoded_bytes.decode("utf-8", errors=errors)
        except Exception as e:
            raise TokenDaggerError(f"Decoding failed: {e}")
    
    # ====================
    # Utility methods
    # ====================
    
    def special_tokens(self) -> list[str]:
        """Get list of special tokens.
        
        Returns:
            List of special token strings
        """
        try:
            return self._core_bpe.special_tokens()
        except Exception as e:
            raise TokenDaggerError(f"Failed to get special tokens: {e}")
    
    @property
    def special_tokens_set(self) -> set[str]:
        """Set of special token strings."""
        return set(self._special_tokens.keys())
    
    @property
    def n_vocab(self) -> int:
        """Total vocabulary size."""
        return self.max_token_value + 1
    
    def is_special_token(self, token: int) -> bool:
        """Check if a token ID corresponds to a special token.
        
        Args:
            token: Token ID to check
            
        Returns:
            True if the token is a special token
        """
        return token in self._special_tokens.values()


# ====================
# Convenience functions
# ====================

def load_tokenizer(
    name: str,
    vocab_file: str | Path,
    pattern: str,
    special_tokens_file: str | Path | None = None,
) -> Tokenizer:
    """Load a tokenizer from files.
    
    Args:
        name: Name of the tokenizer
        vocab_file: Path to vocabulary file
        pattern: Regex pattern for text splitting
        special_tokens_file: Optional path to special tokens file
        
    Returns:
        Initialized Tokenizer instance
    """
    return Tokenizer(
        name=name,
        pattern=pattern,
        vocab_file=vocab_file,
        special_tokens_file=special_tokens_file,
    )


def create_tokenizer(
    name: str,
    pattern: str,
    vocab: list[dict],
    special_tokens: dict[str, int] | None = None,
) -> Tokenizer:
    """Create a tokenizer from in-memory data.
    
    Args:
        name: Name of the tokenizer
        pattern: Regex pattern for text splitting
        vocab: Vocabulary items
        special_tokens: Special tokens mapping
        
    Returns:
        Initialized Tokenizer instance
    """
    return Tokenizer(
        name=name,
        pattern=pattern,
        vocab=vocab,
        special_tokens=special_tokens,
    ) 