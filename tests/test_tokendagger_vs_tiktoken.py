#!/usr/bin/env python3
"""
Test TokenDagger against TikToken for correctness comparison.

Usage:
# Test with Llama tokenizer (default)
python tests/test_tokendagger_vs_tiktoken.py

# Test with Mistral tokenizer
python tests/test_tokendagger_vs_tiktoken.py --tokenizer mistral
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import time
import base64
from pathlib import Path
from typing import List, Dict, Any, Tuple

import tiktoken

# Add tokendagger to path
# sys.path.insert(0, str(Path(__file__).parent.parent / "tokendagger"))

try:
    from tokendagger import wrapper as tokendagger
except ImportError as e:
    print(f"Failed to import tokendagger: {e}")
    print("Make sure to build the Python extension with 'make python'")
    sys.exit(1)


class TokenDaggerVsTikTokenTest:
    """Test suite comparing TokenDagger and TikToken tokenization."""
    
    def __init__(self, tokenizer_type: str = "llama"):
        self.src_dir = Path(__file__).parent.parent / "src"
        self.test_dir = Path(__file__).parent
        self.tokenizer_type = tokenizer_type.lower()
        self.errors = []
        
        # Validate tokenizer type
        if self.tokenizer_type not in ["llama", "mistral"]:
            raise ValueError(f"Invalid tokenizer type: {tokenizer_type}. Must be 'llama' or 'mistral'")
        
    def load_llama_config(self) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """Load Llama 4 configuration from the codebase."""
        # Hard-coded pattern from main.cpp
        pattern = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        
        # Load vocabulary from tokenizer.model
        vocab = self.load_bpe_vocab()
        
        # Load special tokens from tokenizer_config.json
        special_tokens = self.load_special_tokens()
        
        return pattern, vocab, special_tokens
    
    def load_mistral_config(self) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]: 
        """
        Load Mistral's Tekken 7 configuration from the codebase.
        
        Odd notes about Mistral and Tekken configs.
        tekken.json includes two notable fields: 
            - default_vocab_size: the size of the vocabulary
            - default_num_special_tokens: the number of special tokens
        So while the full vocab is about size 150k, the default vocab is ~131k.
        Also, the special tokens are intended to be the first 1k ranks.
        
        So the actual used vocab is the first 130k ranks. Which are then offset by +1k to account for the special tokens.
        
        Then, the actual special tokens are not tokenized by the internal tokenizer (Tiktoken or TokenDagger). But handled manually by the wrapper class.
        
        This loader handles the offsets/truncation. And does not load in the special tokens, for a more fair benchmark between the two tokenizers.
        """
        config_file = self.test_dir / "configs" / "mistral3.2" / "tekken.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Tekken config file not found: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Extract pattern from config
        pattern = config["config"]["pattern"]
        
        # Extract vocabulary
        vocab = []
        max_vocab = config["config"]["default_vocab_size"] - config["config"]["default_num_special_tokens"]
        for i in range(max_vocab):
            vocab.append({
                "rank": i + config["config"]["default_num_special_tokens"],
                "token_bytes": list(base64.b64decode(config["vocab"][i]["token_bytes"])),  # Convert bytes to list of ints
                "token_string": config["vocab"][i].get("token_str", "") or ""  # Ensure it's always a string, never None
            })
        
        # Extract special tokens
        special_tokens = {}
        # oh wait, we don't need special tokens for the benchmark.
        # for item in config["special_tokens"]:
        #     special_tokens[item["token_str"]] = item["rank"]
        
        print(f"Loaded {len(vocab)} vocabulary items and {len(special_tokens)} special tokens from tekken.json")
        return pattern, vocab, special_tokens
    
    def load_bpe_vocab(self) -> List[Dict[str, Any]]:
        """Load vocabulary from tokenizer.model file."""
        
        model_file = self.src_dir / "tokenizer.model"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        vocab = []
        with open(model_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    base64_token = parts[0]
                    rank = int(parts[1])
                    
                    try:
                        # Decode base64 to bytes
                        token_bytes = list(base64.b64decode(base64_token))
                        vocab.append({
                            "rank": rank,
                            "token_bytes": token_bytes,
                            "token_string": ""  # Will be empty for BPE tokens
                        })
                    except Exception as e:
                        print(f"Warning: Failed to decode token {base64_token}: {e}")
                        continue
        
        print(f"Loaded {len(vocab)} vocabulary items from tokenizer.model")
        return vocab
    
    def load_special_tokens(self) -> Dict[str, int]:
        """Load special tokens from tokenizer_config.json."""
        config_file = self.src_dir / "tokenizer_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        special_tokens = {}
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
            # Extract special tokens from added_tokens_decoder
            if "added_tokens_decoder" in config:
                for token_id, token_info in config["added_tokens_decoder"].items():
                    special_tokens[token_info["content"]] = int(token_id)
        
        print(f"Loaded {len(special_tokens)} special tokens from tokenizer_config.json")
        return special_tokens
    
    def setup_tokenizers(self):
        """Initialize both tokenizers."""
        try:
            # Load configuration based on tokenizer type
            if self.tokenizer_type == "llama":
                print("Using Llama 4 configuration...")
                pattern, vocab, special_tokens = self.load_llama_config()
            elif self.tokenizer_type == "mistral":
                print("Using Mistral Tekken 7 configuration...")
                pattern, vocab, special_tokens = self.load_mistral_config()
            else:
                raise ValueError(f"Unsupported tokenizer type: {self.tokenizer_type}")
            
            # Convert TokenDagger format to TikToken format
            mergeable_ranks = {}
            for item in vocab:
                if isinstance(item["token_bytes"], list):
                    token_bytes = bytes(item["token_bytes"])
                else:
                    token_bytes = item["token_bytes"]
                mergeable_ranks[token_bytes] = item["rank"]
            
            # Add special tokens to mergeable_ranks
            for token_str, rank in special_tokens.items():
                mergeable_ranks[token_str.encode('utf-8')] = rank
            
            tokenizer_name = f"{self.tokenizer_type}_test"
            
            # Initialize TokenDagger using TikToken-compatible API
            self.tokendagger_tokenizer = tokendagger.Encoding(
                name=tokenizer_name,
                pat_str=pattern,
                mergeable_ranks=mergeable_ranks,
                special_tokens=special_tokens
            )
            
            # Initialize TikToken with the same configuration
            self.tiktoken_tokenizer = tiktoken.Encoding(
                name=tokenizer_name,
                pat_str=pattern,
                mergeable_ranks=mergeable_ranks,
                special_tokens=special_tokens
            )
            
            print(f"✓ TokenDagger tokenizer initialized with {self.tokenizer_type.title()} config")
            print(f"✓ TikToken tokenizer initialized with {self.tokenizer_type.title()} config")
            
        except Exception as e:
            print(f"Failed to setup tokenizers: {e}")
            raise
    
    def load_test_texts(self) -> List[str]:
        """Load test texts from input directory."""
        test_texts = []
        
        # Load from input files
        input_dir = self.test_dir / "input"
        if input_dir.exists():
            for txt_file in input_dir.glob("*.txt"):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            test_texts.append(content)
                except Exception as e:
                    print(f"Warning: Failed to load {txt_file}: {e}")
        
        # Add some basic test cases
        basic_tests = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "This is a test of the tokenization system.",
            "Special tokens: <|begin_of_text|>Hello<|end_of_text|>",
            "Unicode test: 你好世界 🌍",
            "Numbers and symbols: 123 456.789 @#$%",
            "",  # Empty string
            " ",  # Whitespace
            "\n\t",  # Special whitespace
        ]
        
        test_texts.extend(basic_tests)
        return test_texts
    
    def compare_encoding(self, text: str) -> Dict[str, Any]:
        """Compare encoding results between TokenDagger and TikToken."""
        result = {
            "text": text,
            "tokendagger_tokens": None,
            "tiktoken_tokens": None,
            "tokendagger_error": None,
            "tiktoken_error": None,
            "match": False,
            "tokendagger_time": 0,
            "tiktoken_time": 0,
        }
        
        # Test TokenDagger
        try:
            start_time = time.time()
            result["tokendagger_tokens"] = self.tokendagger_tokenizer.encode_ordinary(text)
            result["tokendagger_time"] = time.time() - start_time
        except Exception as e:
            result["tokendagger_error"] = str(e)
        
        # Test TikToken
        try:
            start_time = time.time()
            result["tiktoken_tokens"] = self.tiktoken_tokenizer.encode_ordinary(text)
            result["tiktoken_time"] = time.time() - start_time
        except Exception as e:
            result["tiktoken_error"] = str(e)
        
        # Compare results
        if (result["tokendagger_tokens"] is not None and 
            result["tiktoken_tokens"] is not None):
            result["match"] = result["tokendagger_tokens"] == result["tiktoken_tokens"]
        
        return result
    
    def compare_decoding(self, tokens: List[int]) -> Dict[str, Any]:
        """Compare decoding results between TokenDagger and TikToken."""
        result = {
            "tokens": tokens,
            "tokendagger_text": None,
            "tiktoken_text": None,
            "tokendagger_error": None,
            "tiktoken_error": None,
            "match": False,
        }
        
        # Test TokenDagger
        try:
            result["tokendagger_text"] = self.tokendagger_tokenizer.decode(tokens)
        except Exception as e:
            result["tokendagger_error"] = str(e)
        
        # Test TikToken
        try:
            result["tiktoken_text"] = self.tiktoken_tokenizer.decode(tokens)
        except Exception as e:
            result["tiktoken_error"] = str(e)
        
        # Compare results
        if (result["tokendagger_text"] is not None and 
            result["tiktoken_text"] is not None):
            result["match"] = result["tokendagger_text"] == result["tiktoken_text"]
        
        return result
    
    def run_encoding_tests(self) -> List[Dict[str, Any]]:
        """Run encoding comparison tests."""
        print("\n" + "="*60)
        print("ENCODING TESTS")
        print("="*60)
        
        test_texts = self.load_test_texts()
        results = []
        
        for i, text in enumerate(test_texts):
            print(f"\nTest {i+1}/{len(test_texts)}: {repr(text[:50])}")
            
            result = self.compare_encoding(text)
            results.append(result)
            
            if result["match"]:
                print("✓ MATCH")
            else:
                print("✗ MISMATCH")
                if result["tokendagger_error"]:
                    print(f"  TokenDagger error: {result['tokendagger_error']}")
                if result["tiktoken_error"]:
                    print(f"  TikToken error: {result['tiktoken_error']}")
                if result["tokendagger_tokens"] and result["tiktoken_tokens"]:
                    print(f"  TokenDagger: {result['tokendagger_tokens'][:10]}...")
                    print(f"  TikToken:    {result['tiktoken_tokens'][:10]}...")
                
                self.errors.append(f"Encoding mismatch for: {repr(text[:50])}")
        
        return results
    
    def run_decoding_tests(self) -> List[Dict[str, Any]]:
        """Run decoding comparison tests."""
        print("\n" + "="*60)
        print("DECODING TESTS")
        print("="*60)
        
        # Test with token sequences appropriate for the tokenizer type
        if self.tokenizer_type == "mistral":
            # Mistral tokens start at 1000 due to special token offset
            test_token_sequences = [
                [1000, 1001, 1002],
                [1100, 1200, 1300],
                [2000, 3000, 4000],
                list(range(1000, 1010)),
                list(range(5000, 5010)),
            ]
        else:
            # Llama tokens start from lower values
            test_token_sequences = [
                [1, 2, 3],
                [100, 200, 300],
                [1000, 2000, 3000],
                list(range(10)),
                list(range(100, 110)),
            ]
        
        results = []
        
        for i, tokens in enumerate(test_token_sequences):
            print(f"\nTest {i+1}/{len(test_token_sequences)}: {tokens}")
            
            result = self.compare_decoding(tokens)
            results.append(result)
            
            if result["match"]:
                print("✓ MATCH")
            else:
                print("✗ MISMATCH")
                if result["tokendagger_error"]:
                    print(f"  TokenDagger error: {result['tokendagger_error']}")
                if result["tiktoken_error"]:
                    print(f"  TikToken error: {result['tiktoken_error']}")
                if result["tokendagger_text"] and result["tiktoken_text"]:
                    print(f"  TokenDagger: {repr(result['tokendagger_text'][:50])}")
                    print(f"  TikToken:    {repr(result['tiktoken_text'][:50])}")
                
                self.errors.append(f"Decoding mismatch for: {tokens}")
        
        return results
    
    def run_roundtrip_tests(self):
        """Test encode->decode roundtrip consistency."""
        print("\n" + "="*60)
        print("ROUNDTRIP TESTS")
        print("="*60)
        
        test_texts = [
            "Hello, world!",
            "The quick brown fox jumps over the lazy dog.",
            "Unicode: 你好 🌍",
        ]
        
        for i, original_text in enumerate(test_texts):
            print(f"\nRoundtrip test {i+1}: {repr(original_text)}")
            
            # TokenDagger roundtrip
            try:
                td_tokens = self.tokendagger_tokenizer.encode_ordinary(original_text)
                td_decoded = self.tokendagger_tokenizer.decode(td_tokens)
                td_match = original_text == td_decoded
                print(f"  TokenDagger: {td_match} ({'✓' if td_match else '✗'})")
                if not td_match:
                    print(f"    Original: {repr(original_text)}")
                    print(f"    Decoded:  {repr(td_decoded)}")
            except Exception as e:
                print(f"  TokenDagger: ERROR - {e}")
            
            # TikToken roundtrip
            try:
                tt_tokens = self.tiktoken_tokenizer.encode_ordinary(original_text)
                tt_decoded = self.tiktoken_tokenizer.decode(tt_tokens)
                tt_match = original_text == tt_decoded
                print(f"  TikToken:    {tt_match} ({'✓' if tt_match else '✗'})")
                if not tt_match:
                    print(f"    Original: {repr(original_text)}")
                    print(f"    Decoded:  {repr(tt_decoded)}")
            except Exception as e:
                print(f"  TikToken: ERROR - {e}")
    
    def print_summary(self, encoding_results: List[Dict], decoding_results: List[Dict]):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        # Encoding summary
        encoding_matches = sum(1 for r in encoding_results if r["match"])
        encoding_total = len(encoding_results)
        print(f"Encoding tests: {encoding_matches}/{encoding_total} matches")
        
        # Decoding summary
        decoding_matches = sum(1 for r in decoding_results if r["match"])
        decoding_total = len(decoding_results)
        print(f"Decoding tests: {decoding_matches}/{decoding_total} matches")
        
        # Performance comparison
        td_times = [r["tokendagger_time"] for r in encoding_results if r["tokendagger_time"] > 0]
        tt_times = [r["tiktoken_time"] for r in encoding_results if r["tiktoken_time"] > 0]
        
        if td_times and tt_times:
            avg_td_time = sum(td_times) / len(td_times)
            avg_tt_time = sum(tt_times) / len(tt_times)
            print(f"\nPerformance (avg encoding time):")
            print(f"  TokenDagger: {avg_td_time:.6f}s")
            print(f"  TikToken:    {avg_tt_time:.6f}s")
            print(f"  Ratio:       {avg_td_time/avg_tt_time:.2f}x")
        
        # Errors
        if self.errors:
            print(f"\nErrors found: {len(self.errors)}")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more")
        else:
            print("\n🎉 All tests passed!")
    
    def run_all_tests(self):
        """Run all test suites."""
        print("TokenDagger vs TikToken Correctness Test")
        print(f"Using {self.tokenizer_type.title()} configuration")
        print("="*60)
        
        try:
            self.setup_tokenizers()
            
            encoding_results = self.run_encoding_tests()
            decoding_results = self.run_decoding_tests()
            self.run_roundtrip_tests()
            
            self.print_summary(encoding_results, decoding_results)
            
        except Exception as e:
            print(f"\n✗ Test suite failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return len(self.errors) == 0


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TokenDagger vs TikToken Correctness Test")
    parser.add_argument("--tokenizer", choices=["llama", "mistral"], default="llama", 
                       help="Tokenizer configuration to use (default: llama)")
    
    args = parser.parse_args()
    
    test_suite = TokenDaggerVsTikTokenTest(tokenizer_type=args.tokenizer)
    success = test_suite.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()