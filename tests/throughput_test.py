#!/usr/bin/env python3
"""
Multithreaded throughput comparison between TokenDagger and TikToken.
Performance measured on 1GB of text across different thread counts.

Usage:
# Full throughput test with Llama tokenizer (default)
python tests/throughput_test.py

# Full throughput test with Mistral tokenizer
python tests/throughput_test.py --tokenizer mistral

# Quick test with fewer iterations
python tests/throughput_test.py --quick

# Custom settings
python tests/throughput_test.py --threads 1,2,4,8,16 --text-size 512 --iterations 5
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import time
import base64
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import tiktoken

# Import Hugging Face tokenizers
from transformers import AutoTokenizer

# Import matplotlib for SVG generation
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
HAS_MATPLOTLIB = True

try:
    from tokendagger import wrapper as tokendagger
except ImportError as e:
    print(f"Failed to import tokendagger: {e}")
    print("Make sure to build the Python extension with 'make python'")
    sys.exit(1)


@dataclass
class ThroughputResult:
    """Results from a throughput test."""
    thread_count: int
    tokenizer_name: str
    total_text_size_mb: float
    total_tokens: int
    total_time_seconds: float
    throughput_mb_per_sec: float
    throughput_tokens_per_sec: float
    avg_latency_ms: float


class ThroughputBenchmark:
    """Multithreaded throughput benchmark suite."""
    
    def __init__(self, 
                 tokenizer_type: str = "llama",
                 thread_counts: List[int] = [1, 2, 4, 8, 16, 32],
                 text_size_mb: int = 1024,  # 1GB by default
                 iterations_per_thread: int = 10):
        self.src_dir = Path(__file__).parent.parent / "src"
        self.test_dir = Path(__file__).parent
        self.tokenizer_type = tokenizer_type.lower()
        self.thread_counts = thread_counts
        self.text_size_mb = text_size_mb
        self.iterations_per_thread = iterations_per_thread
        self.results: List[ThroughputResult] = []
        
        # Validate tokenizer type
        if self.tokenizer_type not in ["llama", "mistral"]:
            raise ValueError(f"Invalid tokenizer type: {tokenizer_type}. Must be 'llama' or 'mistral'")
        
        print(f"Throughput benchmark configuration:")
        print(f"  Tokenizer: {self.tokenizer_type}")
        print(f"  Thread counts: {self.thread_counts}")
        print(f"  Text size: {self.text_size_mb} MB")
        print(f"  Iterations per thread: {self.iterations_per_thread}")
    
    def load_llama_config(self) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """
        Load Llama 4 configuration from the codebase.
        https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
        """
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
        https://huggingface.co/mistralai/Ministral-8B-Instruct-2410/tree/main
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
                "token_bytes": list(base64.b64decode(config["vocab"][i]["token_bytes"])),
                "token_string": config["vocab"][i].get("token_str", "") or ""
            })
        
        # Extract special tokens (empty for benchmark)
        special_tokens = {}
        
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
                            "token_string": ""
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
        """Initialize both tokenizers with identical configuration."""
        print("Setting up tokenizers...")
        
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
        
        tokenizer_name = f"{self.tokenizer_type}_throughput_test"
        
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
        
        # Initialize Hugging Face Fast Tokenizer
        # Use the appropriate HF model based on tokenizer type
        if self.tokenizer_type == "llama":
            model_name = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
        elif self.tokenizer_type == "mistral":
            model_name = "mistralai/Ministral-8B-Instruct-2410"
        
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        print(f"✓ HF Fast Tokenizer initialized ({model_name})")
        
        print(f"✓ TokenDagger tokenizer initialized ({self.tokenizer_type})")
        print(f"✓ TikToken tokenizer initialized ({self.tokenizer_type})")
    
    def generate_test_text(self, size_mb: float) -> str:
        """Generate realistic test text of specified size in MB."""
        print(f"Generating {size_mb:.1f} MB of test text...")
        
        # Common words for realistic text generation
        common_words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", 
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
            "or", "an", "will", "my", "one", "all", "would", "there", "their",
            "what", "so", "up", "out", "if", "about", "who", "get", "which", "go",
            "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them",
            "see", "other", "than", "then", "now", "look", "only", "come", "its", "over",
            "think", "also", "back", "after", "use", "two", "how", "our", "work", "first",
            "well", "way", "even", "new", "want", "because", "any", "these", "give", "day",
            "most", "us", "is", "was", "are", "been", "has", "had", "were", "said", "each",
            "which", "their", "time", "will", "about", "if", "up", "out", "many", "then",
            "them", "these", "so", "some", "her", "would", "make", "like", "into", "him",
            "you", "could", "more", "go", "no", "way", "could", "my", "than", "first",
            "water", "been", "call", "who", "its", "now", "find", "long", "down", "day",
            "did", "get", "come", "made", "may", "part", "over", "new", "sound", "take",
            "only", "little", "work", "know", "place", "year", "live", "me", "back", "give",
            "most", "very", "after", "thing", "our", "just", "name", "good", "sentence",
            "man", "think", "say", "great", "where", "help", "through", "much", "before",
            "line", "right", "too", "mean", "old", "any", "same", "tell", "boy", "follow",
            "came", "want", "show", "also", "around", "form", "three", "small", "set", "put",
            "end", "why", "again", "turn", "here", "off", "went", "old", "number", "great",
            "tell", "men", "say", "small", "every", "found", "still", "between", "mane",
            "should", "home", "big", "give", "air", "line", "set", "own", "under", "read",
            "last", "never", "us", "left", "end", "along", "while", "might", "next", "sound",
            "below", "saw", "something", "thought", "both", "few", "those", "always", "looked",
            "show", "large", "often", "together", "asked", "house", "don't", "world", "going",
            "want", "school", "important", "until", "form", "food", "keep", "children", "feet",
            "land", "side", "without", "boy", "once", "animal", "life", "enough", "took", "four"
        ]
        
        # Calculate target size in bytes
        target_bytes = int(size_mb * 1024 * 1024)
        
        # Generate text chunks
        text_chunks = []
        current_size = 0
        
        # Generate paragraphs of varying lengths
        while current_size < target_bytes:
            # Create a paragraph with 50-200 words
            paragraph_length = random.randint(50, 200)
            paragraph_words = random.choices(common_words, k=paragraph_length)
            
            # Add some punctuation and capitalization
            paragraph = " ".join(paragraph_words)
            paragraph = paragraph.capitalize()
            
            # Add random punctuation
            for i in range(random.randint(2, 5)):
                pos = random.randint(10, len(paragraph) - 10)
                if paragraph[pos] == ' ':
                    paragraph = paragraph[:pos] + random.choice([',', '.', '!', '?']) + paragraph[pos:]
            
            # Add paragraph separators
            paragraph += "\n\n"
            
            text_chunks.append(paragraph)
            current_size += len(paragraph.encode('utf-8'))
            
            # Progress indicator for large files
            if len(text_chunks) % 100000 == 0:
                print(f"Generated {current_size / (1024*1024):.1f} MB...")
        
        full_text = "".join(text_chunks)
        
        # Trim to exact size if needed
        if len(full_text.encode('utf-8')) > target_bytes:
            # Binary search to find the right cutoff point
            left, right = 0, len(full_text)
            while left < right:
                mid = (left + right + 1) // 2
                if len(full_text[:mid].encode('utf-8')) <= target_bytes:
                    left = mid
                else:
                    right = mid - 1
            full_text = full_text[:left]
        
        actual_size_mb = len(full_text.encode('utf-8')) / (1024 * 1024)
        print(f"Generated {actual_size_mb:.2f} MB of text ({len(full_text):,} characters)")
        
        return full_text
    
    def benchmark_hf_throughput(self, tokenizer, tokenizer_name: str, thread_count: int, test_text: str) -> ThroughputResult:
        """Benchmark HF Fast Tokenizer throughput."""
        print(f"  Testing {tokenizer_name} with {thread_count} threads...")
        
        # Split text into chunks for processing
        chunk_size = len(test_text) // (thread_count * self.iterations_per_thread)
        text_chunks = []
        
        for i in range(thread_count * self.iterations_per_thread):
            start_idx = i * chunk_size
            if i == (thread_count * self.iterations_per_thread) - 1:
                # Last chunk gets remaining text
                chunk = test_text[start_idx:]
            else:
                end_idx = (i + 1) * chunk_size
                chunk = test_text[start_idx:end_idx]
            text_chunks.append(chunk)
        
        # Benchmark the tokenizer
        start_time = time.perf_counter()
        
        try:
            # HF Fast Tokenizer batch encoding
            batch_encoding = tokenizer(text_chunks, padding=False, truncation=False, return_tensors=None)
            token_results = batch_encoding['input_ids']
            success = True
        except Exception as e:
            print(f"    ERROR: {tokenizer_name} batch encode failed: {e}")
            return None
        
        end_time = time.perf_counter()
        
        if not success or not token_results:
            print(f"    ERROR: No successful tokenizations for {tokenizer_name}")
            return None
        
        # Calculate statistics
        total_time = end_time - start_time
        total_bytes = sum(len(chunk.encode('utf-8')) for chunk in text_chunks)
        total_tokens = sum(len(tokens) for tokens in token_results)
        total_mb = total_bytes / (1024 * 1024)
        
        throughput_mb_per_sec = total_mb / total_time
        throughput_tokens_per_sec = total_tokens / total_time
        avg_latency_ms = (total_time / len(text_chunks)) * 1000
        
        print(f"    {tokenizer_name}: {throughput_mb_per_sec:.2f} MB/s, {throughput_tokens_per_sec:.0f} tokens/s")
        
        return ThroughputResult(
            thread_count=thread_count,
            tokenizer_name=tokenizer_name,
            total_text_size_mb=total_mb,
            total_tokens=total_tokens,
            total_time_seconds=total_time,
            throughput_mb_per_sec=throughput_mb_per_sec,
            throughput_tokens_per_sec=throughput_tokens_per_sec,
            avg_latency_ms=avg_latency_ms
        )
    
    def benchmark_throughput(self, tokenizer, tokenizer_name: str, thread_count: int, test_text: str) -> ThroughputResult:
        """Benchmark tokenizer throughput with specified thread count."""
        print(f"  Testing {tokenizer_name} with {thread_count} threads...")
        
        # Split text into chunks for processing
        chunk_size = len(test_text) // (thread_count * self.iterations_per_thread)
        text_chunks = []
        
        for i in range(thread_count * self.iterations_per_thread):
            start_idx = i * chunk_size
            if i == (thread_count * self.iterations_per_thread) - 1:
                # Last chunk gets remaining text
                chunk = test_text[start_idx:]
            else:
                end_idx = (i + 1) * chunk_size
                chunk = test_text[start_idx:end_idx]
            text_chunks.append(chunk)
        
        # Benchmark the tokenizer
        start_time = time.perf_counter()
        
        try:
            token_results = tokenizer.encode_batch(text_chunks, num_threads=thread_count)
            success = True
        except Exception as e:
            print(f"    ERROR: {tokenizer_name} encode_batch failed: {e}")
            return None
        
        end_time = time.perf_counter()
        
        if not success or not token_results:
            print(f"    ERROR: No successful tokenizations for {tokenizer_name}")
            return None
        
        # Calculate statistics
        total_time = end_time - start_time
        total_bytes = sum(len(chunk.encode('utf-8')) for chunk in text_chunks)
        total_tokens = sum(len(tokens) for tokens in token_results)
        total_mb = total_bytes / (1024 * 1024)
        
        throughput_mb_per_sec = total_mb / total_time
        throughput_tokens_per_sec = total_tokens / total_time
        avg_latency_ms = (total_time / len(text_chunks)) * 1000
        
        print(f"    {tokenizer_name}: {throughput_mb_per_sec:.2f} MB/s, {throughput_tokens_per_sec:.0f} tokens/s")
        
        return ThroughputResult(
            thread_count=thread_count,
            tokenizer_name=tokenizer_name,
            total_text_size_mb=total_mb,
            total_tokens=total_tokens,
            total_time_seconds=total_time,
            throughput_mb_per_sec=throughput_mb_per_sec,
            throughput_tokens_per_sec=throughput_tokens_per_sec,
            avg_latency_ms=avg_latency_ms
        )
    
    def run_throughput_benchmarks(self):
        """Run throughput benchmarks across all thread counts."""
        print("="*80)
        print("MULTITHREADED THROUGHPUT BENCHMARK")
        print("="*80)
        print(f"Tokenizer: {self.tokenizer_type}")
        print(f"Text size: {self.text_size_mb} MB")
        print(f"Iterations per thread: {self.iterations_per_thread}")
        print()
        
        # Generate test text
        test_text = self.generate_test_text(self.text_size_mb)
        
        # Run benchmarks for each thread count
        for thread_count in self.thread_counts:
            print(f"\n--- TESTING WITH {thread_count} THREADS ---")
            
            # Test TokenDagger
            td_result = self.benchmark_throughput(
                self.tokendagger_tokenizer, 
                "TokenDagger", 
                thread_count, 
                test_text
            )
            if td_result:
                self.results.append(td_result)
            
            # Test TikToken
            tt_result = self.benchmark_throughput(
                self.tiktoken_tokenizer,
                "TikToken",
                thread_count,
                test_text
            )
            if tt_result:
                self.results.append(tt_result)
            
            # Test HF Fast Tokenizer
            hf_result = None
            if self.hf_tokenizer:
                hf_result = self.benchmark_hf_throughput(
                    self.hf_tokenizer,
                    "HF Fast Tokenizer",
                    thread_count,
                    test_text
                )
                if hf_result:
                    self.results.append(hf_result)
            
            # Compare results
            if td_result and tt_result:
                mb_speedup = td_result.throughput_mb_per_sec / tt_result.throughput_mb_per_sec
                token_speedup = td_result.throughput_tokens_per_sec / tt_result.throughput_tokens_per_sec
                print(f"  TokenDagger vs TikToken: {mb_speedup:.2f}x MB/s, {token_speedup:.2f}x tokens/s")
            
            if td_result and hf_result:
                mb_speedup = td_result.throughput_mb_per_sec / hf_result.throughput_mb_per_sec
                token_speedup = td_result.throughput_tokens_per_sec / hf_result.throughput_tokens_per_sec
                print(f"  TokenDagger vs HF Fast: {mb_speedup:.2f}x MB/s, {token_speedup:.2f}x tokens/s")
    
    def print_summary_report(self):
        """Print comprehensive throughput analysis."""
        if not self.results:
            print("No benchmark results to analyze!")
            return
        
        print("\n" + "="*80)
        print("THROUGHPUT ANALYSIS SUMMARY")
        print("="*80)
        
        # Group results by tokenizer
        td_results = [r for r in self.results if r.tokenizer_name == "TokenDagger"]
        tt_results = [r for r in self.results if r.tokenizer_name == "TikToken"]
        hf_results = [r for r in self.results if r.tokenizer_name == "HF Fast Tokenizer"]
        
        if not td_results or not tt_results:
            print("Incomplete results - need both TokenDagger and TikToken results")
            return
        
        print(f"\nTHROUGHPUT BY THREAD COUNT:")
        if hf_results:
            print(f"{'Threads':<8} {'TokenDagger':<15} {'TikToken':<15} {'HF Fast':<15} {'TD/TT':<10} {'TD/HF':<10}")
            print(f"{'='*8} {'='*15} {'='*15} {'='*15} {'='*10} {'='*10}")
        else:
            print(f"{'Threads':<8} {'TokenDagger':<15} {'TikToken':<15} {'Speedup':<10}")
            print(f"{'='*8} {'='*15} {'='*15} {'='*10}")
        
        for thread_count in self.thread_counts:
            td_result = next((r for r in td_results if r.thread_count == thread_count), None)
            tt_result = next((r for r in tt_results if r.thread_count == thread_count), None)
            hf_result = next((r for r in hf_results if r.thread_count == thread_count), None)
            
            if td_result and tt_result:
                if hf_results and hf_result:
                    speedup_tt = td_result.throughput_mb_per_sec / tt_result.throughput_mb_per_sec
                    speedup_hf = td_result.throughput_mb_per_sec / hf_result.throughput_mb_per_sec
                    print(f"{thread_count:<8} {td_result.throughput_mb_per_sec:<15.2f} "
                          f"{tt_result.throughput_mb_per_sec:<15.2f} {hf_result.throughput_mb_per_sec:<15.2f} "
                          f"{speedup_tt:<10.2f}x {speedup_hf:<10.2f}x")
                else:
                    speedup = td_result.throughput_mb_per_sec / tt_result.throughput_mb_per_sec
                    print(f"{thread_count:<8} {td_result.throughput_mb_per_sec:<15.2f} "
                          f"{tt_result.throughput_mb_per_sec:<15.2f} {speedup:<10.2f}x")
        
        print(f"\nTOKENS PER SECOND BY THREAD COUNT:")
        if hf_results:
            print(f"{'Threads':<8} {'TokenDagger':<15} {'TikToken':<15} {'HF Fast':<15} {'TD/TT':<10} {'TD/HF':<10}")
            print(f"{'='*8} {'='*15} {'='*15} {'='*15} {'='*10} {'='*10}")
        else:
            print(f"{'Threads':<8} {'TokenDagger':<15} {'TikToken':<15} {'Speedup':<10}")
            print(f"{'='*8} {'='*15} {'='*15} {'='*10}")
        
        for thread_count in self.thread_counts:
            td_result = next((r for r in td_results if r.thread_count == thread_count), None)
            tt_result = next((r for r in tt_results if r.thread_count == thread_count), None)
            hf_result = next((r for r in hf_results if r.thread_count == thread_count), None)
            
            if td_result and tt_result:
                if hf_results and hf_result:
                    speedup_tt = td_result.throughput_tokens_per_sec / tt_result.throughput_tokens_per_sec
                    speedup_hf = td_result.throughput_tokens_per_sec / hf_result.throughput_tokens_per_sec
                    print(f"{thread_count:<8} {td_result.throughput_tokens_per_sec:<15.0f} "
                          f"{tt_result.throughput_tokens_per_sec:<15.0f} {hf_result.throughput_tokens_per_sec:<15.0f} "
                          f"{speedup_tt:<10.2f}x {speedup_hf:<10.2f}x")
                else:
                    speedup = td_result.throughput_tokens_per_sec / tt_result.throughput_tokens_per_sec
                    print(f"{thread_count:<8} {td_result.throughput_tokens_per_sec:<15.0f} "
                          f"{tt_result.throughput_tokens_per_sec:<15.0f} {speedup:<10.2f}x")
        
        # Scaling analysis
        print(f"\nSCALING ANALYSIS:")
        print("TokenDagger scaling:")
        base_td = next((r for r in td_results if r.thread_count == 1), None)
        if base_td:
            for thread_count in self.thread_counts:
                td_result = next((r for r in td_results if r.thread_count == thread_count), None)
                if td_result:
                    scaling = td_result.throughput_mb_per_sec / base_td.throughput_mb_per_sec
                    efficiency = (scaling / thread_count) * 100
                    print(f"  {thread_count} threads: {scaling:.2f}x scaling ({efficiency:.1f}% efficiency)")
        
        print("TikToken scaling:")
        base_tt = next((r for r in tt_results if r.thread_count == 1), None)
        if base_tt:
            for thread_count in self.thread_counts:
                tt_result = next((r for r in tt_results if r.thread_count == thread_count), None)
                if tt_result:
                    scaling = tt_result.throughput_mb_per_sec / base_tt.throughput_mb_per_sec
                    efficiency = (scaling / thread_count) * 100
                    print(f"  {thread_count} threads: {scaling:.2f}x scaling ({efficiency:.1f}% efficiency)")
        
        # Best performance summary
        best_td = max(td_results, key=lambda r: r.throughput_mb_per_sec)
        best_tt = max(tt_results, key=lambda r: r.throughput_mb_per_sec)
        
        print(f"\nBEST PERFORMANCE:")
        print(f"TokenDagger: {best_td.throughput_mb_per_sec:.2f} MB/s ({best_td.throughput_tokens_per_sec:.0f} tokens/s) "
              f"with {best_td.thread_count} threads")
        print(f"TikToken:    {best_tt.throughput_mb_per_sec:.2f} MB/s ({best_tt.throughput_tokens_per_sec:.0f} tokens/s) "
              f"with {best_tt.thread_count} threads")
        
        if hf_results:
            best_hf = max(hf_results, key=lambda r: r.throughput_mb_per_sec)
            print(f"HF Fast:     {best_hf.throughput_mb_per_sec:.2f} MB/s ({best_hf.throughput_tokens_per_sec:.0f} tokens/s) "
                  f"with {best_hf.thread_count} threads")
        
        overall_speedup = best_td.throughput_mb_per_sec / best_tt.throughput_mb_per_sec
        print(f"\nOVERALL BEST SPEEDUP vs TikToken: {overall_speedup:.2f}x")
        
        if hf_results:
            best_hf = max(hf_results, key=lambda r: r.throughput_mb_per_sec)
            hf_speedup = best_td.throughput_mb_per_sec / best_hf.throughput_mb_per_sec
            print(f"OVERALL BEST SPEEDUP vs HF Fast: {hf_speedup:.2f}x")
        
        # Final conclusion
        print(f"\n" + "="*80)
        if overall_speedup > 1.2:
            print(f"🚀 CONCLUSION: TokenDagger is {overall_speedup:.2f}x faster than TikToken at best!")
        elif overall_speedup > 0.8:
            print(f"⚡ CONCLUSION: TokenDagger and TikToken have similar performance ({overall_speedup:.2f}x)")
        else:
            print(f"🐌 CONCLUSION: TikToken is {1/overall_speedup:.2f}x faster than TokenDagger")
        print("="*80)
    
    def generate_performance_svg(self, output_path: str = "throughput_performance.svg"):
        """Generate an SVG bar chart showing TokenDagger vs TikToken performance."""
        if not HAS_MATPLOTLIB:
            print("Cannot generate SVG: matplotlib not available")
            return
            
        if not self.results:
            print("No benchmark results to visualize!")
            return
        
        # Group results by tokenizer
        td_results = [r for r in self.results if r.tokenizer_name == "TokenDagger"]
        tt_results = [r for r in self.results if r.tokenizer_name == "TikToken"]
        hf_results = [r for r in self.results if r.tokenizer_name == "HF Fast Tokenizer"]
        
        if not td_results or not tt_results:
            print("Incomplete results - need both TokenDagger and TikToken results")
            return
        
        # Prepare data for plotting
        thread_counts = sorted(list(set([r.thread_count for r in self.results])))
        td_throughputs = []
        tt_throughputs = []
        hf_throughputs = []
        
        for thread_count in thread_counts:
            td_result = next((r for r in td_results if r.thread_count == thread_count), None)
            tt_result = next((r for r in tt_results if r.thread_count == thread_count), None)
            hf_result = next((r for r in hf_results if r.thread_count == thread_count), None)
            
            td_throughputs.append(td_result.throughput_mb_per_sec if td_result else 0)
            tt_throughputs.append(tt_result.throughput_mb_per_sec if tt_result else 0)
            hf_throughputs.append(hf_result.throughput_mb_per_sec if hf_result else 0)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up bar positions
        x = np.arange(len(thread_counts))
        has_hf = any(hf_throughputs)
        width = 0.25 if has_hf else 0.35
        
        # Create bars
        if has_hf:
            td_bars = ax.bar(x - width, td_throughputs, width, 
                            label='TokenDagger', color='#2E86AB', alpha=0.8)
            tt_bars = ax.bar(x, tt_throughputs, width,
                            label='TikToken', color='#A23B72', alpha=0.8)
            hf_bars = ax.bar(x + width, hf_throughputs, width,
                            label='HF Fast Tokenizer', color='#F18F01', alpha=0.8)
        else:
            td_bars = ax.bar(x - width/2, td_throughputs, width, 
                            label='TokenDagger', color='#2E86AB', alpha=0.8)
            tt_bars = ax.bar(x + width/2, tt_throughputs, width,
                            label='TikToken', color='#A23B72', alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel('Thread Count', fontsize=14, fontweight='bold')
        ax.set_ylabel('Throughput (MB/s)', fontsize=14, fontweight='bold')
        title = f'TokenDagger vs TikToken'
        if has_hf:
            title += ' vs HF Fast Tokenizer'
        title += f' Performance Comparison\n({self.tokenizer_type.title()} Tokenizer, {self.text_size_mb}MB Text)'
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([str(tc) for tc in thread_counts])
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        max_throughput = max(max(td_throughputs), max(tt_throughputs))
        if has_hf:
            max_throughput = max(max_throughput, max(hf_throughputs))
        
        def add_value_labels(bars):
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + max_throughput * 0.01,
                           f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        add_value_labels(td_bars)
        add_value_labels(tt_bars)
        if has_hf:
            add_value_labels(hf_bars)
        
        # Style improvements
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        
        # Set y-axis to start from 0 and add some padding
        ax.set_ylim(0, max_throughput * 1.15)
        
        # Add speedup annotations
        for i, thread_count in enumerate(thread_counts):
            if td_throughputs[i] > 0 and tt_throughputs[i] > 0:
                speedup = td_throughputs[i] / tt_throughputs[i]
                bar_heights = [td_throughputs[i], tt_throughputs[i]]
                if has_hf and hf_throughputs[i] > 0:
                    bar_heights.append(hf_throughputs[i])
                y_pos = max(bar_heights) + max_throughput * 0.08
                color = 'green' if speedup > 1.0 else 'red' if speedup < 0.9 else 'orange'
                ax.text(i, y_pos, f'{speedup:.2f}x', ha='center', va='center', 
                       fontsize=9, fontweight='bold', color=color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save as SVG
        output_file = Path(output_path)
        plt.savefig(output_file, format='svg', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Performance chart saved to: {output_file.absolute()}")
        
        # Also save raw data as JSON for reference
        data_file = output_file.with_suffix('.json')
        chart_data = {
            'tokenizer_type': self.tokenizer_type,
            'text_size_mb': self.text_size_mb,
            'thread_counts': thread_counts,
            'tokendagger_throughput': td_throughputs,
            'tiktoken_throughput': tt_throughputs,
            'speedups_vs_tiktoken': [td/tt if tt > 0 else 0 for td, tt in zip(td_throughputs, tt_throughputs)]
        }
        
        if has_hf:
            chart_data['hf_fast_throughput'] = hf_throughputs
            chart_data['speedups_vs_hf'] = [td/hf if hf > 0 else 0 for td, hf in zip(td_throughputs, hf_throughputs)]
        
        with open(data_file, 'w') as f:
            json.dump(chart_data, f, indent=2)
        
        print(f"📈 Chart data saved to: {data_file.absolute()}")
    
    def run_full_benchmark(self):
        """Run the complete throughput benchmark suite."""
        try:
            self.setup_tokenizers()
            self.run_throughput_benchmarks()
            self.print_summary_report()
            
            # Generate SVG chart
            chart_filename = f"throughput_{self.tokenizer_type}_{self.text_size_mb}mb.svg"
            self.generate_performance_svg(chart_filename)
            
        except Exception as e:
            print(f"Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        return True


def main():
    """Main benchmark runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="TokenDagger vs TikToken Multithreaded Throughput Benchmark")
    parser.add_argument("--tokenizer", choices=["llama", "mistral"], default="llama", 
                       help="Tokenizer configuration to use (default: llama)")
    parser.add_argument("--threads", type=str, default="1,2,4,8,16,32",
                       help="Comma-separated list of thread counts to test (default: 1,2,4,8,16,32)")
    parser.add_argument("--text-size", type=int, default=1024,
                       help="Size of test text in MB (default: 1024 = 1GB)")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Iterations per thread (default: 10)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark (smaller text size and fewer iterations)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output SVG filename (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Parse thread counts
    thread_counts = [int(x.strip()) for x in args.threads.split(',')]
    
    # Quick mode adjustments
    if args.quick:
        args.text_size = 64  # 64 MB instead of 1 GB
        args.iterations = 3  # 3 iterations instead of 10
        print("Quick mode enabled: reduced text size and iterations")
    
    benchmark = ThroughputBenchmark(
        tokenizer_type=args.tokenizer,
        thread_counts=thread_counts,
        text_size_mb=args.text_size,
        iterations_per_thread=args.iterations
    )
    
    success = benchmark.run_full_benchmark()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()