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
                 thread_counts: List[int] = [1, 2, 4, 8],
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
        """Load Llama 4 configuration from the codebase."""
        # Hard-coded pattern from main.cpp
        pattern = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        
        # Load vocabulary from tokenizer.model
        vocab = self.load_bpe_vocab()
        
        # Load special tokens from tokenizer_config.json
        special_tokens = self.load_special_tokens()
        
        return pattern, vocab, special_tokens
    
    def load_mistral_config(self) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]: 
        """Load Mistral's Tekken 7 configuration from the codebase."""
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
        
        # Initialize TokenDagger
        tokenizer_name = f"{self.tokenizer_type}_throughput_test"
        self.tokendagger_tokenizer = tokendagger.create_tokenizer(
            name=tokenizer_name,
            pattern=pattern,
            vocab=vocab,
            special_tokens=special_tokens
        )
        
        # Initialize TikToken with the same vocab
        # Convert vocab format for TikToken
        tiktoken_vocab = {}
        for item in vocab:
            if isinstance(item["token_bytes"], list):
                # Convert list to bytes for mistral config
                token_bytes = bytes(item["token_bytes"])
            else:
                # Already bytes for llama config
                token_bytes = item["token_bytes"]
            tiktoken_vocab[token_bytes] = item["rank"]
        
        # Add special tokens to vocab
        for token_str, rank in special_tokens.items():
            tiktoken_vocab[token_str.encode('utf-8')] = rank
        
        # Create TikToken encoding with the same vocab
        self.tiktoken_tokenizer = tiktoken.Encoding(
            name=tokenizer_name,
            pat_str=pattern,
            mergeable_ranks=tiktoken_vocab,
            special_tokens=special_tokens
        )
        
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
            if len(text_chunks) % 1000 == 0:
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
            
            # Compare results
            if td_result and tt_result:
                mb_speedup = td_result.throughput_mb_per_sec / tt_result.throughput_mb_per_sec
                token_speedup = td_result.throughput_tokens_per_sec / tt_result.throughput_tokens_per_sec
                print(f"  Speedup: {mb_speedup:.2f}x MB/s, {token_speedup:.2f}x tokens/s")
    
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
        
        if not td_results or not tt_results:
            print("Incomplete results - need both TokenDagger and TikToken results")
            return
        
        print(f"\nTHROUGHPUT BY THREAD COUNT:")
        print(f"{'Threads':<8} {'TokenDagger':<15} {'TikToken':<15} {'Speedup':<10}")
        print(f"{'='*8} {'='*15} {'='*15} {'='*10}")
        
        for thread_count in self.thread_counts:
            td_result = next((r for r in td_results if r.thread_count == thread_count), None)
            tt_result = next((r for r in tt_results if r.thread_count == thread_count), None)
            
            if td_result and tt_result:
                speedup = td_result.throughput_mb_per_sec / tt_result.throughput_mb_per_sec
                print(f"{thread_count:<8} {td_result.throughput_mb_per_sec:<15.2f} "
                      f"{tt_result.throughput_mb_per_sec:<15.2f} {speedup:<10.2f}x")
        
        print(f"\nTOKENS PER SECOND BY THREAD COUNT:")
        print(f"{'Threads':<8} {'TokenDagger':<15} {'TikToken':<15} {'Speedup':<10}")
        print(f"{'='*8} {'='*15} {'='*15} {'='*10}")
        
        for thread_count in self.thread_counts:
            td_result = next((r for r in td_results if r.thread_count == thread_count), None)
            tt_result = next((r for r in tt_results if r.thread_count == thread_count), None)
            
            if td_result and tt_result:
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
        
        overall_speedup = best_td.throughput_mb_per_sec / best_tt.throughput_mb_per_sec
        print(f"\nOVERALL BEST SPEEDUP: {overall_speedup:.2f}x")
        
        # Final conclusion
        print(f"\n" + "="*80)
        if overall_speedup > 1.2:
            print(f"🚀 CONCLUSION: TokenDagger is {overall_speedup:.2f}x faster than TikToken at best!")
        elif overall_speedup > 0.8:
            print(f"⚡ CONCLUSION: TokenDagger and TikToken have similar performance ({overall_speedup:.2f}x)")
        else:
            print(f"🐌 CONCLUSION: TikToken is {1/overall_speedup:.2f}x faster than TokenDagger")
        print("="*80)
    
    def run_full_benchmark(self):
        """Run the complete throughput benchmark suite."""
        try:
            self.setup_tokenizers()
            self.run_throughput_benchmarks()
            self.print_summary_report()
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
    parser.add_argument("--threads", type=str, default="1,2,4,8",
                       help="Comma-separated list of thread counts to test (default: 1,2,4,8)")
    parser.add_argument("--text-size", type=int, default=1024,
                       help="Size of test text in MB (default: 1024 = 1GB)")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Iterations per thread (default: 10)")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick benchmark (smaller text size and fewer iterations)")
    
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