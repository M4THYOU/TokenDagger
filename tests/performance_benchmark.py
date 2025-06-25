#!/usr/bin/env python3
"""
Rigorous performance comparison between TokenDagger and TikToken.
Tests various edge cases, input lengths, and tokenization scenarios.

Usage:
# Full benchmark (100 runs each)
python tests/performance_benchmark.py

# Quick benchmark (10 runs each)
python tests/performance_benchmark.py --quick

# Custom run counts
python tests/performance_benchmark.py --warmup 3 --runs 50
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import time
import base64
import statistics
import random
import string
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import tiktoken

try:
    from tokendagger import wrapper as tokendagger
except ImportError as e:
    print(f"Failed to import tokendagger: {e}")
    print("Make sure to build the Python extension with 'make python'")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    test_name: str
    text_length: int
    token_count: int
    tokendagger_times: List[float]
    tiktoken_times: List[float]
    tokendagger_avg: float
    tiktoken_avg: float
    tokendagger_median: float
    tiktoken_median: float
    tokendagger_min: float
    tiktoken_min: float
    tokendagger_max: float
    tiktoken_max: float
    speedup_ratio: float  # tiktoken_avg / tokendagger_avg
    tokens_per_second_td: float
    tokens_per_second_tt: float


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self, warmup_runs: int = 5, benchmark_runs: int = 100):
        self.src_dir = Path(__file__).parent.parent / "src"
        self.test_dir = Path(__file__).parent
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[BenchmarkResult] = []
        
    def load_llama_config(self) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
        """Load Llama 4 configuration from the codebase."""
        # Hard-coded pattern from main.cpp
        pattern = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        
        # Load vocabulary from tokenizer.model
        vocab = self.load_bpe_vocab()
        
        # Load special tokens from tokenizer_config.json
        special_tokens = self.load_special_tokens()
        
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
        """Initialize both tokenizers with identical configuration."""
        print("Setting up tokenizers...")
        
        # Load Llama config
        pattern, vocab, special_tokens = self.load_llama_config()
        
        # Initialize TokenDagger
        self.tokendagger_tokenizer = tokendagger.create_tokenizer(
            name="llama4_perf_test",
            pattern=pattern,
            vocab=vocab,
            special_tokens=special_tokens
        )
        
        # Initialize TikToken with the same Llama 4 vocab
        # Convert vocab format for TikToken
        tiktoken_vocab = {}
        for item in vocab:
            token_bytes = bytes(item["token_bytes"])
            tiktoken_vocab[token_bytes] = item["rank"]
        
        # Add special tokens to vocab
        for token_str, rank in special_tokens.items():
            tiktoken_vocab[token_str.encode('utf-8')] = rank
        
        # Create TikToken encoding with the same vocab
        self.tiktoken_tokenizer = tiktoken.Encoding(
            name="llama4_perf_test",
            pat_str=pattern,
            mergeable_ranks=tiktoken_vocab,
            special_tokens=special_tokens
        )
        
        print(f"✓ TokenDagger tokenizer initialized")
        print(f"✓ TikToken tokenizer initialized")
    
    def generate_test_texts(self) -> Dict[str, List[str]]:
        """Generate comprehensive test corpus with various edge cases."""
        test_texts = defaultdict(list)
        
        # Load existing test files
        input_dir = self.test_dir / "input"
        if input_dir.exists():
            for txt_file in input_dir.glob("*.txt"):
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            test_texts[f"file_{txt_file.stem}"].append(content)
                except Exception as e:
                    print(f"Warning: Failed to load {txt_file}: {e}")
        
        # Edge cases - very short texts
        test_texts["edge_minimal"] = [
            "",  # Empty string
            " ",  # Single space
            "\n",  # Single newline
            "\t",  # Single tab
            "a",  # Single character
            "hi",  # Two characters
            "the",  # Common word
        ]
        
        # Edge cases - special tokens
        test_texts["edge_special_tokens"] = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|begin_of_text|>Hello<|end_of_text|>",
            "<|fim_prefix|>code<|fim_suffix|>",
            "Multiple <|begin_of_text|> special <|end_of_text|> tokens",
        ]
        
        # Edge cases - unicode and emojis
        test_texts["edge_unicode"] = [
            "Hello 世界",
            "Café résumé naïve",
            "こんにちは世界",
            "🚀🌟✨💫⭐",
            "👨‍💻👩‍🔬🧑‍🎨",  # Complex emoji sequences
            "Ĥëłłø Wörłð",  # Accented characters
            "αβγδεζηθικλμνξοπρστυφχψω",  # Greek alphabet
            "🇺🇸🇬🇧🇫🇷🇩🇪🇯🇵",  # Flag emojis
        ]
        
        # Edge cases - punctuation and symbols
        test_texts["edge_punctuation"] = [
            "!@#$%^&*()_+-={}[]|\\:;\"'<>?,./ ",
            "Hello, world! How are you? I'm fine.",
            "Testing... ellipsis... and --- dashes.",
            "(Parentheses) [brackets] {braces} <angles>",
            "Quote: \"Hello,\" she said. 'Indeed,' he replied.",
            "Code: x = y + z; if (x > 0) { return true; }",
        ]
        
        # Edge cases - numbers and mixed content
        test_texts["edge_numbers"] = [
            "123456789",
            "1.234567890",
            "2024-01-15T14:30:00Z",
            "Price: $123.45 (was $150.00)",
            "Version 2.1.3-beta.4",
            "Phone: +1-555-123-4567",
            "IP: 192.168.1.1:8080",
        ]
        
        # Repetitive patterns (edge case for BPE)
        test_texts["edge_repetitive"] = [
            "a" * 100,
            "the " * 50,
            "hello world " * 25,
            ("Lorem ipsum dolor sit amet " * 10),
            "abcdefghijklmnopqrstuvwxyz" * 4,
        ]
        
        # Generate synthetic texts of varying lengths
        for length in [10, 50, 100, 500, 1000, 2000, 5000, 10000]:
            # Random English-like text
            words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
                    "hello", "world", "test", "example", "performance", "benchmark", 
                    "tokenizer", "comparison", "evaluation", "analysis"]
            text = " ".join(random.choices(words, k=length // 5))
            test_texts[f"synthetic_length_{length}"].append(text)
            
            # Random ASCII text
            ascii_text = ''.join(random.choices(string.ascii_letters + string.digits + " .,!?", k=length))
            test_texts[f"random_ascii_{length}"].append(ascii_text)
        
        # Code snippets (common use case)
        test_texts["code_samples"] = [
            """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
            """,
            """
function quickSort(arr) {
    if (arr.length <= 1) return arr;
    const pivot = arr[Math.floor(arr.length / 2)];
    const left = arr.filter(x => x < pivot);
    const right = arr.filter(x => x > pivot);
    return [...quickSort(left), pivot, ...quickSort(right)];
}
            """,
            """
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::vector<int> v = {3, 1, 4, 1, 5, 9, 2, 6};
    std::sort(v.begin(), v.end());
    for (int i : v) {
        std::cout << i << " ";
    }
    return 0;
}
            """,
        ]
        
        # JSON data (structured text)
        test_texts["structured_json"] = [
            json.dumps({
                "name": "John Doe",
                "age": 30,
                "city": "New York",
                "hobbies": ["reading", "swimming", "coding"],
                "address": {
                    "street": "123 Main St",
                    "zipcode": "10001"
                }
            }, indent=2),
            json.dumps([{"id": i, "value": f"item_{i}"} for i in range(100)]),
        ]
        
        # Long form content (realistic use cases)
        test_texts["long_form"] = [
            self.generate_article(2000),
            self.generate_article(5000),
            self.generate_article(10000),
        ]
        
        return dict(test_texts)
    
    def generate_article(self, target_length: int) -> str:
        """Generate a realistic article of approximately target_length characters."""
        paragraphs = [
            "In the rapidly evolving landscape of artificial intelligence and machine learning, tokenization has emerged as a fundamental preprocessing step that significantly impacts model performance and efficiency.",
            
            "Tokenization, the process of converting raw text into discrete tokens that can be processed by neural networks, involves numerous design decisions that affect both accuracy and computational efficiency.",
            
            "Modern tokenization approaches, including Byte Pair Encoding (BPE), WordPiece, and SentencePiece, each offer distinct advantages and trade-offs in terms of vocabulary size, compression efficiency, and handling of out-of-vocabulary terms.",
            
            "The choice of tokenization strategy becomes particularly critical when dealing with multilingual models, code generation tasks, and domains with specialized vocabulary such as scientific literature or technical documentation.",
            
            "Performance optimization in tokenization systems requires careful consideration of algorithmic complexity, memory usage patterns, and the specific characteristics of the target corpus.",
            
            "Recent advances in tokenization include learned tokenizers that adapt to specific domains, sub-word regularization techniques that improve model robustness, and efficient implementations that leverage parallel processing capabilities.",
            
            "Evaluation of tokenization systems typically involves metrics such as compression ratio, vocabulary coverage, tokenization speed, and downstream task performance across diverse benchmarks and real-world applications.",
        ]
        
        article = ""
        while len(article) < target_length:
            article += random.choice(paragraphs) + "\n\n"
        
        return article[:target_length]
    
    def benchmark_single_text(self, test_name: str, text: str) -> BenchmarkResult:
        """Benchmark both tokenizers on a single text."""
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                self.tokendagger_tokenizer.encode_ordinary(text)
                self.tiktoken_tokenizer.encode_ordinary(text)
            except Exception as e:
                print(f"Warning: Warmup failed for {test_name}: {e}")
                break
        
        # Get token count for metrics
        try:
            tokens = self.tokendagger_tokenizer.encode_ordinary(text)
            token_count = len(tokens)
        except Exception:
            try:
                tokens = self.tiktoken_tokenizer.encode_ordinary(text)
                token_count = len(tokens)
            except Exception:
                token_count = 0
        
        # Benchmark TokenDagger
        tokendagger_times = []
        for _ in range(self.benchmark_runs):
            start_time = time.perf_counter()
            try:
                self.tokendagger_tokenizer.encode_ordinary(text)
                end_time = time.perf_counter()
                tokendagger_times.append(end_time - start_time)
            except Exception as e:
                print(f"TokenDagger error on {test_name}: {e}")
                tokendagger_times.append(float('inf'))
        
        # Benchmark TikToken
        tiktoken_times = []
        for _ in range(self.benchmark_runs):
            start_time = time.perf_counter()
            try:
                self.tiktoken_tokenizer.encode_ordinary(text)
                end_time = time.perf_counter()
                tiktoken_times.append(end_time - start_time)
            except Exception as e:
                print(f"TikToken error on {test_name}: {e}")
                tiktoken_times.append(float('inf'))
        
        # Calculate statistics
        tokendagger_avg = statistics.mean(tokendagger_times)
        tiktoken_avg = statistics.mean(tiktoken_times)
        tokendagger_median = statistics.median(tokendagger_times)
        tiktoken_median = statistics.median(tiktoken_times)
        tokendagger_min = min(tokendagger_times)
        tiktoken_min = min(tiktoken_times)
        tokendagger_max = max(tokendagger_times)
        tiktoken_max = max(tiktoken_times)
        
        speedup_ratio = tiktoken_avg / tokendagger_avg if tokendagger_avg > 0 else float('inf')
        tokens_per_second_td = token_count / tokendagger_avg if tokendagger_avg > 0 else 0
        tokens_per_second_tt = token_count / tiktoken_avg if tiktoken_avg > 0 else 0
        
        return BenchmarkResult(
            test_name=test_name,
            text_length=len(text),
            token_count=token_count,
            tokendagger_times=tokendagger_times,
            tiktoken_times=tiktoken_times,
            tokendagger_avg=tokendagger_avg,
            tiktoken_avg=tiktoken_avg,
            tokendagger_median=tokendagger_median,
            tiktoken_median=tiktoken_median,
            tokendagger_min=tokendagger_min,
            tiktoken_min=tiktoken_min,
            tokendagger_max=tokendagger_max,
            tiktoken_max=tiktoken_max,
            speedup_ratio=speedup_ratio,
            tokens_per_second_td=tokens_per_second_td,
            tokens_per_second_tt=tokens_per_second_tt
        )
    
    def run_benchmarks(self):
        """Run comprehensive benchmark suite."""
        print("="*80)
        print("TOKENIZER PERFORMANCE BENCHMARK")
        print("="*80)
        print(f"Warmup runs: {self.warmup_runs}")
        print(f"Benchmark runs: {self.benchmark_runs}")
        print()
        
        # Generate test corpus
        print("Generating test corpus...")
        test_texts = self.generate_test_texts()
        
        total_tests = sum(len(texts) for texts in test_texts.values())
        current_test = 0
        
        # Run benchmarks
        for category, texts in test_texts.items():
            print(f"\n--- {category.upper().replace('_', ' ')} ---")
            
            for i, text in enumerate(texts):
                current_test += 1
                test_name = f"{category}_{i}"
                
                print(f"[{current_test:3d}/{total_tests}] {test_name:<30} "
                      f"({len(text):6d} chars)", end=" ... ", flush=True)
                
                result = self.benchmark_single_text(test_name, text)
                self.results.append(result)
                
                # Print quick result
                if result.speedup_ratio != float('inf'):
                    speedup_indicator = "🚀" if result.speedup_ratio > 1.5 else "⚡" if result.speedup_ratio > 1.0 else "🐌"
                    print(f"{speedup_indicator} {result.speedup_ratio:.2f}x speedup")
                else:
                    print("❌ ERROR")
    
    def print_summary_report(self):
        """Print comprehensive performance analysis."""
        if not self.results:
            print("No benchmark results to analyze!")
            return
        
        print("\n" + "="*80)
        print("PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        # Overall statistics
        valid_results = [r for r in self.results if r.speedup_ratio != float('inf')]
        if not valid_results:
            print("No valid results to analyze!")
            return
        
        speedups = [r.speedup_ratio for r in valid_results]
        td_throughputs = [r.tokens_per_second_td for r in valid_results]
        tt_throughputs = [r.tokens_per_second_tt for r in valid_results]
        
        print(f"\nOVERALL PERFORMANCE METRICS:")
        print(f"  Total tests completed: {len(valid_results)}")
        print(f"  Failed tests: {len(self.results) - len(valid_results)}")
        print(f"  Average speedup: {statistics.mean(speedups):.3f}x")
        print(f"  Median speedup: {statistics.median(speedups):.3f}x")
        print(f"  Min speedup: {min(speedups):.3f}x")
        print(f"  Max speedup: {max(speedups):.3f}x")
        print(f"  Speedup std dev: {statistics.stdev(speedups):.3f}")
        
        print(f"\nTHROUGHPUT COMPARISON:")
        print(f"  TokenDagger avg: {statistics.mean(td_throughputs):,.0f} tokens/sec")
        print(f"  TikToken avg: {statistics.mean(tt_throughputs):,.0f} tokens/sec")
        print(f"  TokenDagger max: {max(td_throughputs):,.0f} tokens/sec")
        print(f"  TikToken max: {max(tt_throughputs):,.0f} tokens/sec")
        
        # Performance by text length
        print(f"\nPERFORMANCE BY TEXT LENGTH:")
        length_buckets = [
            (0, 100, "Very Short (0-100 chars)"),
            (100, 1000, "Short (100-1K chars)"),
            (1000, 5000, "Medium (1K-5K chars)"),
            (5000, 20000, "Long (5K-20K chars)"),
            (20000, float('inf'), "Very Long (20K+ chars)")
        ]
        
        for min_len, max_len, label in length_buckets:
            bucket_results = [r for r in valid_results 
                            if min_len <= r.text_length < max_len]
            if bucket_results:
                bucket_speedups = [r.speedup_ratio for r in bucket_results]
                bucket_td_throughput = [r.tokens_per_second_td for r in bucket_results]
                print(f"  {label:<25}: {statistics.mean(bucket_speedups):.2f}x speedup, "
                      f"{statistics.mean(bucket_td_throughput):,.0f} tok/s (TD)")
        
        # Best and worst performing cases
        print(f"\nBEST PERFORMING CASES:")
        best_cases = sorted(valid_results, key=lambda r: r.speedup_ratio, reverse=True)[:5]
        for result in best_cases:
            print(f"  {result.test_name:<30}: {result.speedup_ratio:.2f}x "
                  f"({result.text_length:,} chars, {result.token_count:,} tokens)")
        
        print(f"\nWORST PERFORMING CASES:")
        worst_cases = sorted(valid_results, key=lambda r: r.speedup_ratio)[:5]
        for result in worst_cases:
            print(f"  {result.test_name:<30}: {result.speedup_ratio:.2f}x "
                  f"({result.text_length:,} chars, {result.token_count:,} tokens)")
        
        # Performance consistency analysis
        print(f"\nPERFORMANCE CONSISTENCY:")
        for result in valid_results[:5]:  # Sample first 5 results
            td_cv = statistics.stdev(result.tokendagger_times) / statistics.mean(result.tokendagger_times)
            tt_cv = statistics.stdev(result.tiktoken_times) / statistics.mean(result.tiktoken_times)
            print(f"  {result.test_name:<30}: TD CV={td_cv:.3f}, TT CV={tt_cv:.3f}")
        
        # Final recommendation
        avg_speedup = statistics.mean(speedups)
        print(f"\n" + "="*80)
        if avg_speedup > 1.2:
            print(f"🎉 CONCLUSION: TokenDagger is {avg_speedup:.2f}x faster than TikToken on average!")
        elif avg_speedup > 0.8:
            print(f"⚡ CONCLUSION: TokenDagger and TikToken have similar performance ({avg_speedup:.2f}x)")
        else:
            print(f"🐌 CONCLUSION: TikToken is {1/avg_speedup:.2f}x faster than TokenDagger")
        print("="*80)
    
    def run_full_benchmark(self):
        """Run the complete benchmark suite."""
        try:
            self.setup_tokenizers()
            self.run_benchmarks()
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
    
    parser = argparse.ArgumentParser(description="TokenDagger vs TikToken Performance Benchmark")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup runs")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (fewer runs)")
    
    args = parser.parse_args()
    
    if args.quick:
        args.warmup = 2
        args.runs = 10
    
    benchmark = PerformanceBenchmark(warmup_runs=args.warmup, benchmark_runs=args.runs)
    success = benchmark.run_full_benchmark()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()