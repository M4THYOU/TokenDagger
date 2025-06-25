#!/usr/bin/env python3
"""
Code-specific performance benchmark for TokenDagger vs TikToken.
Tests tokenization performance on actual code files from the repository.

Usage:
# Full benchmark on all code files
python tests/code_performance_benchmark.py

# Quick benchmark (fewer runs)
python tests/code_performance_benchmark.py --quick

# Custom run counts
python tests/code_performance_benchmark.py --warmup 3 --runs 50

# Test specific file types only
python tests/code_performance_benchmark.py --extensions .py .cpp
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import argparse

import tiktoken

try:
    from tokendagger import wrapper as tokendagger
except ImportError as e:
    print(f"Failed to import tokendagger: {e}")
    print("Make sure to build the Python extension with 'make python'")
    sys.exit(1)


@dataclass
class CodeBenchmarkResult:
    """Results from benchmarking a single code file."""
    file_path: str
    file_type: str
    file_size: int
    token_count: int
    tokendagger_times: List[float]
    tiktoken_times: List[float]
    tokendagger_avg: float
    tiktoken_avg: float
    tokendagger_median: float
    tiktoken_median: float
    speedup_ratio: float
    tokens_per_second_td: float
    tokens_per_second_tt: float


class CodePerformanceBenchmark:
    """Performance benchmark specifically for code tokenization."""
    
    def __init__(self, warmup_runs: int = 3, benchmark_runs: int = 25):
        self.repo_root = Path(__file__).parent.parent
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs
        self.results: List[CodeBenchmarkResult] = []
        
        # Code file extensions to test
        self.code_extensions = {
            '.py': 'Python',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C Header',
            '.hpp': 'C++ Header',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.rs': 'Rust',
            '.go': 'Go',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.cs': 'C#',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.sh': 'Shell',
            '.bat': 'Batch',
            '.ps1': 'PowerShell',
            '.sql': 'SQL',
            '.json': 'JSON',
            '.xml': 'XML',
            '.yaml': 'YAML',
            '.yml': 'YAML',
            '.md': 'Markdown',
            '.txt': 'Text',
            '.makefile': 'Makefile',
            '.cmake': 'CMake'
        }
        
    def setup_tokenizers(self):
        """Initialize both tokenizers with Llama 4 configuration."""
        print("Setting up tokenizers with Llama 4 configuration...")
        
        # Load Llama 4 configuration
        pattern, vocab, special_tokens = self.load_llama_config()
        
        # Initialize TokenDagger with Llama 4 config
        self.tokendagger_tokenizer = tokendagger.create_tokenizer(
            name="llama4_code_benchmark",
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
            name="llama4_code_benchmark",
            pat_str=pattern,
            mergeable_ranks=tiktoken_vocab,
            special_tokens=special_tokens
        )
        
        print(f"✓ TokenDagger tokenizer initialized with Llama 4 config")
        print(f"✓ TikToken tokenizer initialized with Llama 4 config")
    
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
        model_file = self.repo_root / "src" / "tokenizer.model"
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
                        import base64
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
        config_file = self.repo_root / "src" / "tokenizer_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        special_tokens = {}
        import json
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
            # Extract special tokens from added_tokens_decoder
            if "added_tokens_decoder" in config:
                for token_id, token_info in config["added_tokens_decoder"].items():
                    special_tokens[token_info["content"]] = int(token_id)
        
        print(f"Loaded {len(special_tokens)} special tokens from tokenizer_config.json")
        return special_tokens
    
    def find_code_files(self, extensions: Optional[List[str]] = None) -> Dict[str, List[Path]]:
        """Find all code files in the repository, excluding external libraries."""
        if extensions is None:
            extensions = list(self.code_extensions.keys())
        
        code_files = defaultdict(list)
        
        # Directories to exclude (external libraries, build artifacts, etc.)
        exclude_dirs = {
            'extern', 'build', '__pycache__', '.git', '.vscode', 
            'node_modules', 'target', 'dist', 'out', '.pytest_cache'
        }
        
        # Directories to specifically include (our source code)
        include_dirs = {'src', 'tokendagger', 'tests'}
        
        def should_include_path(path: Path) -> bool:
            """Check if a path should be included in the benchmark."""
            # Check if any parent directory is in exclude list
            for part in path.parts:
                if part in exclude_dirs:
                    return False
            
            # Only include files in our source directories
            for part in path.parts:
                if part in include_dirs:
                    return True
            
            # Include files in root directory
            if len(path.parts) <= 2:  # Root or one level deep
                return True
                
            return False
        
        for ext in extensions:
            pattern = f"**/*{ext}"
            for file_path in self.repo_root.glob(pattern):
                if file_path.is_file() and should_include_path(file_path):
                    relative_path = file_path.relative_to(self.repo_root)
                    code_files[ext].append(relative_path)
        
        # Add Makefile and other files without extensions
        for special_file in ['Makefile', 'CMakeLists.txt']:
            special_path = self.repo_root / special_file
            if special_path.exists():
                code_files['.makefile'].append(special_path.relative_to(self.repo_root))
        
        # Sort files by size (smaller first for faster initial results)
        for ext in code_files:
            code_files[ext].sort(key=lambda p: (self.repo_root / p).stat().st_size)
        
        return dict(code_files)
    
    def read_file_safely(self, file_path: Path) -> Optional[str]:
        """Safely read a file with fallback encodings."""
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                print(f"Warning: Failed to read {file_path}: {e}")
                return None
        
        print(f"Warning: Could not decode {file_path} with any encoding")
        return None
    
    def benchmark_single_file(self, file_path: Path) -> Optional[CodeBenchmarkResult]:
        """Benchmark both tokenizers on a single code file."""
        # Read file content
        content = self.read_file_safely(file_path)
        if content is None:
            return None
        
        # Skip very large files (>1MB) to keep benchmark reasonable
        if len(content) > 1024 * 1024:
            print(f"Skipping large file: {file_path} ({len(content):,} chars)")
            return None
        
        # Skip empty files
        if not content.strip():
            return None
        
        file_type = self.code_extensions.get(file_path.suffix, 'Unknown')
        relative_path = str(file_path.relative_to(self.repo_root))
        
        # Warmup runs
        for _ in range(self.warmup_runs):
            try:
                self.tokendagger_tokenizer.encode(content, allowed_special=set(), disallowed_special=set())
                self.tiktoken_tokenizer.encode(content, allowed_special=set(), disallowed_special=set())
            except Exception as e:
                print(f"Warning: Warmup failed for {relative_path}: {e}")
                break
        
        # Get token count for metrics
        try:
            tokens = self.tokendagger_tokenizer.encode(content, allowed_special=set(), disallowed_special=set())
            token_count = len(tokens)
        except Exception:
            try:
                tokens = self.tiktoken_tokenizer.encode(content, allowed_special=set(), disallowed_special=set())
                token_count = len(tokens)
            except Exception:
                print(f"Error: Could not tokenize {relative_path}")
                return None
        
        # Benchmark TokenDagger
        tokendagger_times = []
        for _ in range(self.benchmark_runs):
            start_time = time.perf_counter()
            try:
                self.tokendagger_tokenizer.encode(content, allowed_special=set(), disallowed_special=set())
                end_time = time.perf_counter()
                tokendagger_times.append(end_time - start_time)
            except Exception as e:
                print(f"TokenDagger error on {relative_path}: {e}")
                tokendagger_times.append(float('inf'))
        
        # Benchmark TikToken
        tiktoken_times = []
        for _ in range(self.benchmark_runs):
            start_time = time.perf_counter()
            try:
                self.tiktoken_tokenizer.encode(content, allowed_special=set(), disallowed_special=set())
                end_time = time.perf_counter()
                tiktoken_times.append(end_time - start_time)
            except Exception as e:
                print(f"TikToken error on {relative_path}: {e}")
                tiktoken_times.append(float('inf'))
        
        # Calculate statistics
        tokendagger_avg = statistics.mean(tokendagger_times)
        tiktoken_avg = statistics.mean(tiktoken_times)
        tokendagger_median = statistics.median(tokendagger_times)
        tiktoken_median = statistics.median(tiktoken_times)
        
        speedup_ratio = tiktoken_avg / tokendagger_avg if tokendagger_avg > 0 else float('inf')
        tokens_per_second_td = token_count / tokendagger_avg if tokendagger_avg > 0 else 0
        tokens_per_second_tt = token_count / tiktoken_avg if tiktoken_avg > 0 else 0
        
        return CodeBenchmarkResult(
            file_path=relative_path,
            file_type=file_type,
            file_size=len(content),
            token_count=token_count,
            tokendagger_times=tokendagger_times,
            tiktoken_times=tiktoken_times,
            tokendagger_avg=tokendagger_avg,
            tiktoken_avg=tiktoken_avg,
            tokendagger_median=tokendagger_median,
            tiktoken_median=tiktoken_median,
            speedup_ratio=speedup_ratio,
            tokens_per_second_td=tokens_per_second_td,
            tokens_per_second_tt=tokens_per_second_tt
        )
    
    def run_benchmarks(self, extensions: Optional[List[str]] = None):
        """Run benchmark suite on code files."""
        print("="*80)
        print("CODE TOKENIZATION PERFORMANCE BENCHMARK")
        print("="*80)
        print(f"Warmup runs: {self.warmup_runs}")
        print(f"Benchmark runs: {self.benchmark_runs}")
        print()
        
        # Find code files
        print("Finding code files...")
        code_files = self.find_code_files(extensions)
        
        if not code_files:
            print("No code files found!")
            return
        
        total_files = sum(len(files) for files in code_files.values())
        current_file = 0
        
        print(f"Found {total_files} code files across {len(code_files)} file types")
        print()
        
        # Run benchmarks by file type
        for ext, files in code_files.items():
            if not files:
                continue
                
            file_type = self.code_extensions.get(ext, 'Unknown')
            print(f"--- {file_type.upper()} FILES ({ext}) ---")
            
            for file_path in files:
                current_file += 1
                full_path = self.repo_root / file_path
                
                print(f"[{current_file:3d}/{total_files}] {str(file_path):<50} "
                      f"({full_path.stat().st_size:6,} bytes)", end=" ... ", flush=True)
                
                result = self.benchmark_single_file(full_path)
                if result is None:
                    print("⏭️  SKIPPED")
                    continue
                
                self.results.append(result)
                
                # Print quick result
                if result.speedup_ratio != float('inf'):
                    if result.speedup_ratio > 2.0:
                        speedup_indicator = "🚀"
                    elif result.speedup_ratio > 1.2:
                        speedup_indicator = "⚡"
                    elif result.speedup_ratio > 0.8:
                        speedup_indicator = "⚖️"
                    else:
                        speedup_indicator = "🐌"
                    print(f"{speedup_indicator} {result.speedup_ratio:.2f}x speedup "
                          f"({result.tokens_per_second_td:,.0f} tok/s)")
                else:
                    print("❌ ERROR")
            
            print()  # Empty line between file types
    
    def print_summary_report(self):
        """Print comprehensive analysis of code tokenization performance."""
        if not self.results:
            print("No benchmark results to analyze!")
            return
        
        print("="*80)
        print("CODE TOKENIZATION ANALYSIS SUMMARY")
        print("="*80)
        
        # Overall statistics
        valid_results = [r for r in self.results if r.speedup_ratio != float('inf')]
        if not valid_results:
            print("No valid results to analyze!")
            return
        
        speedups = [r.speedup_ratio for r in valid_results]
        td_throughputs = [r.tokens_per_second_td for r in valid_results]
        tt_throughputs = [r.tokens_per_second_tt for r in valid_results]
        
        print(f"\nOVERALL PERFORMANCE:")
        print(f"  Files processed: {len(valid_results)}")
        print(f"  Failed files: {len(self.results) - len(valid_results)}")
        print(f"  Average speedup: {statistics.mean(speedups):.3f}x")
        print(f"  Median speedup: {statistics.median(speedups):.3f}x")
        print(f"  Best speedup: {max(speedups):.3f}x")
        print(f"  Worst speedup: {min(speedups):.3f}x")
        
        print(f"\nTHROUGHPUT COMPARISON:")
        print(f"  TokenDagger avg: {statistics.mean(td_throughputs):,.0f} tokens/sec")
        print(f"  TikToken avg: {statistics.mean(tt_throughputs):,.0f} tokens/sec")
        print(f"  TokenDagger peak: {max(td_throughputs):,.0f} tokens/sec")
        print(f"  TikToken peak: {max(tt_throughputs):,.0f} tokens/sec")
        
        # Performance by file type
        print(f"\nPERFORMANCE BY FILE TYPE:")
        file_type_stats = defaultdict(list)
        for result in valid_results:
            file_type_stats[result.file_type].append(result.speedup_ratio)
        
        for file_type, speedups_by_type in sorted(file_type_stats.items()):
            avg_speedup = statistics.mean(speedups_by_type)
            print(f"  {file_type:<15}: {avg_speedup:.2f}x speedup "
                  f"({len(speedups_by_type)} files)")
        
        # Performance by file size
        print(f"\nPERFORMANCE BY FILE SIZE:")
        size_buckets = [
            (0, 1000, "Small (0-1KB)"),
            (1000, 10000, "Medium (1-10KB)"),
            (10000, 50000, "Large (10-50KB)"),
            (50000, float('inf'), "Very Large (50KB+)")
        ]
        
        for min_size, max_size, label in size_buckets:
            bucket_results = [r for r in valid_results 
                            if min_size <= r.file_size < max_size]
            if bucket_results:
                bucket_speedups = [r.speedup_ratio for r in bucket_results]
                bucket_throughput = [r.tokens_per_second_td for r in bucket_results]
                print(f"  {label:<20}: {statistics.mean(bucket_speedups):.2f}x speedup, "
                      f"{statistics.mean(bucket_throughput):,.0f} tok/s "
                      f"({len(bucket_results)} files)")
        
        # Top performing files
        print(f"\nTOP PERFORMING FILES:")
        best_files = sorted(valid_results, key=lambda r: r.speedup_ratio, reverse=True)[:10]
        for i, result in enumerate(best_files, 1):
            print(f"  {i:2d}. {result.file_path:<40} "
                  f"{result.speedup_ratio:.2f}x ({result.file_type})")
        
        # Worst performing files
        print(f"\nWORST PERFORMING FILES:")
        worst_files = sorted(valid_results, key=lambda r: r.speedup_ratio)[:5]
        for i, result in enumerate(worst_files, 1):
            print(f"  {i:2d}. {result.file_path:<40} "
                  f"{result.speedup_ratio:.2f}x ({result.file_type})")
        
        # Final conclusion
        avg_speedup = statistics.mean(speedups)
        print(f"\n" + "="*80)
        if avg_speedup > 1.3:
            print(f"🎉 CONCLUSION: TokenDagger is {avg_speedup:.2f}x faster on code tokenization!")
        elif avg_speedup > 0.9:
            print(f"⚡ CONCLUSION: Similar performance ({avg_speedup:.2f}x) on code tokenization")
        else:
            print(f"🐌 CONCLUSION: TikToken is {1/avg_speedup:.2f}x faster on code tokenization")
        print("="*80)
    
    def run_full_benchmark(self, extensions: Optional[List[str]] = None):
        """Run the complete code benchmark suite."""
        try:
            self.setup_tokenizers()
            self.run_benchmarks(extensions)
            self.print_summary_report()
        except Exception as e:
            print(f"Benchmark failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        return True


def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Code Tokenization Performance Benchmark")
    parser.add_argument("--warmup", type=int, default=3, 
                       help="Number of warmup runs (default: 3)")
    parser.add_argument("--runs", type=int, default=25, 
                       help="Number of benchmark runs (default: 25)")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick benchmark (fewer runs)")
    parser.add_argument("--extensions", nargs="+", 
                       help="File extensions to test (e.g., .py .cpp .js)")
    
    args = parser.parse_args()
    
    if args.quick:
        args.warmup = 2
        args.runs = 10
    
    benchmark = CodePerformanceBenchmark(
        warmup_runs=args.warmup, 
        benchmark_runs=args.runs
    )
    
    success = benchmark.run_full_benchmark(args.extensions)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()