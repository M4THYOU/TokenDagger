# TokenDagger: High-Performance Implementation of OpenAI's TikToken

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/tokendagger.svg)](https://badge.fury.io/py/tokendagger)

A fast implementation of OpenAI's [TikToken](https://github.com/openai/tiktoken), designed for large-scale text processing. 2x Throughput and 4x faster on code sample tokenization.

## Benchmarks

Performed on an `AMD EPYC 4584PX - 16c/32t - 4.2 GHz`.

![Throughput Benchmark Results](throughput_llama_1024mb.svg)

- **Fast Regex Parsing**: Optimized PCRE2 regex engine for efficient token pattern matching
- **Simplified BPE**: Simplied algorithm to reduce performance impact of large special token vocabulary.
- **OpenAI Compatible**: Full compatibility with OpenAI's TikToken tokenizer

## Run Tests

```bash
make clean && make
pip3 install tiktoken
python3 tests/test_tokendagger_vs_tiktoken.py --tokenizer llama
python3 tests/test_tokendagger_vs_tiktoken.py --tokenizer mistral
python3 tests/performance_benchmark.py --tokenizer llama
python3 tests/performance_benchmark.py --tokenizer mistral
python3 tests/code_performance_benchmark.py --tokenizer llama
```

```
================================================================================
🎉 CONCLUSION: TokenDagger is 4.02x faster on code tokenization!
================================================================================
```

## 📦 Installation

### From PyPI (Recommended)

```
pip install tokendagger
```


## 🛠️ Dev Install

```
git clone git@github.com:M4THYOU/TokenDagger.git
sudo apt install libpcre2-dev
git submodule update --init --recursive
sudo apt update && sudo apt install -y python3-dev
```

And optionally for running the tests:
```
pip3 install tiktoken
```



## Dependencies
- **PCRE2**: Perl Compatible Regular Expressions - [GitHub](https://github.com/PCRE2Project/pcre2)
