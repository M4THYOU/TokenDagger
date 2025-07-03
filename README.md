# TokenDagger: High-Performance Implementation of OpenAI's TikToken

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/tokendagger.svg)](https://badge.fury.io/py/tokendagger)

A fast, drop-in implementation of OpenAI's [TikToken](https://github.com/openai/tiktoken), designed for large-scale text processing. 2x Throughput and 4x faster on code sample tokenization.

## Benchmarks

Performed on an `AMD EPYC 4584PX - 16c/32t - 4.2 GHz` w/64GB memory. Hugging Face's batch tokenizer used way more memory than Tiktoken and TokenDagger. 256MB was the largest input size it could process with OOM.

The benchmark was performed on [meta-llama/Llama-4-Scout-17B-16E-Instruct](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/tree/main). [mistralai/Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410/tree/main) is also supported for testing, by using argument `--tokenizer mistral`.

![Throughput Benchmark Results](throughput_llama_1024mb.svg)
![Throughput Benchmark Results](throughput_llama_256mb.svg)

- **Fast Regex Parsing**: Optimized PCRE2 regex engine for efficient token pattern matching
- **Drop-In Replacement**: Full compatibility with OpenAI's TikToken tokenizer
- **Simplified BPE**: Simplied algorithm to reduce performance impact of large special token vocabulary.

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

## 📦 Usage


```
pip install tokendagger
```

Then replace Tiktoken:

```python
# import tiktoken --- Remove this line!
import tokendagger as tiktoken

...

tokenizer = tiktoken.Encoding(
    name=name,
    pat_str=pattern,
    mergeable_ranks=vocab,
    special_tokens=special_tokens,
)
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
