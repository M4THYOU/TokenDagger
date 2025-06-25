import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tokendagger
import time

# Create a simple vocabulary for testing
# In a real scenario, you'd load this from actual tokenizer files
def create_test_vocab():
    vocab_data = []
    
    # Add basic ASCII characters and common words
    vocab_items = [
        # Common punctuation and symbols
        " ", ".", ",", "!", "?", ":", ";", "'", '"', "-", "(", ")", "[", "]", "{", "}", 
        "\n", "\t",
        # Common letters and combinations
        "a", "e", "i", "o", "u", "t", "n", "s", "r", "h", "l", "d", "c", "m", "f", "p", "g", "w", "y", "b", "v", "k", "x", "j", "q", "z",
        "A", "E", "I", "O", "U", "T", "N", "S", "R", "H", "L", "D", "C", "M", "F", "P", "G", "W", "Y", "B", "V", "K", "X", "J", "Q", "Z",
        # Common words
        "the", "and", "of", "to", "in", "for", "is", "on", "that", "by", "this", "with", "are", "as", "be", "or", "an", "will", "my", "one", "all", "would", "there", "their",
        "What", "Here", "Python", "JavaScript", "web", "development", "programming", "language", "code", "function", "example", "syntax", "differences",
        "applications", "backend", "frontend", "beginner", "recommend", "projects", "todo", "list", "script", "html", "css", "react", "node",
        # Numbers
        "0", "1", "2", "3", "4", "5", "2024",
        # Programming-related
        "def", "if", "else", "for", "while", "class", "import", "from", "return", "print", "console", "log",
        "**", "##", "```", "python", "javascript", "const", "let", "var", "function",
    ]
    
    for i, item in enumerate(vocab_items):
        vocab_data.append({
            "rank": i,
            "token_bytes": list(item.encode('utf-8')),
            "token_string": item
        })
    
    return vocab_data

def create_special_tokens():
    return {
        "<|begin_of_text|>": 50000,
        "<|end_of_text|>": 50001,
        "<|eot|>": 50002,
        "<s>": 50003,
        "</s>": 50004,
        "[INST]": 50005,
        "[/INST]": 50006,
    }

####################################################################################

# Load lorem ipsum text from file
try:
    with open("./input/lorem.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
except FileNotFoundError:
    # Fallback text if file doesn't exist
    prompt = "This is a test prompt for tokenization."

# Create TokenDagger tokenizer
print("Creating TokenDagger tokenizer...")
try:
    vocab_data = create_test_vocab()
    special_tokens = create_special_tokens()
    
    tokenizer = tokendagger.create_tokenizer(
        name="test_tokenizer",
        pattern=r"[a-zA-Z]+|\s+|[0-9]+|[^\w\s]",  # NEW - includes \s+ for spaces
        vocab=vocab_data,
        special_tokens=special_tokens
    )
    
    print(f"✓ TokenDagger tokenizer created successfully!")
    print(f"  Vocabulary size: {tokenizer.n_vocab}")
    print(f"  Special tokens: {list(tokenizer.special_tokens_set)}")
    
except Exception as e:
    print(f"✗ Failed to create tokenizer: {e}")
    sys.exit(1)

####################################################################################

# Test messages
user_message = "Please list the top 3 programming languages in 2024."
assistant_message = "Here are the top 3 programming languages in 2024:\n\n1. **Python**: Widely used for AI/ML\n2. **JavaScript**: Essential for web development\n3. **TypeScript**: Like JS, but with types."

# Use the ACTUAL special tokens (not the chat template format)
formatted_prompt = f"<|begin_of_text|>{user_message}<|eot|>{assistant_message}<|eot|><|end_of_text|>"
incorrectly_formatted_prompt = f"<<begin_of_text|>{user_message}<<eot|>{assistant_message}<<eot|><<end_of_text|>"

# Shorter test text for initial testing
test_prompt = "What are the main differences between Python and JavaScript?"

print("\n" + "="*80)
print("TOKENIZATION TEST")
print("="*80)

# Test 1: Basic encoding
print(f"\n1. Testing basic encoding:")
print(f"   Text: '{test_prompt[:50]}...'")

try:
    t0 = time.time()
    tokens = tokenizer.encode_ordinary(test_prompt)
    t1 = time.time()
    
    print(f"   ✓ Time taken: {(t1 - t0) * 1000:.2f}ms")
    print(f"   ✓ Token count: {len(tokens)}")
    print(f"   ✓ First 10 tokens: {tokens[:10]}")
    
    # Test decoding
    decoded = tokenizer.decode(tokens)
    print(f"   ✓ Decoded matches: {decoded == test_prompt}")
    print(decoded)
    
except Exception as e:
    print(f"   ✗ Encoding failed: {e}")

####################################################################################

# Test 2: Special token handling
print(f"\n2. Testing special token handling:")
print(f"   Text: '{formatted_prompt[:50]}...'")

try:
    # Test with special tokens allowed
    t0 = time.time()
    tokens_with_special = tokenizer.encode(
        formatted_prompt, 
        allowed_special={"<|begin_of_text|>", "<|eot|>", "<|end_of_text|>"}
    )
    t1 = time.time()
    
    print(f"   ✓ Time taken: {(t1 - t0) * 1000:.2f}ms")
    print(f"   ✓ Token count with special: {len(tokens_with_special)}")
    print(f"   ✓ First 10 tokens: {tokens_with_special[:10]}")
    
    # Test decoding
    decoded_special = tokenizer.decode(tokens_with_special)
    print(f"   ✓ Contains special tokens: {'<|begin_of_text|>' in decoded_special}")
    
except Exception as e:
    print(f"   ✗ Special token encoding failed: {e}")

####################################################################################

# Test 3: Error handling for disallowed special tokens
print(f"\n3. Testing disallowed special token handling:")

try:
    # This should raise an error
    tokens_error = tokenizer.encode(formatted_prompt)  # No special tokens allowed by default
    print(f"   ✗ Should have raised an error but didn't!")
    
except ValueError as e:
    print(f"   ✓ Correctly caught disallowed special token: {str(e)[:100]}...")
    
except Exception as e:
    print(f"   ✗ Unexpected error: {e}")

####################################################################################

# Test 4: Performance comparison (if we have a longer text)
if len(prompt) > 100:
    print(f"\n4. Testing with longer text ({len(prompt)} characters):")
    
    try:
        t0 = time.time()
        long_tokens = tokenizer.encode_ordinary(prompt[:1000])  # First 1000 chars to avoid issues
        t1 = time.time()
        
        print(f"   ✓ Time taken: {(t1 - t0) * 1000:.2f}ms")
        print(f"   ✓ Token count: {len(long_tokens)}")
        print(f"   ✓ Tokens per character: {len(long_tokens) / 1000:.2f}")
        
    except Exception as e:
        print(f"   ✗ Long text encoding failed: {e}")

####################################################################################

print(f"\n5. Tokenizer information:")
print(f"   Name: {tokenizer.name}")
print(f"   Vocabulary size: {tokenizer.n_vocab}")
print(f"   Max token value: {tokenizer.max_token_value}")
print(f"   Special tokens: {len(tokenizer.special_tokens_set)}")

# Show some token examples
if len(tokens) > 0:
    print(f"\n6. Token examples:")
    for i, token_id in enumerate(tokens[:5]):
        try:
            decoded_token = tokenizer.decode([token_id])
            print(f"   Token {token_id}: '{decoded_token}'")
        except:
            print(f"   Token {token_id}: <decode error>")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
