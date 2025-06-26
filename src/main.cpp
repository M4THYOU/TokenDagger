#include <iostream>
#include <string>
#include <cstdio>
#include <optional>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>

// Suppress warnings from third-party library
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

// Define this only once in your project to include the implementation
// #define SAFETENSORS_CPP_IMPLEMENTATION
// #include "safetensors-cpp/safetensors.hh"
#include "nlohmann/json.hpp"
#include "tiktoken/tiktoken.hpp"
using VocabItem = VocabItem;

#pragma GCC diagnostic pop

struct InternalSpecialToken {
    int rank;
    std::string content;
};

namespace Llama4SpecialTokens {
    const InternalSpecialToken BOS = {200000, "<|begin_of_text|>"};
    const InternalSpecialToken EOS = {200008, "<|eot|>"};
    const InternalSpecialToken FULL_EOS = {200001, "<|end_of_text|>"};
}

struct SpecialToken {
    std::string content;
};

struct TokenizerConfig {
    std::unordered_map<std::string, SpecialToken> added_tokens_decoder;
};

struct Llama4Tokenizer {
    TokenizerConfig config;
    std::unique_ptr<tiktoken::CoreBPE> bpe;

    std::vector<int> encode(const std::string& prompt) const {
        // std::vector<int> tokens = {MistralSpecialTokens::BOS, MistralSpecialTokens::BEGIN_INST};
        std::vector<int> tokens = {Llama4SpecialTokens::BOS.rank};
        tokens.reserve(prompt.size()); // some compression occurs, so shouldn't be greater than the prompt size.
        if (bpe) {
            // auto result = bpe->encode_ordinary(prompt);
            auto result = bpe->encode(prompt, {Llama4SpecialTokens::BOS.content, Llama4SpecialTokens::EOS.content, Llama4SpecialTokens::FULL_EOS.content});
            for (auto token : result.first) {
                tokens.push_back(token);
            }
        }
        // tokens.push_back(MistralSpecialTokens::END_INST);
        return tokens;
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(SpecialToken, content);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TokenizerConfig, added_tokens_decoder);


static std::vector<unsigned char> base64_decode(const std::string &in) {
    std::vector<unsigned char> out;
    std::vector<int> T(256,-1);
    for (int i=0; i<64; i++) T["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] = i;

    int val=0, valb=-8;
    for (unsigned char c : in) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back((val>>valb)&0xFF);
            valb -= 8;
        }
    }
    return out;
}


void LoadBPEFile(const std::string& filename, std::vector<VocabItem>& vocab) {
    std::ifstream file(filename);
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::string base64_token;
        int rank;
        
        if (iss >> base64_token >> rank) {
            // Decode base64 to bytes
            std::vector<unsigned char> token_bytes = base64_decode(base64_token);
            
            VocabItem item;
            item.rank = rank;
            item.token_bytes = token_bytes;
            vocab.push_back(item);
        }
    }
}

void LoadTokenizer(const std::string& tokenizer_path, const std::string& bpe_path, Llama4Tokenizer& tokenizer) {
    // hard code it for now.
    std::string pattern_str = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+";

    // load base vocab.
    std::vector<VocabItem> vocab;
    LoadBPEFile(bpe_path, vocab);

    // load special tokens.
    std::ifstream file(tokenizer_path);
    nlohmann::json json_data;
    file >> json_data;
    // Parse config
    tokenizer.config = json_data.get<TokenizerConfig>();
    std::vector<VocabItem> special_vocab;
    for (const auto& [token_str, special_token] : tokenizer.config.added_tokens_decoder) {
        VocabItem item;
        item.rank = std::stoi(token_str);
        item.token_string = special_token.content;
        item.token_bytes = std::vector<unsigned char>(special_token.content.begin(), special_token.content.end());
        special_vocab.push_back(item);
    }

    // Create the BPE tokenizer and initialize it
    tokenizer.bpe = std::make_unique<tiktoken::CoreBPE>(pattern_str, vocab, special_vocab);
}


void Tokenize(const Llama4Tokenizer& tokenizer, const std::string& prompt) {
    // Perform 5 warmup runs
    for (int i = 0; i < 5; i++) {
        tokenizer.encode(prompt);
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<int> tokens = tokenizer.encode(prompt);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("Tokenization took %lld μs\n", duration.count());
    printf("Token count: %zu\n", tokens.size());
    // for (size_t i = 0; i < tokens.size(); i++) {
    //     printf("%d\n", tokens[i]);
    // }

    // decode the tokens.
    std::vector<unsigned char> decoded_bytes = tokenizer.bpe->decode_bytes(tokens);
    // printf("Decoded bytes: %s\n", decoded_bytes.data());

    std::vector<int> times;
    for (int i = 0; i < 1000000; i++) {
        auto start_time = std::chrono::high_resolution_clock::now();
        std::vector<int> _tokens = tokenizer.encode(prompt);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        times.push_back(duration.count());
    }

    printf("Average tokenization time: %lld μs\n", std::accumulate(times.begin(), times.end(), 0LL) / times.size());
    auto min_time = *std::min_element(times.begin(), times.end());
    auto max_time = *std::max_element(times.begin(), times.end());
    printf("Min tokenization time: %lld μs\n", min_time);
    printf("Max tokenization time: %lld μs\n", max_time);

//     // auto start_time = std::chrono::high_resolution_clock::now();
//     // std::vector<int> tokens = tokenizer.encode(prompt);
//     // auto end_time = std::chrono::high_resolution_clock::now();
//     // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
//     // printf("Tokenization took %lld μs\n", duration.count());
}

int main() {
    // build the tokenizer.
    std::string tokenizer_path = "/home/ubuntu/TokenDagger/src/tokenizer_config.json";
    std::string bpe_file_path = "/home/ubuntu/TokenDagger/src/tokenizer.model";
    Llama4Tokenizer tokenizer;
    LoadTokenizer(tokenizer_path, bpe_file_path, tokenizer);

    // printf("Tokenizer loaded successfully!\n");

    std::string prompt = R"""(You are an expert urban planner and cost estimator with deep knowledge of Paris, France. I need you to provide a comprehensive analysis of what it would cost to hire professional window cleaners to clean all the windows in Paris.

Consider the following factors in your detailed estimate:
1. The total number of buildings and windows in Paris (both residential and commercial)
2. Different types of buildings (apartments, offices, shops, historical buildings, etc.)
3. The varying heights and accessibility of buildings
4. Labor costs for professional window cleaners in Paris
5. Equipment and safety requirements for high-rise buildings
6. Seasonal variations and weather considerations
7. Time estimates for completion
8. Any special considerations for historical or landmark buildings

Please provide your estimate in US Dollars, breaking down the major cost components. Also include any assumptions you're making and potential challenges that could affect the final cost.)""";

    std::string prompt2 = "<|begin_of_text|>Please list the top 3 programming languages in 2024.<|eot|>Here are the top 3 programming languages in 2024:\n\n1. **Python**: Widely used for AI/ML\n2. **JavaScript**: Essential for web development\n3. **TypeScript**: Like JS, but with types.<|eot|><|end_of_text|>";
    std::string prompt_long = "<|begin_of_text|>What are the main differences between Python and JavaScript?<|eot|>Here are the key differences between Python and JavaScript:\n\n**1. Syntax and Readability**\n- Python: Uses indentation for code blocks, very readable\n- JavaScript: Uses curly braces, more C-like syntax\n\n**2. Type System**\n- Python: Dynamically typed with optional type hints\n- JavaScript: Dynamically typed, TypeScript adds static typing\n\n**3. Primary Use Cases**\n- Python: Data science, AI/ML, backend development, automation\n- JavaScript: Web development (frontend/backend), mobile apps\n\n**4. Performance**\n- Python: Generally slower, interpreted language\n- JavaScript: V8 engine makes it quite fast for web applications<|eot|>Can you give me a concrete example of the syntax differences?<|eot|>Absolutely! Here are concrete examples showing the syntax differences:\n\n**Python Example:**\n```python\ndef calculate_factorial(n):\n    if n <= 1:\n        return 1\n    else:\n        return n * calculate_factorial(n - 1)\n\n# Usage\nresult = calculate_factorial(5)\nprint(f\"Factorial of 5 is: {result}\")\n\n# List comprehension\nnumbers = [1, 2, 3, 4, 5]\nsquares = [x**2 for x in numbers if x % 2 == 0]\nprint(squares)  # Output: [4, 16]\n```\n\n**JavaScript Example:**\n```javascript\nfunction calculateFactorial(n) {\n    if (n <= 1) {\n        return 1;\n    } else {\n        return n * calculateFactorial(n - 1);\n    }\n}\n\n// Usage\nconst result = calculateFactorial(5);\nconsole.log(`Factorial of 5 is: ${result}`);\n\n// Array methods\nconst numbers = [1, 2, 3, 4, 5];\nconst squares = numbers.filter(x => x % 2 === 0).map(x => x**2);\nconsole.log(squares); // Output: [4, 16]\n```\n\n**Key Differences Highlighted:**\n- Python uses indentation vs JavaScript's curly braces `{}`\n- Python's `def` vs JavaScript's `function`\n- Python's f-strings vs JavaScript's template literals\n- Python's list comprehensions vs JavaScript's array methods<|eot|>Which one would you recommend for a beginner programmer?<|eot|>For a beginner programmer, I'd generally recommend **Python** for these reasons:\n\n**Why Python is Great for Beginners:**\n\n1. **Readable Syntax**: Python's syntax closely resembles English, making it intuitive\n   ```python\n   if age >= 18:\n       print(\"You can vote!\")\n   ```\n\n2. **Less Boilerplate**: You can write functional programs with minimal setup\n   ```python\n   name = input(\"What's your name? \")\n   print(f\"Hello, {name}!\")\n   ```\n\n3. **Excellent Learning Resources**: Tons of tutorials, books, and courses designed for beginners\n\n4. **Versatile Applications**: Once you learn Python, you can:\n   - Build web applications (Django, Flask)\n   - Analyze data (pandas, NumPy)\n   - Create AI/ML models (TensorFlow, PyTorch)\n   - Automate tasks (scripting)\n   - Develop games (Pygame)\n\n5. **Gentle Learning Curve**: Focus on problem-solving rather than complex syntax\n\n**However, JavaScript might be better if you:**\n- Want to see immediate visual results (web pages)\n- Are specifically interested in web development\n- Prefer learning through interactive projects\n\n**My Recommendation**: Start with Python to learn programming fundamentals, then add JavaScript when you want to build web applications. This gives you a solid foundation plus practical web skills!\n\nWhat type of projects are you most interested in creating?<|eot|>I'm interested in building web applications. Should I still start with Python?<|eot|>Given your interest in web applications, this changes my recommendation! Here's what I'd suggest:\n\n**For Web Development, Consider Starting with JavaScript:**\n\n**Advantages of JavaScript-First for Web Development:**\n1. **Immediate Visual Feedback**: See your changes instantly in the browser\n2. **One Language, Full Stack**: JavaScript works for both frontend and backend (Node.js)\n3. **No Setup Required**: Just open a browser and start coding\n4. **Huge Ecosystem**: React, Vue, Angular for frontend; Express, Next.js for backend\n5. **High Demand**: Web developers are in high demand\n\n**Learning Path for Web Development:**\n```\n1. HTML + CSS (structure and styling)\n2. JavaScript fundamentals\n3. DOM manipulation\n4. Frontend framework (React recommended)\n5. Backend with Node.js/Express\n6. Database integration (MongoDB/PostgreSQL)\n```\n\n**Sample First Project** (you can build this in days):\n```html\n<!DOCTYPE html>\n<html>\n<head><title>Todo App</title></head>\n<body>\n    <h1>My Todo List</h1>\n    <input id=\"todoInput\" placeholder=\"Add a task...\">\n    <button onclick=\"addTodo()\">Add</button>\n    <ul id=\"todoList\"></ul>\n    \n    <script>\n    function addTodo() {\n        const input = document.getElementById('todoInput');\n        const list = document.getElementById('todoList');\n        const li = document.createElement('li');\n        li.textContent = input.value;\n        list.appendChild(li);\n        input.value = '';\n    }\n    </script>\n</body>\n</html>\n```\n\n**Alternative: Python for Web (Still Valid!)**\n- Django/Flask are excellent for web backends\n- Python + JavaScript frontend is a common combination\n- Many successful web companies use Python (Instagram, Spotify, Dropbox)\n\n**My Updated Recommendation**: Start with JavaScript since you want immediate web results, but don't completely ignore Python—you might use it later for backend services, data processing, or AI features in your web apps!\n\nWould you like me to suggest some specific first projects to try?<|eot|><|end_of_text|>";

    std::string edge_unicode_prompt = "αβγδεζηθικλμνξοπ αβγδεζηθικλμνξοπρστυφχψω αβγδεζηθικλμνξοπ ρστυφχψω";

    Tokenize(tokenizer, edge_unicode_prompt);


    /////////////////////////////////////////////////

    // std::string lorem_prompt;
    // std::ifstream lorem_file("./tests/input/lorem.txt");
    // if (lorem_file.is_open()) {
    //     std::stringstream buffer;
    //     buffer << lorem_file.rdbuf();
    //     lorem_prompt = buffer.str();
    //     lorem_file.close();
    // } else {
    //     printf("Error: Could not open ./tests/lorem.txt\n");
    //     return 1;
    // }


    // // printf("\nLoaded lorem ipsum prompt (%zu characters)\n", lorem_prompt.length());
    // Tokenize(tokenizer, lorem_prompt);


    // std::string emoji_prompt;
    // std::ifstream emoji_file("./tests/input/emoji.txt");
    // if (emoji_file.is_open()) {
    //     std::stringstream buffer;
    //     buffer << emoji_file.rdbuf();
    //     emoji_prompt = buffer.str();
    //     emoji_file.close();
    // } else {
    //     printf("Error: Could not open ./tests/emoji.txt\n");
    //     return 1;
    // }

    // printf("\nLoaded emoji prompt (%zu characters)\n", emoji_prompt.length());
    // Tokenize(tokenizer, emoji_prompt);

    // // tokenize the prompt.
    // auto start_time = std::chrono::high_resolution_clock::now();
    // std::vector<int> tokens = tokenizer.encode(prompt);
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // printf("Tokenization took %lld μs\n", duration.count());

    // print the tokens.
    // for (const auto& token : tokens) {
    //     printf("Token: %d\n", token);
    // }
    // for (const auto& item : tokenizer.vocab) {
    //     printf("Vocab item: %s\n", item.token_str.c_str());
    // }



    return 0;
}
