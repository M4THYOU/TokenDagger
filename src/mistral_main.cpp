#include <iostream>
#include <string>
#include <cstdio>
#include <optional>
#include <vector>
#include <fstream>
#include <memory>

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
#include "tiktoken/tiktoken.hpp"  // Include the header
using VocabItem = VocabItem;

#pragma GCC diagnostic pop

struct TokenizerConfig {
    std::string pattern;
    int num_vocab_tokens;
    int default_vocab_size;
    int default_num_special_tokens;
    std::string version;
};

// Special token IDs as constants
namespace MistralSpecialTokens {
    constexpr int UNK = 0;
    constexpr int BOS = 1;
    constexpr int EOS = 2;
    constexpr int BEGIN_INST = 3;
    constexpr int END_INST = 4;
    constexpr int BEGIN_TOOLS = 5;
    constexpr int END_TOOLS = 6;
    constexpr int BEGIN_TOOL_RESULTS = 7;
    constexpr int END_TOOL_RESULTS = 8;
    constexpr int TOOL_CALLS = 9;
    constexpr int IMG = 10;
    constexpr int PAD = 11;
    constexpr int IMG_BREAK = 12;
    constexpr int IMG_END = 13;
    constexpr int PREFIX = 14;
    constexpr int MIDDLE = 15;
    constexpr int SUFFIX = 16;
    constexpr int BEGIN_SYSTEM = 17;
    constexpr int END_SYSTEM = 18;
    constexpr int BEGIN_TOOL_CONTENT = 19;
}

struct MistralTokenizer {
    TokenizerConfig config;
    std::vector<VocabItem> vocab;
    std::unique_ptr<tiktoken::CoreBPE> bpe;

    std::vector<int> encode(const std::string& prompt) const {
        std::vector<int> tokens = {MistralSpecialTokens::BOS, MistralSpecialTokens::BEGIN_INST};
        if (bpe) {
            auto result = bpe->encode_ordinary(prompt);
            for (int token : result) {
                tokens.push_back(token + config.default_num_special_tokens);
            }
        }
        tokens.push_back(MistralSpecialTokens::END_INST);
        return tokens;
    }
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(TokenizerConfig, pattern, num_vocab_tokens, default_vocab_size, default_num_special_tokens, version);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(VocabItem, rank, token_bytes, token_str);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(MistralTokenizer, config, vocab);

static std::string base64_decode(const std::string &in) {
    std::string out;
    std::vector<int> T(256,-1);
    for (int i=0; i<64; i++) T["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] = i;

    int val=0, valb=-8;
    for (unsigned char c : in) {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            out.push_back(char((val>>valb)&0xFF));
            valb -= 8;
        }
    }
    return out;
}


void LoadTokenizer(const std::string& filename, MistralTokenizer& tokenizer) {
    std::ifstream file(filename);
    nlohmann::json json_data;
    file >> json_data;
    
    // Parse config
    tokenizer.config = json_data["config"].get<TokenizerConfig>();
    
    // Parse vocab with custom handling for optional token_str
    tokenizer.vocab.clear();

    int vocab_size = tokenizer.config.default_vocab_size - tokenizer.config.default_num_special_tokens;

    for (size_t i = 0; i < vocab_size; i++) {
        const auto& vocab_item = json_data["vocab"][i];
        VocabItem item;
        item.rank = vocab_item["rank"].get<int>();
        
        std::string token_bytes_str = vocab_item["token_bytes"].get<std::string>();
        
        // Decode base64 string to binary data
        std::string decoded_bytes;
        try {
            decoded_bytes = base64_decode(token_bytes_str);
        } catch (const std::exception& e) {
            fprintf(stderr, "ERROR: Failed to decode base64 token_bytes: %s\n", e.what());
            continue; // Skip this vocab item
        }
        
        item.token_bytes = std::vector<unsigned char>(decoded_bytes.begin(), decoded_bytes.end());
        
        // Handle optional token_str field
        if (vocab_item.contains("token_str") && !vocab_item["token_str"].is_null()) {
            item.token_str = vocab_item["token_str"].get<std::string>();
        } else {
            item.token_str = ""; // Default to empty string if not present or null
        }
        
        tokenizer.vocab.push_back(item);
    }
    
    // Create the BPE tokenizer and initialize it
    tokenizer.bpe = std::make_unique<tiktoken::CoreBPE>(tokenizer.vocab);
    if (!tokenizer.bpe->init_regex(tokenizer.config.pattern)) {
        fprintf(stderr, "ERROR: Failed to initialize regex for tokenizer\n");
        tokenizer.bpe.reset(); // Clear the pointer
    }
}


void Tokenize(const MistralTokenizer& tokenizer, const std::string& prompt) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<int> tokens = tokenizer.encode(prompt);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("Tokenization took %lld μs\n", duration.count());
    // for (size_t i = 0; i < tokens.size(); i++) {
    //     printf("%d\n", tokens[i]);
    // }
    // return;

    std::vector<int> times;
    for (int i = 0; i < 1000; i++) {
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

    // auto start_time = std::chrono::high_resolution_clock::now();
    // std::vector<int> tokens = tokenizer.encode(prompt);
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    // printf("Tokenization took %lld μs\n", duration.count());
}

int main() {
    // printf("Hello, World!\n");
    // printf("Safetensors library included successfully!\n");

    // LoadModel("/home/ubuntu/mistral_models/ministral-8b-2410/consolidated.safetensors");

    // build the tokenizer.
    std::string tokenizer_path = "/home/ubuntu/mistral_models/ministral-8b-2410/tekken.json";
    MistralTokenizer tokenizer;
    LoadTokenizer(tokenizer_path, tokenizer);

    // printf("Tokenizer loaded successfully!\n");
    // printf("Tokenizer config: %s\n", tokenizer.config.version.c_str());
    // printf("Tokenizer vocab size: %zu\n", tokenizer.vocab.size());

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

    Tokenize(tokenizer, prompt);

    std::string lorem_prompt;
    std::ifstream lorem_file("./tests/input/lorem.txt");
    if (lorem_file.is_open()) {
        std::stringstream buffer;
        buffer << lorem_file.rdbuf();
        lorem_prompt = buffer.str();
        lorem_file.close();
    } else {
        printf("Error: Could not open ./tests/lorem.txt\n");
        return 1;
    }


    printf("\nLoaded lorem ipsum prompt (%zu characters)\n", lorem_prompt.length());
    Tokenize(tokenizer, lorem_prompt);


    std::string emoji_prompt;
    std::ifstream emoji_file("./tests/input/emoji.txt");
    if (emoji_file.is_open()) {
        std::stringstream buffer;
        buffer << emoji_file.rdbuf();
        emoji_prompt = buffer.str();
        emoji_file.close();
    } else {
        printf("Error: Could not open ./tests/emoji.txt\n");
        return 1;
    }

    printf("\nLoaded emoji prompt (%zu characters)\n", emoji_prompt.length());
    Tokenize(tokenizer, emoji_prompt);

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
