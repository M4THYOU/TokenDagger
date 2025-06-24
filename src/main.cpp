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
            for (int token : result) {
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
        item.token_bytes = std::vector<unsigned char>(token_str.begin(), token_str.end());
        special_vocab.push_back(item);
    }

    // Create the BPE tokenizer and initialize it
    tokenizer.bpe = std::make_unique<tiktoken::CoreBPE>(pattern_str, vocab, special_vocab);
}


void Tokenize(const Llama4Tokenizer& tokenizer, const std::string& prompt) {
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<int> tokens = tokenizer.encode(prompt);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    printf("Tokenization took %lld μs\n", duration.count());
    printf("Token count: %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        printf("%d\n", tokens[i]);
    }

//     std::vector<int> times;
//     for (int i = 0; i < 1000; i++) {
//         auto start_time = std::chrono::high_resolution_clock::now();
//         std::vector<int> _tokens = tokenizer.encode(prompt);
//         auto end_time = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
//         times.push_back(duration.count());
//     }

//     printf("Average tokenization time: %lld μs\n", std::accumulate(times.begin(), times.end(), 0LL) / times.size());
//     auto min_time = *std::min_element(times.begin(), times.end());
//     auto max_time = *std::max_element(times.begin(), times.end());
//     printf("Min tokenization time: %lld μs\n", min_time);
//     printf("Max tokenization time: %lld μs\n", max_time);

//     // auto start_time = std::chrono::high_resolution_clock::now();
//     // std::vector<int> tokens = tokenizer.encode(prompt);
//     // auto end_time = std::chrono::high_resolution_clock::now();
//     // auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
//     // printf("Tokenization took %lld μs\n", duration.count());
}

int main() {
    // build the tokenizer.
    std::string tokenizer_path = "/home/ubuntu/fs1-kikashi/TokenDagger/src/tokenizer_config.json";
    std::string bpe_file_path = "/home/ubuntu/fs1-kikashi/TokenDagger/src/tokenizer.model";
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

    Tokenize(tokenizer, prompt2);


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
