#pragma once

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

// Include emhash - you'll need to download hash_table8.hpp from https://github.com/ktprime/emhash
#include "hash_table8.hpp"

struct VocabItem {
    int rank;
    std::vector<unsigned char> token_bytes;
    std::string token_str;
};

namespace tiktoken {

    // Custom hash function for std::vector<unsigned char>
    struct VectorHash {
        std::size_t operator()(const std::vector<unsigned char>& vec) const {
            std::size_t hash = vec.size();
            for (unsigned char byte : vec) {
                hash ^= byte + 0x9e3779b9 + (hash << 6) + (hash >> 2);
            }
            return hash;
        }
    };
    struct VectorHashSimple {
        std::size_t operator()(const std::vector<unsigned char>& vec) const {
            std::size_t hash = 0;
            for (unsigned char byte : vec) {
                hash = hash * 31 + byte;  // Simple hash function
            }
            return hash;
        }
    };
    struct VectorHashEmhash {
        std::size_t operator()(const std::vector<unsigned char>& vec) const {
            std::size_t hash = 0;
            // Use a more efficient hash for emhash
            for (size_t i = 0; i < vec.size(); ++i) {
                hash = hash * 131 + vec[i];  // Prime number multiplier for better distribution
            }
            return hash;
        }
    };

    // Exception class
    class TiktokenError : public std::runtime_error {
    public:
        explicit TiktokenError(const std::string& message) : std::runtime_error(message) {}
    };

    // Core BPE implementation - using emhash8 for better performance
    class CoreBPE {
    private:
        // Replace std::unordered_map with emhash8::HashMap
        emhash8::HashMap<std::vector<unsigned char>, int, VectorHashEmhash> encoder;
        pcre2_code* regex_pattern = nullptr;
        pcre2_match_data* match_data = nullptr;

    public:
        CoreBPE(const std::vector<VocabItem>& vocab) {
            // Reserve space for better performance
            encoder.reserve(vocab.size()*1.5);
            for (const auto& item : vocab) {
                encoder.emplace_unique(item.token_bytes, item.rank);  // Use emplace_unique for better performance
            }
            // printf("encoder load factor: %f\n", encoder.load_factor());
        }
        
        ~CoreBPE() { 
            if (regex_pattern) {
                pcre2_code_free_8(regex_pattern);
            }
            if (match_data) {
                pcre2_match_data_free(match_data);
            }
        }
        
        bool init_regex(const std::string& pattern);
        
        // BPE-specific methods
        std::vector<int> encode_ordinary(const std::string& text) const;
        std::vector<int> encode(const std::string& text, const std::vector<std::string>& allowed_special) const;
        std::string decode(const std::vector<int>& tokens) const;
        
    private:
        std::vector<std::string> split_text(const std::string& text) const;
    };

    // Function declarations - updated to use emhash
    void byte_pair_encode(const std::vector<unsigned char>& piece, 
                         const emhash8::HashMap<std::vector<unsigned char>, int, VectorHashEmhash>& encoder, 
                         std::vector<int>& result);
}
