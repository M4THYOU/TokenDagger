#pragma once

#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>
#include "hash_set8.hpp"
#include "hash_table8.hpp"

struct VocabItem {
    int rank;
    std::vector<unsigned char> token_bytes;
    std::string token_string;
};

namespace tiktoken {

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
        emhash8::HashMap<std::string, int> special_encoder;
        pcre2_code* regex_pattern = nullptr;
        pcre2_match_data* match_data = nullptr;

    public:
        CoreBPE(const std::string& pattern, const std::vector<VocabItem>& vocab, const std::vector<VocabItem>& special_vocab) {
            // Reserve space for better performance
            encoder.reserve(vocab.size()*1.5);
            for (const auto& item : vocab) {
                encoder.emplace_unique(item.token_bytes, item.rank);  // Use emplace_unique for better performance
            }
            special_encoder.reserve(special_vocab.size()*1.5);
            for (const auto& item : special_vocab) {
                special_encoder.emplace_unique(item.token_string, item.rank);
            }
            init_regex(pattern);
        }
        
        ~CoreBPE() { 
            if (regex_pattern) {
                pcre2_code_free_8(regex_pattern);
            }
            if (match_data) {
                pcre2_match_data_free(match_data);
            }
        }
        
        // BPE-specific methods
        std::vector<int> encode_ordinary(const std::string& text) const;
        std::pair<std::vector<int>, int> encode(const std::string& text, const emhash8::HashSet<std::string>& allowed_special) const;
        std::vector<unsigned char> decode(const std::vector<int>& tokens) const;
        
    private:
        // [start_offset, end_offset). Caller is responsible for ensuring that end_offset is <= text.length().
        std::vector<std::string> split_text(const std::string& text, const size_t start_offset, const size_t end_offset) const;
        std::pair<size_t, std::string> find_next_special_token(const std::string& text, size_t start_pos) const;
        bool init_regex(const std::string& pattern);
    };

    // Function declarations - updated to use emhash
    void byte_pair_encode(const std::vector<unsigned char>& piece, 
                         const emhash8::HashMap<std::vector<unsigned char>, int, VectorHashEmhash>& encoder, 
                         std::vector<int>& result);
}
