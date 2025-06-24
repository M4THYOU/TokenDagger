#include "tiktoken.hpp"
#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

namespace tiktoken {

    bool CoreBPE::init_regex(const std::string& pattern) {
        int error_number;
        PCRE2_SIZE error_offset;
        // Enable UTF-8 mode and Unicode properties for proper Unicode support
        regex_pattern = pcre2_compile_8(
            (PCRE2_SPTR8)pattern.c_str(), 
            PCRE2_ZERO_TERMINATED, 
            PCRE2_UTF | PCRE2_UCP,  // Enable UTF-8 and Unicode character properties
            &error_number, 
            &error_offset, 
            NULL
        );
        if (regex_pattern == nullptr) {
            printf("ERROR: PCRE2 compilation failed: %d at offset %zu\n", error_number, (size_t)error_offset);
            return false;
        }
        if (pcre2_jit_compile_8(regex_pattern, PCRE2_JIT_COMPLETE) < 0) {
            printf("ERROR: PCRE2 built without JIT, expect it to be slow\n");
        }
        match_data = pcre2_match_data_create_from_pattern(regex_pattern, nullptr);

        return true;
    }

    std::vector<std::string> CoreBPE::split_text(const std::string& text, const size_t start, const size_t end_offset) const {
        std::vector<std::string> pieces;

        if (regex_pattern == nullptr) {
            printf("ERROR: Pattern not compiled\n");
            return pieces;
        }

        PCRE2_SIZE start_offset = start;
        while (start_offset < end_offset) {
            int rc = pcre2_match(
                regex_pattern,
                (PCRE2_SPTR8)text.data(), end_offset,
                start_offset,
                PCRE2_NO_UTF_CHECK | PCRE2_NOTEMPTY,   // for performance
                match_data,
                nullptr);
            
            if (rc < 0) {
                if (rc == PCRE2_ERROR_NOMATCH) {
                    // No more matches, add remaining text if any
                    if (start_offset < end_offset) {
                        pieces.push_back(text.substr(start_offset));
                    }
                    break;
                } else {
                    // Handle other errors
                    break;
                }
            }
            
            // Get match information
            PCRE2_SIZE* ovector = pcre2_get_ovector_pointer(match_data);
            PCRE2_SIZE match_start = ovector[0];
            PCRE2_SIZE match_end = ovector[1];
            
            // Add the matched piece
            if (match_start < match_end) {  // TODO: might need to include start offset here.
                pieces.push_back(text.substr(match_start, match_end - match_start));
            }
            
            // Move to next position
            start_offset = match_end;
            
            // Avoid infinite loop if no progress
            if (match_start == match_end) {
                start_offset++;
            }
        }
        
        return pieces;
    }

    std::pair<size_t, std::string> CoreBPE::find_next_special_token(const std::string& text, size_t start_pos) const {
        size_t min_pos = std::string::npos;
        std::string found_token;
        for (const auto& [token_str, rank] : special_encoder) {
            size_t pos = text.find(token_str, start_pos);
            if (pos != std::string::npos && (min_pos == std::string::npos || pos < min_pos)) {
                min_pos = pos;
                found_token = token_str;
            }
        }
        return {min_pos, found_token};
    }

    std::vector<int> CoreBPE::encode_ordinary(const std::string& text) const {
        std::vector<int> result;
        // Split text using regex
        auto pieces = split_text(text, 0, text.length());
        for (const auto& piece : pieces) {
            std::vector<unsigned char> bytes(piece.begin(), piece.end());
            byte_pair_encode(bytes, encoder, result);
        }
        
        return result;
    }

    std::pair<std::vector<int>, int> CoreBPE::encode(const std::string& text, const emhash8::HashSet<std::string>& allowed_special) const {
        std::vector<int> result;
        result.reserve(text.length());

        int start = 0;
        int last_piece_token_len = 0; // number of tokens in the last piece (i.e. last regex match).
        std::string next_special_token;
        while (true) {
            // find the next allowed special token.
            next_special_token = "";
            int next_special_token_pos = std::string::npos;
            int start_offset = start;
            while (true) {
                auto [pos, token] = find_next_special_token(text, start_offset);
                if (pos == std::string::npos) { // not found, no more special tokens.
                    break;
                }
                if (allowed_special.find(token) == allowed_special.end()) {
                    // not allowed, continue searching.
                    start_offset = pos + token.length();
                    continue;
                }
                // found an allowed special token.
                next_special_token = token;
                next_special_token_pos = pos;
                break;
            }
            // printf("next_special_token: %s, next_special_token_pos: %d\n", next_special_token.c_str(), next_special_token_pos);
            int end = next_special_token_pos == std::string::npos ? text.length() : next_special_token_pos;

            // printf("start: %d, end: %d\n", start, end);
            
            // now process the text normally, up until the found special token (or the end of the text).
            auto pieces = split_text(text, start, end);
            int prev_result_size = result.size();
            for (const auto& piece : pieces) {
                prev_result_size = result.size();
                std::vector<unsigned char> bytes(piece.begin(), piece.end());
                byte_pair_encode(bytes, encoder, result);
            }
            last_piece_token_len = result.size() - prev_result_size;

            // encode the special token.
            if (next_special_token != "") {
                result.push_back(special_encoder.at(next_special_token));
                // update the start position.
                start = end + next_special_token.length();
                last_piece_token_len = 0;
            } else {
                // otherwise, we must have reached the end of the text.
                break;
            }

        }
        
        return {result, last_piece_token_len};
    }

    std::vector<unsigned char> CoreBPE::decode(const std::vector<int>& tokens) const {
        // TODO: implement after encode is fully implemented.
        // std::vector<unsigned char> ret;
        // ret.reserve(tokens.size() * 2);
        // for (int token : tokens) {
        //     std::vector<unsigned char> token_bytes;
        //     auto it = decoder.find(token);
        //     if (it != decoder.end()) {
        //         token_bytes = it->second;
        //     } else {
        //         auto special_it = special_tokens_decoder.find(token);
        //         if (special_it != special_tokens_decoder.end()) {
        //             token_bytes = special_it->second;
        //         } else {
        //             throw TiktokenError("Invalid token for decoding: " + std::to_string(token));
        //         }
        //     }
        //     ret.insert(ret.end(), token_bytes.begin(), token_bytes.end());
        // }
        // return ret;
    }

    int get_rank(const std::vector<unsigned char>& piece, const emhash8::HashMap<std::vector<unsigned char>, int, VectorHashEmhash>& encoder, const std::vector<std::pair<size_t, int>>& parts, size_t idx) {
        if ((idx+3) < parts.size()) {
            // need to extract the bytes from piece. And check if they exist in the encoder.
            int start_idx = parts[idx].first;
            int end_idx = parts[idx+3].first;
            std::vector<unsigned char> pair;
            pair.reserve(end_idx-start_idx);
            for (size_t i = start_idx; i < end_idx; ++i) {
                pair.push_back(piece[i]);
            }
            auto* value_ptr = encoder.try_get(pair);
            return value_ptr ? *value_ptr : INT_MAX;
        }
        return INT_MAX;
    }

    void bpe_merge(const std::vector<unsigned char>& piece, const emhash8::HashMap<std::vector<unsigned char>, int, VectorHashEmhash>& encoder, std::vector<int>& result) {
        // TODO: allocate a large enough vector once outside of this function and reuse it.
        // (start_idx, rank)
        std::vector<std::pair<size_t, int>> parts;
        parts.reserve(piece.size()+2);

        int min_rank = INT_MAX;
        size_t min_rank_idx = 0;

        for (size_t i = 0; i < piece.size() - 1; ++i) {
            const auto &pair = std::vector<unsigned char>{piece[i], piece[i + 1]};
            auto* value_ptr = encoder.try_get(pair);
            int rank = value_ptr ? *value_ptr : INT_MAX;
            
            if (rank < min_rank) {
                min_rank = rank;
                min_rank_idx = i;
            }
            parts.push_back({i, rank});
        }
        parts.push_back({piece.size() - 1, INT_MAX});
        parts.push_back({piece.size(), INT_MAX});

        // while we have more merges to complete...
        while (min_rank != INT_MAX) {
            // do the merge.
            if (min_rank_idx > 0) {
                // i-1 get rechecked for mergeability. we take a look at the bytes existing at i-1 through i+1 (inclusive)
                parts[min_rank_idx-1].second = get_rank(piece, encoder, parts, min_rank_idx-1);
            }
            // i gets merged.
            // i+1 gets removed.
            parts[min_rank_idx].second = get_rank(piece, encoder, parts, min_rank_idx);
            parts.erase(parts.begin()+min_rank_idx+1);

            // find new min rank (our next merge).
            min_rank = INT_MAX;
            min_rank_idx = 0;
            for (size_t i = 0; i < parts.size() - 1; ++i) {
                int rank = parts[i].second;
                if (rank < min_rank) {
                    min_rank = rank;
                    min_rank_idx = i;
                }
            }
        }

        // now apply to the result.
        for (size_t i = 0; i < parts.size()-1; ++i) {
            int start_idx = parts[i].first;
            int end_idx = parts[i+1].first;

            std::vector<unsigned char> pair;
            pair.reserve(end_idx - start_idx);
            for (size_t j = start_idx; j < end_idx; ++j) {
                pair.push_back(piece[j]);
            }
            const auto* value_ptr = encoder.try_get(pair);
            if (value_ptr) {
                result.push_back(*value_ptr);
            } else {
                printf("No value found for pair: ");
                for (const auto& byte : pair) {
                    printf("%d ", byte);
                }
                printf("\n");
                result.push_back(INT_MAX);
            }
        }
    }

    // Byte pair encoding
    void byte_pair_encode(const std::vector<unsigned char>& piece, const emhash8::HashMap<std::vector<unsigned char>, int, VectorHashEmhash>& encoder, std::vector<int>& result) {
        if (piece.size() == 1) {
            result.push_back(encoder.at(piece));
            return;
        }
        // TODO: stuff
        bpe_merge(piece, encoder, result);
    }

} // namespace tiktoken
