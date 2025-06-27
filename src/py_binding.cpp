#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include "tiktoken/tiktoken.hpp"
// Include your tokenizer headers here

PYBIND11_MODULE(_tokendagger_core, m) {
    m.doc() = "TokenDagger low-level C++ bindings for tiktoken";
    
    // Bind VocabItem struct
    pybind11::class_<VocabItem>(m, "VocabItem")
        .def(pybind11::init<>())
        .def_readwrite("rank", &VocabItem::rank)
        .def_readwrite("token_bytes", &VocabItem::token_bytes)
        .def_readwrite("token_string", &VocabItem::token_string);
    
    // Bind TiktokenError exception
    pybind11::register_exception<tiktoken::TiktokenError>(m, "TiktokenError");
    
    // Bind CoreBPE class
    pybind11::class_<tiktoken::CoreBPE>(m, "CoreBPE")
        .def(pybind11::init<const std::string&, const std::vector<VocabItem>&, const std::vector<VocabItem>&>(),
             "Initialize CoreBPE with pattern, vocabulary, and special vocabulary",
             pybind11::arg("pattern"), pybind11::arg("vocab"), pybind11::arg("special_vocab"))
        .def("encode_ordinary", [](tiktoken::CoreBPE& self, const std::string& text) {
            pybind11::gil_scoped_release release;
            return self.encode_ordinary(text);
        }, "Encode text using ordinary BPE tokens only",
           pybind11::arg("text"))
        .def("encode", [](tiktoken::CoreBPE& self, const std::string& text, const std::set<std::string>& allowed_special) {
            // Convert std::set to emhash8::HashSet
            pybind11::gil_scoped_release release;
            emhash8::HashSet<std::string> allowed_set;
            for (const auto& token : allowed_special) {
                allowed_set.insert(token);
            }
            return self.encode(text, allowed_set);
        }, "Encode text with allowed special tokens",
           pybind11::arg("text"), pybind11::arg("allowed_special"))
        .def("decode_bytes", [](tiktoken::CoreBPE& self, const std::vector<int>& tokens) {
            pybind11::gil_scoped_release release;
            return self.decode_bytes(tokens);
        }, "Decode tokens back to bytes",
           pybind11::arg("tokens"))
        .def("special_tokens", &tiktoken::CoreBPE::special_tokens,
             "Get list of special tokens")
        .def("encode_with_special_tokens", &tiktoken::CoreBPE::encode_with_special_tokens,
             "Encode text including special tokens",
             pybind11::arg("text"));
    
    // Add other functions as needed
}
