#include "phones.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace parrot {

const std::array<const char*, kPhones> kPhoneNames = {
    "AA", "AE", "AH", "AO", "AW", "AY", "B",  "CH", "D",  "DH", "EH", "ER", "EY",
    "F",  "G",  "HH", "IH", "IY", "JH", "K",  "L",  "M",  "N",  "NG", "OW", "OY",
    "P",  "R",  "S",  "SH", "T",  "TH", "UH", "UW", "V",  "W",  "Y",  "Z",  "ZH",
};

int phone_index(const std::string& arpabet) {
    std::string base;
    for (char c : arpabet)
        if (std::isalpha(static_cast<unsigned char>(c))) base += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    for (int i = 0; i < kPhones; ++i)
        if (base == kPhoneNames[i]) return i;
    return -1;
}

namespace {

// IPA tokens (as espeak writes English) and the phonemes they are. Several tokens for one
// phoneme are the variants espeak uses (a flapped t is still T or D; an unstressed vowel is ə).
const std::multimap<std::string, std::vector<const char*>>& ipa_table() {
    static const std::multimap<std::string, std::vector<const char*>> t = {
        {"ɑː", {"AA"}}, {"ɑ", {"AA"}}, {"a", {"AE"}}, {"æ", {"AE"}},
        {"ʌ", {"AH"}}, {"ə", {"AH"}}, {"ɐ", {"AH"}}, {"ᵻ", {"IH"}},
        {"ɔː", {"AO"}}, {"ɔ", {"AO"}}, {"ɒ", {"AA"}},
        {"aʊ", {"AW"}}, {"aɪ", {"AY"}}, {"b", {"B"}}, {"tʃ", {"CH"}}, {"d", {"D"}}, {"ð", {"DH"}},
        {"ɛ", {"EH"}}, {"e", {"EH"}}, {"ɚ", {"ER"}}, {"ɜː", {"ER"}}, {"ɜ", {"ER"}}, {"eɪ", {"EY"}},
        {"f", {"F"}}, {"ɡ", {"G"}}, {"h", {"HH"}}, {"ɪ", {"IH"}}, {"i", {"IY"}}, {"iː", {"IY"}},
        {"dʒ", {"JH"}}, {"k", {"K"}}, {"l", {"L"}}, {"ɫ", {"L"}}, {"m", {"M"}}, {"n", {"N"}}, {"n̩", {"N"}},
        {"ŋ", {"NG"}}, {"oʊ", {"OW"}}, {"o", {"OW"}}, {"əʊ", {"OW"}}, {"ɔɪ", {"OY"}}, {"p", {"P"}},
        {"ɹ", {"R"}}, {"r", {"R"}}, {"s", {"S"}}, {"ʃ", {"SH"}}, {"t", {"T"}}, {"ɾ", {"T"}}, {"ʔ", {"T"}},
        {"θ", {"TH"}}, {"ʊ", {"UH"}}, {"uː", {"UW"}}, {"u", {"UW"}}, {"v", {"V"}}, {"w", {"W"}},
        {"j", {"Y"}}, {"z", {"Z"}}, {"ʒ", {"ZH"}},
        // Tokens that span two phonemes.
        {"əl", {"AH", "L"}}, {"l̩", {"L"}}, {"ɑːɹ", {"AA", "R"}}, {"ɔːɹ", {"AO", "R"}}, {"oːɹ", {"AO", "R"}},
        {"ɪɹ", {"IH", "R"}}, {"ɛɹ", {"EH", "R"}}, {"ʊɹ", {"UH", "R"}}, {"aɪɚ", {"AY", "ER"}}, {"aɪə", {"AY", "AH"}},
        {"iə", {"IY", "AH"}}, {"ju", {"Y", "UW"}},
    };
    return t;
}

}  // namespace

PhoneMap PhoneMap::from_tokens(const std::vector<std::string>& tokens, const std::string& blank) {
    PhoneMap m;
    m.token_phones_.resize(tokens.size());
    const auto& table = ipa_table();
    for (size_t id = 0; id < tokens.size(); ++id) {
        const std::string& tok = tokens[id];
        auto& out = m.token_phones_[id];
        if (tok == blank) {
            out = {kBlank};
            m.blank_ = static_cast<int>(id);
            continue;
        }
        auto it = table.find(tok);
        if (it == table.end()) {
            out = {kOther};
            continue;
        }
        for (const char* p : it->second) out.push_back(phone_index(p));
    }
    return m;
}

PhoneMap PhoneMap::load(const std::string& vocab_json) {
    std::ifstream f(vocab_json, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + vocab_json);
    nlohmann::json j = nlohmann::json::parse(f);
    std::vector<std::string> tokens(j.size());
    for (auto it = j.begin(); it != j.end(); ++it) {
        int id = it.value().get<int>();
        if (id < 0 || id >= static_cast<int>(tokens.size())) throw std::runtime_error("bad token id in " + vocab_json);
        tokens[id] = it.key();
    }
    return from_tokens(tokens);
}

}  // namespace parrot
