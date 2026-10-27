// The 39 phonemes of American English (ARPAbet, as the CMU dictionary and speechocean762 write them)
// and how the acoustic model's IPA tokens map onto them.
//
// The acoustic model (wav2vec2 fine-tuned on espeak phonemes in many languages) outputs ~390 IPA
// tokens. Its probabilities are folded into 39 phonemes, a blank (CTC) and "anything else": a token
// that covers two phonemes ("ɑːɹ" is AA and R, "aɪɚ" is AY and ER) counts for both. Forced
// alignment and the pronunciation features work in that folded space, the same for training
// (Python, through the C API) and for the app.
#pragma once

#include <array>
#include <string>
#include <vector>

namespace parrot {

constexpr int kPhones = 39;
constexpr int kBlank = kPhones;      // index of the CTC blank in folded posteriors
constexpr int kOther = kPhones + 1;  // tokens that are no English phoneme
constexpr int kClasses = kPhones + 2;

extern const std::array<const char*, kPhones> kPhoneNames;  // "AA", "AE", ..., "ZH"

// "AH0" -> index of AH; -1 for an unknown name.
int phone_index(const std::string& arpabet);

// The model's vocabulary, folded: for each token, the phonemes it counts for.
class PhoneMap {
public:
    // `vocab_json`: the model's vocab.json ({"token": id, ...}).
    static PhoneMap load(const std::string& vocab_json);
    // From token strings in id order (for tests).
    static PhoneMap from_tokens(const std::vector<std::string>& tokens, const std::string& blank = "<pad>");

    int vocab_size() const { return static_cast<int>(token_phones_.size()); }
    const std::vector<int>& phones_of(int token) const { return token_phones_[token]; }
    int blank_token() const { return blank_; }

private:
    std::vector<std::vector<int>> token_phones_;  // per token: phoneme indices, or {kOther}, or {kBlank}
    int blank_ = 0;
};

}  // namespace parrot
