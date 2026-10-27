// The partner's voice: Piper (VITS, ONNX) with the LJSpeech voice (public domain), 22.05 kHz.
//
// Piper voices read espeak's IPA. Instead of espeak (GPL), the text goes through the same CMU
// dictionary as pronunciation scoring, and its ARPAbet is written as the IPA espeak would produce
// for American English, with the stress marks. Numbers are spelled out first.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "g2p.hpp"
#include "onnx.hpp"

namespace parrot {

// "I have 25 cats." -> "I have twenty five cats."
std::string spell_numbers(const std::string& text);

// ARPAbet with stress -> IPA in espeak's style ("AH0" -> "ə", "AH1" -> "ˈʌ").
std::string arpabet_to_ipa(const std::vector<std::string>& phones);

class Tts {
public:
    Tts(const std::filesystem::path& voice, const std::filesystem::path& config, const G2P& g2p, int threads);
    int sample_rate() const { return rate_; }
    // The IPA the voice will read (for tests and the CLI).
    std::string ipa(const std::string& text) const;
    std::vector<int16_t> speak(const std::string& text);

private:
    std::vector<int64_t> ids(const std::string& ipa) const;
    std::unique_ptr<Ort::Session> session_;
    const G2P& g2p_;
    std::unordered_map<uint32_t, int64_t> id_of_;
    int rate_ = 22050;
    float noise_ = 0.667f, length_ = 1.f, noise_w_ = 0.8f;
};

}  // namespace parrot
