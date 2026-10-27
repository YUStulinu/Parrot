// The Scorer is implemented with the Acoustic model in acoustic.cpp (both are onnxruntime sessions);
// this file holds the pieces of pronunciation scoring that sit between them.
#include "pronounce.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace parrot {

Pronunciation score_pronunciation(Acoustic& acoustic, Scorer& scorer, const PhoneMap& map, const G2P& g2p, const float* audio,
                                  int samples, const std::string& text) {
    auto t0 = std::chrono::steady_clock::now();
    Pronunciation out;
    out.words = g2p.words(text);
    std::vector<int> phones, word_of;
    for (size_t w = 0; w < out.words.size(); ++w)
        for (const auto& p : out.words[w].phones) {
            int i = phone_index(p);
            if (i < 0) continue;
            phones.push_back(i);
            word_of.push_back(static_cast<int>(w));
        }
    if (phones.empty()) throw std::runtime_error("nothing to pronounce in the text");
    int frames = 0, vocab = 0;
    auto lp = acoustic.logprobs(audio, samples, frames, vocab);
    auto folded = fold(map, lp.data(), frames, vocab);
    std::vector<float> features;
    if (!align(folded, frames, phones, out.segments, features)) throw std::runtime_error("the recording is too short for the text");
    auto s = scorer.score(features, kFeatures, phones, word_of);
    out.phone_scores = s.phones;
    out.word_scores = s.words;
    out.utterance = s.utterance;
    out.word_of = word_of;
    out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace parrot
