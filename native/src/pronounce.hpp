// Pronunciation scoring end to end: the text's expected phonemes (G2P), the phoneme model, forced
// alignment, the scorer.
#pragma once

#include <string>
#include <vector>

#include "acoustic.hpp"
#include "align.hpp"
#include "g2p.hpp"
#include "phones.hpp"

namespace parrot {

struct Pronunciation {
    std::vector<Word> words;
    std::vector<Segment> segments;   // one per expected phoneme, in order
    std::vector<int> word_of;        // the word of each phoneme
    std::vector<float> phone_scores; // 0–2
    std::vector<float> word_scores;  // 0–10
    float utterance = 0;             // 0–10
    double ms = 0;
};

Pronunciation score_pronunciation(Acoustic& acoustic, Scorer& scorer, const PhoneMap& map, const G2P& g2p, const float* audio,
                                  int samples, const std::string& text);

}  // namespace parrot
