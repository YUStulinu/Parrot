// Forced alignment of what you said against what you were meant to say, and the features of each
// phoneme that the pronunciation scorer reads.
//
// 1. The acoustic model's per-frame log probabilities (20 ms frames) are folded into the 39
//    phonemes, the blank and "other" (phones.hpp).
// 2. CTC Viterbi alignment of the expected phoneme sequence: each phoneme gets the stretch of frames
//    that best explains it, given the order. Unlike free recognition, this never invents a phoneme,
//    so a mispronounced one shows as a low probability where it should be.
// 3. Per phoneme, "goodness of pronunciation" features: the mean log probability of the expected
//    phoneme over its frames (GOP), how far the best competitor is (LPR), the mean log probability of
//    every phoneme (which one it sounded like), and duration. Frames are weighted by how much they
//    are not blank, and read without the blank: CTC models spend most frames on it.
#pragma once

#include <vector>

#include "phones.hpp"

namespace parrot {

// Features per phoneme: kClasses mean log-posteriors, then GOP, LPR, the share of frames where the
// expected phoneme is the best class, duration (seconds) and log duration.
constexpr int kFeatures = kClasses + 5;

struct Segment {
    int phone = 0;     // index into kPhoneNames
    int start = 0;     // first frame
    int end = 0;       // one past the last frame
    float gop = 0;     // mean log P(expected)
    float lpr = 0;     // gop minus the best competing phoneme's mean log P
    int best = 0;      // the phoneme it sounded most like
};

// Folds log-softmax logits (T x V) into T x kClasses log-posteriors.
std::vector<float> fold(const PhoneMap& map, const float* logprobs, int frames, int vocab);

// Aligns `phones` to folded posteriors; fills features (n x kFeatures). Returns false if the audio
// is too short for the phonemes (fewer frames than phonemes).
bool align(const std::vector<float>& folded, int frames, const std::vector<int>& phones, std::vector<Segment>& segments,
           std::vector<float>& features, float frame_seconds = 0.02f);

}  // namespace parrot
