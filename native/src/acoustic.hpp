// The phoneme model (wav2vec2, ONNX): 16 kHz audio in, CTC log-probabilities over its IPA tokens
// out, one frame per 20 ms. The export normalizes the waveform inside the graph.
#pragma once

#include <filesystem>
#include <memory>
#include <vector>

#include "onnx.hpp"

namespace parrot {

class Acoustic {
public:
    Acoustic(const std::filesystem::path& model, int threads);
    // Log-probabilities, frames x vocab, row-major.
    std::vector<float> logprobs(const float* audio, int samples, int& frames, int& vocab);

private:
    std::unique_ptr<Ort::Session> session_;
};

// The pronunciation scorer (scorer.onnx, trained by ml/parrot_ml/train.py).
class Scorer {
public:
    Scorer(const std::filesystem::path& model, int threads);
    struct Result {
        std::vector<float> phones;  // 0–2 per phoneme
        std::vector<float> words;   // 0–10 per word
        float utterance = 0;        // 0–10
    };
    // features: n x F, phones: n phoneme indices, words: n word indices (0, 0, 1, 1, 1, 2, ...).
    Result score(const std::vector<float>& features, int n_features, const std::vector<int>& phones, const std::vector<int>& words);

private:
    std::unique_ptr<Ort::Session> session_;
};

}  // namespace parrot
