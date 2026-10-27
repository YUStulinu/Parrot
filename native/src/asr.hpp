// Speech recognition with whisper.cpp: an utterance in, its words with timings and confidence out.
#pragma once

#include <string>
#include <vector>

struct whisper_context;

namespace parrot {

struct AsrWord {
    std::string text;
    float start = 0, end = 0;  // seconds
    float p = 0;               // mean token probability
};

struct Transcript {
    std::string text;
    std::vector<AsrWord> words;
    double ms = 0;
};

class Asr {
public:
    Asr(const std::string& model, int threads);
    ~Asr();
    Asr(const Asr&) = delete;
    Asr& operator=(const Asr&) = delete;

    // 16 kHz mono. `prompt` nudges the vocabulary (the sentence the learner is reading, say).
    Transcript transcribe(const float* audio, int samples, const std::string& prompt = "");

private:
    whisper_context* ctx_ = nullptr;
    int threads_;
};

}  // namespace parrot
