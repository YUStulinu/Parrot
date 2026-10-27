#include "asr.hpp"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <whisper.h>

namespace parrot {

Asr::Asr(const std::string& model, int threads)
    : threads_(threads > 0 ? threads : static_cast<int>(std::max(2u, std::thread::hardware_concurrency() / 2))) {
    // The libraries log every load to stderr; PARROT_VERBOSE=1 keeps it.
    if (!std::getenv("PARROT_VERBOSE")) whisper_log_set([](ggml_log_level, const char*, void*) {}, nullptr);
    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = true;  // used only when built with a GPU backend (PARROT_CUDA); the CPU otherwise
    ctx_ = whisper_init_from_file_with_params(model.c_str(), cp);
    if (!ctx_) throw std::runtime_error("cannot load the speech model " + model);
}

Asr::~Asr() {
    if (ctx_) whisper_free(ctx_);
}

Transcript Asr::transcribe(const float* audio, int samples, const std::string& prompt) {
    auto t0 = std::chrono::steady_clock::now();
    whisper_full_params p = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    p.n_threads = threads_;
    p.language = "en";
    p.translate = false;
    p.no_context = true;
    p.single_segment = false;
    p.print_progress = false;
    p.print_realtime = false;
    p.print_timestamps = false;
    p.print_special = false;
    p.token_timestamps = true;
    p.suppress_blank = true;
    // Keep the learner's own words: no "uh"-free polishing beyond what the model does anyway.
    if (!prompt.empty()) p.initial_prompt = prompt.c_str();
    // Whisper wants at least a second; pad short utterances with silence.
    std::vector<float> buf(audio, audio + samples);
    if (buf.size() < 16000) buf.resize(16000 + 800, 0.f);
    if (whisper_full(ctx_, p, buf.data(), static_cast<int>(buf.size())) != 0) throw std::runtime_error("speech recognition failed");

    Transcript out;
    const whisper_token eot = whisper_token_eot(ctx_);
    for (int s = 0; s < whisper_full_n_segments(ctx_); ++s) {
        out.text += whisper_full_get_segment_text(ctx_, s);
        // Words from tokens: a token starting with a space starts a word.
        for (int k = 0; k < whisper_full_n_tokens(ctx_, s); ++k) {
            whisper_token_data d = whisper_full_get_token_data(ctx_, s, k);
            if (d.id >= eot) continue;  // special and timestamp tokens
            std::string t = whisper_full_get_token_text(ctx_, s, k);
            bool starts = !t.empty() && t[0] == ' ';
            if (starts || out.words.empty()) {
                AsrWord w;
                w.text = starts ? t.substr(1) : t;
                w.start = static_cast<float>(d.t0) / 100.f;
                w.end = static_cast<float>(d.t1) / 100.f;
                w.p = d.p;
                out.words.push_back(w);
            } else {
                AsrWord& w = out.words.back();
                w.text += t;
                w.end = static_cast<float>(d.t1) / 100.f;
                w.p = std::min(w.p, d.p);  // a word is as sure as its least sure piece
            }
        }
    }
    // Leading space of the first segment.
    out.text.erase(0, out.text.find_first_not_of(' '));
    out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace parrot
