#include "acoustic.hpp"

#include <array>
#include <stdexcept>
#include <thread>

namespace parrot {

Ort::Env& ort_env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "parrot");
    return env;
}

std::unique_ptr<Ort::Session> ort_session(const std::filesystem::path& model, int threads) {
    if (!std::filesystem::exists(model)) throw std::runtime_error("model not found: " + model.string());
    Ort::SessionOptions opt;
    opt.SetIntraOpNumThreads(threads > 0 ? threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency() / 2)));
    opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return std::make_unique<Ort::Session>(ort_env(), model.native().c_str(), opt);
}

Acoustic::Acoustic(const std::filesystem::path& model, int threads) : session_(ort_session(model, threads)) {}

std::vector<float> Acoustic::logprobs(const float* audio, int samples, int& frames, int& vocab) {
    if (samples < 1600) throw std::runtime_error("too little audio (under 0.1 s)");
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> shape{1, samples};
    auto input = Ort::Value::CreateTensor<float>(mem, const_cast<float*>(audio), static_cast<size_t>(samples), shape.data(), 2);
    const char* in_names[] = {"audio"};
    const char* out_names[] = {"logprobs"};
    auto out = session_->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    auto info = out[0].GetTensorTypeAndShapeInfo().GetShape();
    frames = static_cast<int>(info[1]);
    vocab = static_cast<int>(info[2]);
    const float* p = out[0].GetTensorData<float>();
    return std::vector<float>(p, p + static_cast<size_t>(frames) * vocab);
}

Scorer::Scorer(const std::filesystem::path& model, int threads) : session_(ort_session(model, threads)) {}

Scorer::Result Scorer::score(const std::vector<float>& features, int n_features, const std::vector<int>& phones,
                             const std::vector<int>& words) {
    const int64_t n = static_cast<int64_t>(phones.size());
    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 3> fshape{1, n, n_features};
    std::array<int64_t, 2> pshape{1, n};
    std::vector<int64_t> ph(phones.begin(), phones.end()), wd(words.begin(), words.end());
    std::vector<float> f = features;
    std::array<Ort::Value, 3> inputs = {
        Ort::Value::CreateTensor<float>(mem, f.data(), f.size(), fshape.data(), 3),
        Ort::Value::CreateTensor<int64_t>(mem, ph.data(), ph.size(), pshape.data(), 2),
        Ort::Value::CreateTensor<int64_t>(mem, wd.data(), wd.size(), pshape.data(), 2),
    };
    const char* in_names[] = {"features", "phones", "words"};
    const char* out_names[] = {"phone_scores", "word_scores", "utterance_score"};
    auto out = session_->Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 3, out_names, 3);
    Result r;
    const float* p = out[0].GetTensorData<float>();
    r.phones.assign(p, p + n);
    const float* w = out[1].GetTensorData<float>();
    int n_words = words.empty() ? 0 : *std::max_element(words.begin(), words.end()) + 1;
    r.words.assign(w, w + n_words);
    r.utterance = *out[2].GetTensorData<float>();
    return r;
}

}  // namespace parrot
