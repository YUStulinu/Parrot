// onnxruntime sessions, shared by the phoneme model, the scorer and the voice.
#pragma once

#include <filesystem>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>

namespace parrot {

Ort::Env& ort_env();

// A session for a model file, on the CPU, with `threads` intra-op threads (0: onnxruntime's choice).
std::unique_ptr<Ort::Session> ort_session(const std::filesystem::path& model, int threads);

}  // namespace parrot
