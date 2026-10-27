// The C API's alignment part: no models, only the vocabulary.
#include <cstring>
#include <memory>
#include <sstream>
#include <string>

#include "align.hpp"
#include "api_util.hpp"
#include "parrot.h"
#include "phones.hpp"

namespace {
std::unique_ptr<parrot::PhoneMap> g_vocab;
}

namespace parrot::api {
thread_local std::string last_error;

const PhoneMap* vocab() { return g_vocab.get(); }
void set_vocab(PhoneMap m) { g_vocab = std::make_unique<PhoneMap>(std::move(m)); }

std::vector<int> parse_phones(const char* phones) {
    std::vector<int> out;
    std::istringstream in(phones ? phones : "");
    std::string p;
    while (in >> p) {
        int i = phone_index(p);
        if (i < 0) throw std::invalid_argument("unknown phoneme " + p);
        out.push_back(i);
    }
    return out;
}

char* dup(const std::string& s) {
    char* out = new char[s.size() + 1];
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}
}  // namespace parrot::api

using namespace parrot;

extern "C" {

PARROT_API const char* parrot_last_error(void) { return api::last_error.c_str(); }

PARROT_API void parrot_free(char* s) { delete[] s; }

PARROT_API int parrot_feature_count(void) { return kFeatures; }

PARROT_API int parrot_load_vocab(const char* vocab_json) {
    return api::guard([&] {
        api::set_vocab(PhoneMap::load(vocab_json));
        return 0;
    });
}

PARROT_API int parrot_align_features(const float* logprobs, int frames, int vocab, const char* phones, float* features,
                                     int32_t* segments) {
    return api::guard([&] {
        if (!api::vocab()) throw std::runtime_error("parrot_load_vocab first");
        auto ph = api::parse_phones(phones);
        auto folded = fold(*api::vocab(), logprobs, frames, vocab);
        std::vector<Segment> segs;
        std::vector<float> feats;
        if (!align(folded, frames, ph, segs, feats)) throw std::runtime_error("the audio is too short for the phonemes");
        std::memcpy(features, feats.data(), feats.size() * sizeof(float));
        for (size_t k = 0; k < segs.size(); ++k) {
            segments[3 * k] = segs[k].phone;
            segments[3 * k + 1] = segs[k].start;
            segments[3 * k + 2] = segs[k].end;
        }
        return 0;
    });
}

}  // extern "C"
