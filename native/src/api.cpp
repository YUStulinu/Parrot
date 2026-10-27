// The C API's engine part: models, scoring, conversation, voice and live audio.
#include <chrono>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>

#include "acoustic.hpp"
#include "api_util.hpp"
#include "asr.hpp"
#include "audio.hpp"
#include "g2p.hpp"
#include "llm.hpp"
#include "parrot.h"
#include "pronounce.hpp"
#include "tts.hpp"
#include "tutor.hpp"

using namespace parrot;
using nlohmann::json;
namespace fs = std::filesystem;

struct parrot_engine {
    fs::path dir;
    int threads = 0;
    std::unique_ptr<G2P> g2p;
    std::unique_ptr<PhoneMap> phones;
    std::unique_ptr<Asr> asr;
    std::unique_ptr<Acoustic> acoustic;
    std::unique_ptr<Scorer> scorer;
    std::unique_ptr<Tts> tts;
    std::unique_ptr<Llm> llm;
    std::string llm_path;
    std::mutex llm_mu;  // the model loads on first use (seconds), once
    Audio audio;
};

namespace {

fs::path model_path(const parrot_engine* e, const json& cfg, const char* key, const char* fallback) {
    std::string f = cfg.value(key, fallback);
    if (f.empty()) return {};
    fs::path p(f);
    return p.is_absolute() ? p : e->dir / p;
}

Llm& tutor_model(parrot_engine* e) {
    std::lock_guard<std::mutex> lock(e->llm_mu);
    if (!e->llm) {
        if (e->llm_path.empty() || !fs::exists(e->llm_path)) throw std::runtime_error("no language model installed");
        e->llm = std::make_unique<Llm>(e->llm_path, 4096, e->threads);
    }
    return *e->llm;
}

}  // namespace

extern "C" {

PARROT_API parrot_engine* parrot_open(const char* config_json) {
    try {
        json cfg = json::parse(config_json ? config_json : "{}");
        auto e = std::make_unique<parrot_engine>();
        e->dir = cfg.value("models", ".");
        e->threads = cfg.value("threads", 0);
        e->g2p = std::make_unique<G2P>(G2P::load((e->dir / cfg.value("dictionary", "cmudict.dict")).string()));
        e->phones = std::make_unique<PhoneMap>(PhoneMap::load((e->dir / cfg.value("vocab", "acoustic.vocab.json")).string()));
        api::set_vocab(*e->phones);
        if (auto p = model_path(e.get(), cfg, "whisper", "ggml-base.en-q5_1.bin"); !p.empty() && fs::exists(p))
            e->asr = std::make_unique<Asr>(p.string(), e->threads);
        if (auto p = model_path(e.get(), cfg, "acoustic", "acoustic-int8.onnx"); !p.empty() && fs::exists(p))
            e->acoustic = std::make_unique<Acoustic>(p, e->threads);
        if (auto p = model_path(e.get(), cfg, "scorer", "scorer.onnx"); !p.empty() && fs::exists(p))
            e->scorer = std::make_unique<Scorer>(p, e->threads);
        if (auto p = model_path(e.get(), cfg, "voice", "en_US-ljspeech-medium.onnx"); !p.empty() && fs::exists(p))
            e->tts = std::make_unique<Tts>(p, fs::path(p.string() + ".json"), *e->g2p, e->threads);
        if (auto p = model_path(e.get(), cfg, "llm", ""); !p.empty()) e->llm_path = p.string();
        return e.release();
    } catch (const std::exception& ex) {
        api::last_error = ex.what();
        return nullptr;
    }
}

PARROT_API void parrot_close(parrot_engine* e) { delete e; }

PARROT_API char* parrot_transcribe(parrot_engine* e, const float* audio, int samples) {
    return api::guard_str([&] {
        if (!e->asr) throw std::runtime_error("no speech model installed");
        Transcript t = e->asr->transcribe(audio, samples);
        json words = json::array();
        for (auto& w : t.words) words.push_back({{"text", w.text}, {"start", w.start}, {"end", w.end}, {"p", w.p}});
        return api::dup(json{{"text", t.text}, {"words", words}, {"ms", t.ms}}.dump());
    });
}

PARROT_API char* parrot_score(parrot_engine* e, const float* audio, int samples, const char* text) {
    return api::guard_str([&] {
        if (!e->acoustic || !e->scorer) throw std::runtime_error("the pronunciation models are not installed");
        Pronunciation p = score_pronunciation(*e->acoustic, *e->scorer, *e->phones, *e->g2p, audio, samples,
                                              spell_numbers(text ? text : ""));
        json words = json::array();
        size_t k = 0;
        for (size_t w = 0; w < p.words.size(); ++w) {
            json phones = json::array();
            for (; k < p.segments.size() && p.word_of[k] == static_cast<int>(w); ++k) {
                const auto& s = p.segments[k];
                phones.push_back({{"phone", kPhoneNames[s.phone]}, {"expected", kPhoneNames[s.phone]}, {"heard", kPhoneNames[s.best]},
                                  {"score", std::round(p.phone_scores[k] * 50.f)}, {"start", s.start * 0.02}, {"end", s.end * 0.02}});
            }
            words.push_back({{"text", p.words[w].text}, {"guessed", p.words[w].guessed},
                             {"score", w < p.word_scores.size() ? std::round(p.word_scores[w] * 10.f) : 0.f}, {"phones", phones}});
        }
        return api::dup(json{{"words", words}, {"score", std::round(p.utterance * 10.f)}, {"ms", p.ms}}.dump());
    });
}

PARROT_API char* parrot_chat(parrot_engine* e, const char* request_json) {
    return api::guard_str([&] {
        json r = json::parse(request_json);
        TutorRequest req;
        req.level = r.value("level", "B1");
        req.topic = r.value("topic", "");
        req.said = r.value("said", "");
        for (auto& t : r.value("history", json::array())) req.history.push_back({t.value("role", ""), t.value("text", "")});
        TutorReply out = tutor_turn(tutor_model(e), req);
        json cs = json::array();
        for (auto& c : out.corrections)
            cs.push_back({{"original", c.original}, {"corrected", c.corrected}, {"explanation", c.explanation}, {"kind", c.kind}});
        return api::dup(json{{"reply", out.reply}, {"corrections", cs}, {"dropped", out.dropped}, {"ms", out.ms}}.dump());
    });
}

PARROT_API int parrot_voice_rate(parrot_engine* e) { return e->tts ? e->tts->sample_rate() : 0; }

PARROT_API int parrot_speak(parrot_engine* e, const char* text, int16_t** pcm, int* samples) {
    return api::guard([&] {
        if (!e->tts) throw std::runtime_error("no voice installed");
        auto out = e->tts->speak(text ? text : "");
        *samples = static_cast<int>(out.size());
        *pcm = new int16_t[out.size()];
        std::memcpy(*pcm, out.data(), out.size() * sizeof(int16_t));
        return 0;
    });
}

PARROT_API void parrot_free_pcm(int16_t* pcm) { delete[] pcm; }

PARROT_API char* parrot_phonemes(parrot_engine* e, const char* text) {
    return api::guard_str([&] {
        json words = json::array();
        for (auto& w : e->g2p->words(spell_numbers(text ? text : ""))) words.push_back({{"text", w.text}, {"phones", w.phones}, {"guessed", w.guessed}});
        return api::dup(json{{"words", words}}.dump());
    });
}

PARROT_API int parrot_listen_start(parrot_engine* e, int silence_ms) {
    return api::guard([&] {
        e->audio.listen_start(silence_ms > 0 ? silence_ms : 700);
        return 0;
    });
}

PARROT_API void parrot_listen_stop(parrot_engine* e) { e->audio.listen_stop(); }

PARROT_API float parrot_listen_level(parrot_engine* e, int* speaking) {
    bool s = false;
    float l = e->audio.level(s);
    if (speaking) *speaking = s ? 1 : 0;
    return l;
}

PARROT_API int parrot_listen_take(parrot_engine* e, float** audio, int* samples) {
    std::vector<float> u;
    if (!e->audio.take(u)) return 0;
    *samples = static_cast<int>(u.size());
    *audio = new float[u.size()];
    std::memcpy(*audio, u.data(), u.size() * sizeof(float));
    return 1;
}

PARROT_API void parrot_free_audio(float* audio) { delete[] audio; }

PARROT_API int parrot_play(parrot_engine* e, const int16_t* pcm, int samples, int rate) {
    return api::guard([&] {
        e->audio.play(pcm, samples, rate);
        return 0;
    });
}

PARROT_API int parrot_playing(parrot_engine* e) { return e->audio.playing() ? 1 : 0; }

}  // extern "C"
