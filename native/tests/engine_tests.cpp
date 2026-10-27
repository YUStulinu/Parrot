// Tests of the engine: voice activity, number spelling and IPA for the voice, the tutor's checks, and
// (when the models are installed) the voice read back by the recognizer and pronunciation scoring.
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "acoustic.hpp"
#include "asr.hpp"
#include "audio.hpp"
#include "g2p.hpp"
#include "phones.hpp"
#include "pronounce.hpp"
#include "tts.hpp"
#include "tutor.hpp"

using namespace parrot;
namespace fs = std::filesystem;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

const fs::path kModels = PARROT_MODELS;

// Seconds of quiet noise, then of a voiced-like signal (a 150 Hz tone with harmonics), at 16 kHz.
void append(std::vector<float>& a, double seconds, bool voice, std::mt19937& rng) {
    std::normal_distribution<float> noise(0.f, 0.002f);
    size_t n = static_cast<size_t>(seconds * 16000);
    for (size_t i = 0; i < n; ++i) {
        float s = noise(rng);
        if (voice) {
            double t = static_cast<double>(a.size()) / 16000.0;
            for (int h = 1; h <= 4; ++h) s += static_cast<float>(0.08 / h * std::sin(2 * 3.14159265358979 * 150 * h * t));
        }
        a.push_back(s);
    }
}

std::vector<float> to16k(const std::vector<int16_t>& pcm, int rate) {
    std::vector<float> out(static_cast<size_t>(pcm.size() * 16000.0 / rate));
    for (size_t i = 0; i < out.size(); ++i) {
        size_t k = std::min(pcm.size() - 1, static_cast<size_t>(i * static_cast<double>(rate) / 16000.0));
        out[i] = pcm[k] / 32768.f;
    }
    return out;
}

std::string lower_words(const std::string& s) {
    std::string out;
    for (unsigned char c : s)
        if (std::isalpha(c) || c == ' ') out += static_cast<char>(std::tolower(c));
    return out;
}
}  // namespace

static void test_vad() {
    std::mt19937 rng(7);
    std::vector<float> a;
    append(a, 1.0, false, rng);
    append(a, 0.8, true, rng);
    append(a, 1.0, false, rng);
    append(a, 0.05, true, rng);  // a click: too short to be an utterance
    append(a, 1.0, false, rng);
    append(a, 1.2, true, rng);
    append(a, 1.0, false, rng);
    Vad vad(600);
    std::vector<std::vector<float>> got;
    for (size_t i = 0; i < a.size(); i += 160) {  // 10 ms blocks, as a device delivers them
        auto u = vad.feed(a.data() + i, static_cast<int>(std::min<size_t>(160, a.size() - i)));
        got.insert(got.end(), u.begin(), u.end());
    }
    check(got.size() == 2, "two utterances, the click ignored (" + std::to_string(got.size()) + ")");
    if (got.size() == 2) {
        double d0 = got[0].size() / 16000.0, d1 = got[1].size() / 16000.0;
        check(d0 > 0.9 && d0 < 1.5, "the first keeps its preroll and some silence (" + std::to_string(d0) + " s)");
        check(d1 > 1.3 && d1 < 1.9, "the second likewise (" + std::to_string(d1) + " s)");
    }
    check(!vad.speaking(), "silence at the end");

    Vad open(600);
    std::vector<float> b;
    append(b, 0.5, false, rng);
    append(b, 1.0, true, rng);
    open.feed(b.data(), static_cast<int>(b.size()));
    check(open.speaking() && open.flush().size() > 16000, "flush returns the utterance in progress");
}

static void test_voice_text() {
    check(spell_numbers("I have 3 cats and 21 dogs") == "I have three cats and twenty one dogs", "numbers are spelled out");
    check(spell_numbers("in 1999") == "in one thousand nine hundred ninety nine", "large numbers");
    check(arpabet_to_ipa({"HH", "AH0", "L", "OW1"}) == "həlˈoʊ", "hello in IPA: " + arpabet_to_ipa({"HH", "AH0", "L", "OW1"}));
    check(arpabet_to_ipa({"B", "ER1", "D"}) == "bˈɜːd" && arpabet_to_ipa({"B", "AH1", "T", "ER0"}) == "bˈʌtɚ",
          "stressed and unstressed vowels differ");
}

static void test_tutor_checks() {
    std::string said = "Yesterday I goed to the cinema with my friends";
    auto j = [](const std::string& o, const std::string& c, const std::string& k) {
        return "{\"original\":\"" + o + "\",\"corrected\":\"" + c + "\",\"explanation\":\"x\",\"kind\":\"" + k + "\"}";
    };
    std::string empty = j("", "", "");
    TutorReply r = tutor_parse("{\"reply\":\"Oh, you went to the cinema?\",\"corrections\":{\"first\":" + j("I goed", "I went", "grammar") +
                                   ",\"second\":" + empty + ",\"third\":" + empty + "}}",
                               said);
    check(r.reply == "Oh, you went to the cinema?" && r.corrections.size() == 1 && r.dropped.empty(), "a real correction is kept");
    r = tutor_parse("{\"reply\":\"ok\",\"corrections\":{\"first\":" + j("I goes", "I go", "grammar") + ",\"second\":" +
                        j("yesterday i", "Yesterday I", "grammar") + ",\"third\":" + j("I goed", "I went", "") + "}}",
                    said);
    check(r.corrections.size() == 1 && r.corrections[0].kind == "grammar" && r.dropped.size() == 2,
          "invented quotes and case-only changes are dropped; a missing kind is grammar");
    r = tutor_parse("{\"reply\":\"ok\",\"corrections\":{\"first\":" + j("I goed", "I went", "grammar") + ",\"second\":" +
                        j("i goed", "I went", "grammar") + ",\"third\":" + empty + "}}",
                    said);
    check(r.corrections.size() == 1, "duplicates are merged");
    auto parts = split_correction({"Yesterday I goed to the cinema with my friends, it make me laugh",
                                   "Yesterday I went to the cinema with my friends, it made me laugh", "Use 'went', not 'goed'.",
                                   "grammar"});
    check(parts.size() == 2 && parts[0].original == "Yesterday I goed to the" && parts[0].corrected == "Yesterday I went to the" &&
              parts[1].original == "friends, it make me laugh" && parts[1].corrected == "friends, it made me laugh",
          "a long correction is split into its changes, with context");
    check(parts.size() == 2 && !parts[0].explanation.empty() && parts[1].explanation.empty(),
          "the explanation stays with the change it names");
    check(split_correction({"I goed", "I went", "x", "grammar"}).size() == 1, "short corrections stay as they are");
    auto missing = split_correction({"She is the most intelligent person which I know very well indeed",
                                     "She is the most intelligent person I know very well indeed", "x", "grammar"});
    check(missing.size() == 1 && missing[0].original == "intelligent person which I know" && missing[0].corrected == "intelligent person I know",
          "a removed word keeps its neighbours: " + (missing.empty() ? std::string() : missing[0].original));
    auto cont = split_correction({"I am agree", "I agree with you about the film.", "x", "grammar"});
    check(cont.size() == 1 && cont[0].original == "I am agree" && cont[0].corrected == "I agree",
          "the model continuing the sentence is not part of the fix: " + (cont.empty() ? std::string() : cont[0].corrected));
    r = tutor_parse("{\"reply\":\"ok\",\"corrections\":{\"first\":" + j("We arrived at the airport very early.", "arrived", "grammar") +
                        ",\"second\":" + j("very early", "", "grammar") + ",\"third\":" + j("I goed", "I went", "") + "}}",
                    "We arrived at the airport very early. I goed home.");
    check(r.corrections.size() == 1 && r.corrections[0].kind == "grammar" && r.dropped.size() == 2,
          "quotes and empty fixes are dropped, a missing kind becomes grammar");
    check(tutor_user({"B1", "travel", {}, ""}).find("Start the conversation") != std::string::npos, "an empty turn starts the conversation");
    check(tutor_system("A2").find("A2") != std::string::npos && tutor_system("C1").find("C1") != std::string::npos, "levels change the prompt");
}

static void test_models() {
    fs::path voice = kModels / "en_US-ljspeech-medium.onnx", whisper = kModels / "ggml-base.en-q5_1.bin";
    if (!fs::exists(voice) || !fs::exists(whisper)) {
        std::printf("skip the model tests (no models in %s)\n", kModels.string().c_str());
        return;
    }
    G2P g2p = G2P::load((kModels / "cmudict.dict").string());
    Tts tts(voice, fs::path(voice.string() + ".json"), g2p, 0);
    Asr asr(whisper.string(), 0);
    check(tts.ipa("I bought coffee.") == "ˈaɪ bˈɔːt kˈɔːfiː.", "the voice keeps caught apart from cot: " + tts.ipa("I bought coffee."));
    const std::string text = "What did you do at the weekend?";
    auto pcm = tts.speak(text);
    check(pcm.size() > static_cast<size_t>(tts.sample_rate()), "the voice speaks (" + std::to_string(pcm.size()) + " samples)");
    auto audio = to16k(pcm, tts.sample_rate());
    Transcript t = asr.transcribe(audio.data(), static_cast<int>(audio.size()));
    check(lower_words(t.text) == lower_words(text), "the recognizer reads the voice back: \"" + t.text + "\"");

    fs::path acoustic = kModels / "acoustic-int8.onnx", scorer = kModels / "scorer.onnx";
    if (!fs::exists(acoustic) || !fs::exists(scorer)) {
        std::printf("skip the scoring tests (no pronunciation models)\n");
        return;
    }
    PhoneMap map = PhoneMap::load((kModels / "acoustic.vocab.json").string());
    Acoustic am(acoustic, 0);
    Scorer sc(scorer, 0);
    Pronunciation good = score_pronunciation(am, sc, map, g2p, audio.data(), static_cast<int>(audio.size()), text);
    Pronunciation bad = score_pronunciation(am, sc, map, g2p, audio.data(), static_cast<int>(audio.size()),
                                            "Where did you go in the summer?");
    check(good.words.size() == 7 && good.word_scores.size() == 7, "one score per word");
    check(good.utterance > bad.utterance + 0.2f, "the sentence said scores higher than another one (" + std::to_string(good.utterance) +
                                                     " vs " + std::to_string(bad.utterance) + ")");
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    test_vad();
    test_voice_text();
    test_tutor_checks();
    try {
        test_models();
    } catch (const std::exception& e) {
        check(false, std::string("the model tests: ") + e.what());
    }
    std::printf("%d failure(s)\n", failures);
    return failures ? 1 : 0;
}
