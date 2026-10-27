#include "tts.hpp"

#include <array>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace parrot {

namespace {

const char* kOnes[] = {"zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
                       "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"};
const char* kTens[] = {"", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"};

std::string say_number(long n) {
    if (n < 20) return kOnes[n];
    if (n < 100) return std::string(kTens[n / 10]) + (n % 10 ? std::string(" ") + kOnes[n % 10] : "");
    if (n < 1000) return std::string(kOnes[n / 100]) + " hundred" + (n % 100 ? " " + say_number(n % 100) : "");
    if (n < 1000000) return say_number(n / 1000) + " thousand" + (n % 1000 ? " " + say_number(n % 1000) : "");
    return say_number(n / 1000000) + " million" + (n % 1000000 ? " " + say_number(n % 1000000) : "");
}

// UTF-8 -> code points.
std::vector<uint32_t> codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        uint32_t cp;
        int len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 6) { cp = c & 0x1f; len = 2; }
        else if ((c >> 4) == 14) { cp = c & 0x0f; len = 3; }
        else { cp = c & 0x07; len = 4; }
        for (int k = 1; k < len && i + k < s.size(); ++k) cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3f);
        out.push_back(cp);
        i += len;
    }
    return out;
}

// The voice was trained on espeak's American English, which keeps "caught" (AO) apart from "cot" (AA);
// CMUdict often lists the merged AA first ("bought B AA1 T", then "B AO1 T"). The voice gets the AO one,
// else it says "Bart".
const std::vector<std::string>& voice_phones(const G2P& g2p, const Word& w) {
    const auto* all = g2p.lookup(w.key);
    if (!all || all->size() < 2) return w.phones;
    const auto& first = (*all)[0];
    for (size_t v = 1; v < all->size(); ++v) {
        const auto& alt = (*all)[v];
        if (alt.size() != first.size()) continue;
        bool merger = false, other = false;
        for (size_t i = 0; i < alt.size(); ++i) {
            if (alt[i] == first[i]) continue;
            if (first[i].rfind("AA", 0) == 0 && alt[i].rfind("AO", 0) == 0 && first[i].substr(2) == alt[i].substr(2)) merger = true;
            else other = true;
        }
        if (merger && !other) return alt;
    }
    return w.phones;
}

}  // namespace

std::string spell_numbers(const std::string& text) {
    std::string out;
    for (size_t i = 0; i < text.size();) {
        if (std::isdigit(static_cast<unsigned char>(text[i]))) {
            size_t j = i;
            while (j < text.size() && std::isdigit(static_cast<unsigned char>(text[j]))) ++j;
            std::string digits = text.substr(i, j - i);
            if (digits.size() <= 7) out += say_number(std::stol(digits));
            else for (char d : digits) out += std::string(kOnes[d - '0']) + " ";
            i = j;
        } else {
            out += text[i++];
        }
    }
    return out;
}

std::string arpabet_to_ipa(const std::vector<std::string>& phones) {
    static const std::map<std::string, std::string> base = {
        {"AA", "ɑː"}, {"AE", "æ"}, {"AO", "ɔː"}, {"AW", "aʊ"}, {"AY", "aɪ"}, {"B", "b"}, {"CH", "tʃ"}, {"D", "d"},
        {"DH", "ð"}, {"EH", "ɛ"}, {"EY", "eɪ"}, {"F", "f"}, {"G", "ɡ"}, {"HH", "h"}, {"IH", "ɪ"}, {"IY", "iː"},
        {"JH", "dʒ"}, {"K", "k"}, {"L", "l"}, {"M", "m"}, {"N", "n"}, {"NG", "ŋ"}, {"OW", "oʊ"}, {"OY", "ɔɪ"},
        {"P", "p"}, {"R", "ɹ"}, {"S", "s"}, {"SH", "ʃ"}, {"T", "t"}, {"TH", "θ"}, {"UH", "ʊ"}, {"UW", "uː"},
        {"V", "v"}, {"W", "w"}, {"Y", "j"}, {"Z", "z"}, {"ZH", "ʒ"},
    };
    std::string out;
    for (const auto& p : phones) {
        std::string name;
        char stress = 0;
        for (char c : p) {
            if (std::isdigit(static_cast<unsigned char>(c))) stress = c;
            else name += c;
        }
        std::string ipa;
        if (name == "AH") ipa = (stress == '1' || stress == '2') ? "ʌ" : "ə";
        else if (name == "ER") ipa = (stress == '1' || stress == '2') ? "ɜː" : "ɚ";
        else {
            auto it = base.find(name);
            if (it == base.end()) continue;
            ipa = it->second;
        }
        if (stress == '1') out += "ˈ";
        else if (stress == '2') out += "ˌ";
        out += ipa;
    }
    return out;
}

Tts::Tts(const std::filesystem::path& voice, const std::filesystem::path& config, const G2P& g2p, int threads)
    : session_(ort_session(voice, threads)), g2p_(g2p) {
    std::ifstream f(config, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + config.string());
    nlohmann::json c = nlohmann::json::parse(f);
    rate_ = c["audio"]["sample_rate"].get<int>();
    noise_ = c["inference"].value("noise_scale", noise_);
    length_ = c["inference"].value("length_scale", length_) * 1.08f;  // a little slower: the listener is learning
    noise_w_ = c["inference"].value("noise_w", noise_w_);
    for (auto it = c["phoneme_id_map"].begin(); it != c["phoneme_id_map"].end(); ++it) {
        auto cps = codepoints(it.key());
        if (cps.size() == 1) id_of_[cps[0]] = it.value()[0].get<int64_t>();
    }
}

std::string Tts::ipa(const std::string& text) const {
    std::string spelled = spell_numbers(text);
    std::string out;
    // Words keep the text's punctuation after them: Piper uses it for pauses and intonation.
    size_t i = 0;
    auto words = g2p_.words(spelled);
    size_t wi = 0;
    while (i < spelled.size()) {
        char c = spelled[i];
        if (std::isalnum(static_cast<unsigned char>(c))) {
            size_t j = i;
            while (j < spelled.size() && (std::isalnum(static_cast<unsigned char>(spelled[j])) || spelled[j] == '\'')) ++j;
            if (wi < words.size()) {
                if (!out.empty() && out.back() != ' ') out += ' ';
                out += arpabet_to_ipa(voice_phones(g2p_, words[wi]));
                ++wi;
            }
            i = j;
        } else {
            if (c == ',' || c == '.' || c == '?' || c == '!' || c == ';' || c == ':') out += c;
            ++i;
        }
    }
    return out;
}

std::vector<int64_t> Tts::ids(const std::string& ipa) const {
    // Piper's layout: ^ _ p1 _ p2 _ ... $
    std::vector<int64_t> out = {id_of_.at('^'), id_of_.at('_')};
    for (uint32_t cp : codepoints(ipa)) {
        auto it = id_of_.find(cp);
        if (it == id_of_.end()) continue;
        out.push_back(it->second);
        out.push_back(id_of_.at('_'));
    }
    out.push_back(id_of_.at('$'));
    return out;
}

std::vector<int16_t> Tts::speak(const std::string& text) {
    std::vector<int16_t> pcm;
    // Sentence by sentence: shorter inputs synthesize faster and more steadily.
    std::string sentence;
    auto flush = [&] {
        std::string s = ipa(sentence);
        sentence.clear();
        if (s.find_first_not_of(" ,.?!;:") == std::string::npos) return;
        auto in = ids(s);
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<int64_t, 2> shape{1, static_cast<int64_t>(in.size())};
        std::array<int64_t, 1> lens{static_cast<int64_t>(in.size())};
        std::array<int64_t, 1> lshape{1};
        std::array<float, 3> scales{noise_, length_, noise_w_};
        std::array<int64_t, 1> sshape{3};
        std::array<Ort::Value, 3> inputs = {
            Ort::Value::CreateTensor<int64_t>(mem, in.data(), in.size(), shape.data(), 2),
            Ort::Value::CreateTensor<int64_t>(mem, lens.data(), 1, lshape.data(), 1),
            Ort::Value::CreateTensor<float>(mem, scales.data(), 3, sshape.data(), 1),
        };
        const char* in_names[] = {"input", "input_lengths", "scales"};
        const char* out_names[] = {"output"};
        auto out = session_->Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 3, out_names, 1);
        size_t n = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
        const float* a = out[0].GetTensorData<float>();
        float peak = 1e-3f;
        for (size_t k = 0; k < n; ++k) peak = std::max(peak, std::fabs(a[k]));
        float gain = std::min(1.f / peak, 4.f) * 0.9f;
        for (size_t k = 0; k < n; ++k) pcm.push_back(static_cast<int16_t>(std::lround(std::clamp(a[k] * gain, -1.f, 1.f) * 32767)));
        pcm.insert(pcm.end(), static_cast<size_t>(rate_ / 5), 0);  // 200 ms between sentences
    };
    for (char c : text) {
        sentence += c;
        if (c == '.' || c == '?' || c == '!') flush();
    }
    flush();
    return pcm;
}

}  // namespace parrot
