// Tests of the core: phonemes, the dictionary, folding and alignment.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "align.hpp"
#include "g2p.hpp"
#include "phones.hpp"

using namespace parrot;

namespace {
int failures = 0;
void check(bool ok, const std::string& what) {
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}

// A tiny vocabulary: blank, then a few English tokens and one foreign one.
const std::vector<std::string> kTokens = {"<pad>", "w", "iː", "k", "ɔː", "l", "ɑːɹ", "x"};

// Log-probabilities of a sequence of frames, each frame one token (prob 0.9) or a mix.
std::vector<float> frames_of(const std::vector<int>& tokens) {
    const int V = static_cast<int>(kTokens.size());
    std::vector<float> out(tokens.size() * V);
    for (size_t t = 0; t < tokens.size(); ++t)
        for (int v = 0; v < V; ++v) out[t * V + v] = std::log(v == tokens[t] ? 0.9f : 0.1f / (V - 1));
    return out;
}
}  // namespace

static void test_phone_names() {
    check(phone_index("AH0") == phone_index("AH") && phone_index("AH") >= 0, "stress digits are ignored");
    check(phone_index("XX") == -1, "unknown phonemes are -1");
    check(std::string(kPhoneNames[phone_index("ZH")]) == "ZH", "names round-trip");
}

static void test_fold() {
    auto map = PhoneMap::from_tokens(kTokens);
    check(map.blank_token() == 0, "the blank is found");
    check(map.phones_of(6).size() == 2, "a token spanning two phonemes counts for both");
    check(map.phones_of(7)[0] == kOther, "foreign tokens are 'other'");
    auto lp = frames_of({1, 2, 0});
    auto f = fold(map, lp.data(), 3, static_cast<int>(kTokens.size()));
    float sum = 0;
    for (int c = 0; c < kClasses; ++c) sum += std::exp(f[c]);
    check(std::fabs(sum - 1.f) < 1e-3f, "folded frames are distributions");
    check(f[phone_index("W")] > std::log(0.85f), "a frame of 'w' is mostly W");
}

static void test_align() {
    auto map = PhoneMap::from_tokens(kTokens);
    // "we call": w iː _ k ɔː ɔː l _  (blanks between)
    std::vector<int> tokens = {0, 1, 1, 2, 2, 0, 3, 4, 4, 4, 5, 5, 0};
    auto lp = frames_of(tokens);
    auto folded = fold(map, lp.data(), static_cast<int>(tokens.size()), static_cast<int>(kTokens.size()));
    std::vector<int> phones = {phone_index("W"), phone_index("IY"), phone_index("K"), phone_index("AO"), phone_index("L")};
    std::vector<Segment> segs;
    std::vector<float> feats;
    check(align(folded, static_cast<int>(tokens.size()), phones, segs, feats), "alignment succeeds");
    check(segs.size() == 5 && feats.size() == 5 * kFeatures, "one segment and one feature row per phoneme");
    check(segs[0].start <= 1 && segs[1].start == 3 && segs[2].start == 6 && segs[3].start == 7 && segs[4].start == 10,
          "segments start where their phoneme does");
    bool good = true;
    for (auto& s : segs) good = good && s.gop > std::log(0.5f) && s.best == s.phone && s.lpr > 0;
    check(good, "well-pronounced phonemes have a high GOP and are their own best match");

    // The same audio scored against a wrong phoneme (CALL -> "kill"): IY where AO was said.
    std::vector<int> wrong = {phone_index("W"), phone_index("IY"), phone_index("K"), phone_index("IH"), phone_index("L")};
    check(align(folded, static_cast<int>(tokens.size()), wrong, segs, feats), "alignment with a wrong expectation succeeds");
    check(segs[3].gop < -3.f && segs[3].best == phone_index("AO"), "the wrong phoneme scores low and was heard as AO");

    check(!align(folded, 3, phones, segs, feats), "too few frames for the phonemes fails");
}

static void test_g2p() {
    std::string path = std::string(PARROT_MODELS) + "/cmudict.dict";
    if (!std::filesystem::exists(path)) {
        std::printf("skip dictionary test: run scripts/fetch-deps.sh\n");
        return;
    }
    auto g = G2P::load(path);
    check(g.size() > 100000, "the CMU dictionary loads (" + std::to_string(g.size()) + " words)");
    auto w = g.words("We call it Bear, don't we?");
    check(w.size() == 6 && w[3].key == "bear" && w[4].key == "don't", "words are split, lowercased, apostrophes kept");
    std::string bear;
    for (auto& p : w[3].phones) bear += p + " ";
    check(bear == "B EH1 R ", "bear is B EH1 R (" + bear + ")");
    auto oov = g.words("Zorblax");
    check(oov.size() == 1 && oov[0].guessed && !oov[0].phones.empty(), "unknown words are guessed and marked");
}

int main() {
    test_phone_names();
    test_fold();
    test_align();
    test_g2p();
    std::printf("%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
