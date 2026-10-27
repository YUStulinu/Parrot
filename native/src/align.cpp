#include "align.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace parrot {

namespace {
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

float log_add(float a, float b) {
    if (a == kNegInf) return b;
    if (b == kNegInf) return a;
    float m = std::max(a, b);
    return m + std::log1p(std::exp(-std::fabs(a - b)));
}
}  // namespace

std::vector<float> fold(const PhoneMap& map, const float* logprobs, int frames, int vocab) {
    std::vector<float> out(static_cast<size_t>(frames) * kClasses, kNegInf);
    for (int t = 0; t < frames; ++t) {
        const float* row = logprobs + static_cast<size_t>(t) * vocab;
        float* o = out.data() + static_cast<size_t>(t) * kClasses;
        for (int v = 0; v < vocab && v < map.vocab_size(); ++v)
            for (int c : map.phones_of(v)) o[c] = log_add(o[c], row[v]);
        // A token counted for two phonemes adds mass twice: normalize back to a distribution.
        float z = kNegInf;
        for (int c = 0; c < kClasses; ++c) z = log_add(z, o[c]);
        for (int c = 0; c < kClasses; ++c) o[c] = (o[c] == kNegInf) ? -1e4f : o[c] - z;
    }
    return out;
}

bool align(const std::vector<float>& folded, int frames, const std::vector<int>& phones, std::vector<Segment>& segments,
           std::vector<float>& features, float frame_seconds) {
    const int n = static_cast<int>(phones.size());
    segments.clear();
    features.clear();
    if (n == 0 || frames < n) return false;
    auto lp = [&](int t, int c) { return folded[static_cast<size_t>(t) * kClasses + c]; };

    // CTC states: blank, p0, blank, p1, ..., p(n-1), blank.
    const int S = 2 * n + 1;
    auto label = [&](int s) { return (s % 2 == 0) ? kBlank : phones[s / 2]; };
    std::vector<float> score(static_cast<size_t>(frames) * S, kNegInf);
    std::vector<int> from(static_cast<size_t>(frames) * S, -1);
    auto at = [&](int t, int s) -> float& { return score[static_cast<size_t>(t) * S + s]; };
    at(0, 0) = lp(0, kBlank);
    at(0, 1) = lp(0, label(1));
    for (int t = 1; t < frames; ++t) {
        for (int s = 0; s < S; ++s) {
            float best = at(t - 1, s);
            int arg = s;
            if (s >= 1 && at(t - 1, s - 1) > best) {
                best = at(t - 1, s - 1);
                arg = s - 1;
            }
            // Skipping a blank is allowed between two different phonemes.
            if (s >= 2 && s % 2 == 1 && label(s) != label(s - 2) && at(t - 1, s - 2) > best) {
                best = at(t - 1, s - 2);
                arg = s - 2;
            }
            if (best == kNegInf) continue;
            at(t, s) = best + lp(t, label(s));
            from[static_cast<size_t>(t) * S + s] = arg;
        }
    }
    int s = (at(frames - 1, S - 1) >= at(frames - 1, S - 2)) ? S - 1 : S - 2;
    if (at(frames - 1, s) == kNegInf) return false;
    std::vector<int> state(frames);
    for (int t = frames - 1; t >= 0; --t) {
        state[t] = s;
        if (t > 0) s = from[static_cast<size_t>(t) * S + s];
    }

    // Each phoneme's frames: those in its state, plus the blanks up to the next phoneme (CTC puts a
    // phoneme's peak on a few frames and blanks around it; the sound lasts the whole stretch).
    segments.resize(n);
    std::vector<int> first(n, -1), last(n, -1);
    for (int t = 0; t < frames; ++t) {
        if (state[t] % 2 == 1) {
            int k = state[t] / 2;
            if (first[k] < 0) first[k] = t;
            last[k] = t;
        }
    }
    for (int k = 0; k < n; ++k) {
        int start = first[k];
        int end = (k + 1 < n) ? first[k + 1] : frames;
        if (start < 0) start = (k > 0) ? segments[k - 1].end : 0;
        if (end <= start) end = start + 1;
        segments[k] = Segment{phones[k], start, end, 0, 0, phones[k]};
    }

    features.assign(static_cast<size_t>(n) * kFeatures, 0.f);
    for (int k = 0; k < n; ++k) {
        Segment& seg = segments[k];
        float* f = features.data() + static_cast<size_t>(k) * kFeatures;
        int len = seg.end - seg.start;
        int wins = 0;
        // CTC spends most frames on the blank. A phoneme's evidence is in the frames that are not
        // blank: each frame's distribution without the blank, weighted by how little blank it has.
        float weight = 0;
        for (int t = seg.start; t < seg.end; ++t) {
            float pb = std::exp(lp(t, kBlank));
            float w = std::max(1.f - pb, 1e-3f);
            float norm = std::log(std::max(1.f - pb, 1e-6f));
            weight += w;
            int best_phone = 0;
            for (int c = 0; c < kClasses; ++c) {
                if (c == kBlank) {
                    f[c] += lp(t, c) / static_cast<float>(len);  // the plain mean, for the blank itself
                    continue;
                }
                f[c] += w * (lp(t, c) - norm);
                if (c < kPhones && lp(t, c) > lp(t, best_phone)) best_phone = c;
            }
            if (best_phone == seg.phone && pb < 0.5f) ++wins;
        }
        for (int c = 0; c < kClasses; ++c)
            if (c != kBlank) f[c] /= weight;
        seg.gop = f[seg.phone];
        float competitor = -1e9f;
        for (int c = 0; c < kPhones; ++c) {
            if (c != seg.phone && f[c] > competitor) {
                competitor = f[c];
                seg.best = c;
            }
        }
        if (seg.gop >= competitor) seg.best = seg.phone;
        seg.lpr = seg.gop - competitor;
        float dur = len * frame_seconds;
        f[kClasses + 0] = seg.gop;
        f[kClasses + 1] = seg.lpr;
        f[kClasses + 2] = static_cast<float>(wins) / static_cast<float>(len);  // share of frames it clearly won
        f[kClasses + 3] = dur;
        f[kClasses + 4] = std::log(dur);
    }
    return true;
}

}  // namespace parrot
