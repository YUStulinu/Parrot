// Minimal WAV reading (16-bit or float, any rate, mono or stereo -> 16 kHz mono) and writing.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace wav {

inline std::vector<float> read16k(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (d.size() < 44 || std::memcmp(d.data(), "RIFF", 4) || std::memcmp(d.data() + 8, "WAVE", 4)) throw std::runtime_error(path + ": not WAV");
    uint16_t fmt = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    const char* data = nullptr;
    uint32_t size = 0;
    for (size_t p = 12; p + 8 <= d.size();) {
        uint32_t len;
        std::memcpy(&len, d.data() + p + 4, 4);
        if (!std::memcmp(d.data() + p, "fmt ", 4)) {
            std::memcpy(&fmt, d.data() + p + 8, 2);
            std::memcpy(&channels, d.data() + p + 10, 2);
            std::memcpy(&rate, d.data() + p + 12, 4);
            std::memcpy(&bits, d.data() + p + 22, 2);
        } else if (!std::memcmp(d.data() + p, "data", 4)) {
            data = d.data() + p + 8;
            size = static_cast<uint32_t>(std::min<size_t>(len, d.size() - p - 8));
        }
        p += 8 + len + (len & 1);
    }
    if (!data || !channels) throw std::runtime_error(path + ": no audio");
    std::vector<float> mono;
    size_t frames = size / (bits / 8) / channels;
    for (size_t i = 0; i < frames; ++i) {
        float s = 0;
        for (int c = 0; c < channels; ++c) {
            size_t k = i * channels + c;
            if (fmt == 3 && bits == 32) { float v; std::memcpy(&v, data + 4 * k, 4); s += v; }
            else if (bits == 16) { int16_t v; std::memcpy(&v, data + 2 * k, 2); s += v / 32768.f; }
            else throw std::runtime_error(path + ": 16-bit PCM or 32-bit float only");
        }
        mono.push_back(s / channels);
    }
    if (rate == 16000) return mono;
    // Linear resampling is enough for speech models' input.
    std::vector<float> out(static_cast<size_t>(mono.size() * 16000.0 / rate));
    for (size_t i = 0; i < out.size(); ++i) {
        double x = i * static_cast<double>(rate) / 16000.0;
        size_t a = static_cast<size_t>(x);
        double t = x - a;
        out[i] = static_cast<float>(mono[std::min(a, mono.size() - 1)] * (1 - t) + mono[std::min(a + 1, mono.size() - 1)] * t);
    }
    return out;
}

inline void write16(const std::string& path, const int16_t* pcm, size_t n, uint32_t rate) {
    std::ofstream f(path, std::ios::binary);
    auto u32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto u16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };
    f.write("RIFF", 4); u32(static_cast<uint32_t>(36 + n * 2)); f.write("WAVE", 4);
    f.write("fmt ", 4); u32(16); u16(1); u16(1); u32(rate); u32(rate * 2); u16(2); u16(16);
    f.write("data", 4); u32(static_cast<uint32_t>(n * 2));
    f.write(reinterpret_cast<const char*>(pcm), static_cast<std::streamsize>(n * 2));
}

}  // namespace wav
