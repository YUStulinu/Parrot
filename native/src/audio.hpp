// Live audio: the microphone at 16 kHz mono, cut into utterances by voice activity, and playback.
//
// The capture callback runs on the audio thread and must never block: it computes the frame's level
// and copies samples into a buffer under a short lock; utterances are cut there and queued.
//
// Voice activity: 30 ms frames. The noise floor follows the quietest frames (fast down, slow up);
// speech starts when three frames in a row are 12 dB over it, and ends after `silence_ms` under
// floor + 6 dB. 300 ms before the start are kept, so the first consonant is not clipped.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

namespace parrot {

// Voice activity detection on 16 kHz frames, separate from the device (testable on files).
class Vad {
public:
    explicit Vad(int silence_ms = 700) : silence_frames_(silence_ms / 30) {}
    // Feeds samples; returns finished utterances.
    std::vector<std::vector<float>> feed(const float* samples, int n);
    // Ends the stream: an utterance in progress is returned.
    std::vector<float> flush();
    float level() const { return level_; }   // last frame, 0–1 (dBFS mapped from -60..0)
    bool speaking() const { return speaking_; }

private:
    void frame(const float* f);
    static constexpr int kFrame = 480;  // 30 ms at 16 kHz
    static constexpr int kPreroll = 10; // frames
    int silence_frames_;
    std::vector<float> pending_;
    std::deque<std::vector<float>> preroll_;
    std::vector<float> current_;
    std::vector<std::vector<float>> done_;
    float floor_db_ = -50.f;
    float level_ = 0;
    int loud_run_ = 0, quiet_run_ = 0;
    bool speaking_ = false;
};

class Audio {
public:
    Audio();
    ~Audio();
    Audio(const Audio&) = delete;
    Audio& operator=(const Audio&) = delete;

    void listen_start(int silence_ms);
    void listen_stop();
    float level(bool& speaking);
    bool take(std::vector<float>& utterance);

    void play(const int16_t* pcm, int samples, int rate);
    bool playing() const;

    // Called from the device callbacks.
    void on_capture(const float* in, uint32_t frames);
    void on_playback(int16_t* out, uint32_t frames);

private:
    struct Devices;
    std::unique_ptr<Devices> dev_;
    std::mutex mu_;
    Vad vad_;
    std::deque<std::vector<float>> ready_;
    std::mutex play_mu_;
    std::vector<int16_t> play_buf_;
    size_t play_pos_ = 0;
    std::atomic<bool> listening_{false};
};

}  // namespace parrot
