#define NOMINMAX
#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#include "audio.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "miniaudio.h"

namespace parrot {

// --- voice activity -------------------------------------------------------------------------------

std::vector<std::vector<float>> Vad::feed(const float* samples, int n) {
    pending_.insert(pending_.end(), samples, samples + n);
    size_t i = 0;
    for (; i + kFrame <= pending_.size(); i += kFrame) frame(pending_.data() + i);
    pending_.erase(pending_.begin(), pending_.begin() + static_cast<long>(i));
    auto out = std::move(done_);
    done_.clear();
    return out;
}

std::vector<float> Vad::flush() {
    std::vector<float> out;
    if (speaking_) out = std::move(current_);
    current_.clear();
    speaking_ = false;
    return out;
}

void Vad::frame(const float* f) {
    double e = 0;
    for (int i = 0; i < kFrame; ++i) e += static_cast<double>(f[i]) * f[i];
    float db = 10.f * std::log10(static_cast<float>(e / kFrame) + 1e-10f);
    level_ = std::clamp((db + 60.f) / 60.f, 0.f, 1.f);
    // The floor drops at once to a quieter frame and rises slowly (1 dB a second).
    if (db < floor_db_) floor_db_ = db;
    else floor_db_ += 0.03f;
    floor_db_ = std::clamp(floor_db_, -80.f, -20.f);

    bool loud = db > floor_db_ + 12.f;
    bool quiet = db < floor_db_ + 6.f;
    if (!speaking_) {
        preroll_.emplace_back(f, f + kFrame);
        if (preroll_.size() > kPreroll) preroll_.pop_front();
        loud_run_ = loud ? loud_run_ + 1 : 0;
        if (loud_run_ >= 3) {
            speaking_ = true;
            quiet_run_ = 0;
            current_.clear();
            for (auto& p : preroll_) current_.insert(current_.end(), p.begin(), p.end());
            preroll_.clear();
        }
        return;
    }
    current_.insert(current_.end(), f, f + kFrame);
    quiet_run_ = quiet ? quiet_run_ + 1 : 0;
    if (quiet_run_ >= silence_frames_) {
        // Drop most of the trailing silence, keep 150 ms.
        size_t keep = current_.size() - static_cast<size_t>(std::max(0, quiet_run_ - 5)) * kFrame;
        current_.resize(keep);
        if (current_.size() > 16000 / 4) done_.push_back(std::move(current_));  // ignore clicks under 250 ms
        current_.clear();
        speaking_ = false;
        loud_run_ = 0;
    }
    if (current_.size() > 16000 * 30) {  // 30 s at most
        done_.push_back(std::move(current_));
        current_.clear();
        speaking_ = false;
    }
}

// --- devices --------------------------------------------------------------------------------------

struct Audio::Devices {
    ma_device capture{};
    bool capture_on = false;
    ma_device playback{};
    bool playback_on = false;
    int playback_rate = 0;
};

namespace {
void capture_cb(ma_device* d, void*, const void* in, ma_uint32 frames) {
    static_cast<Audio*>(d->pUserData)->on_capture(static_cast<const float*>(in), frames);
}
void playback_cb(ma_device* d, void* out, const void*, ma_uint32 frames) {
    static_cast<Audio*>(d->pUserData)->on_playback(static_cast<int16_t*>(out), frames);
}
}  // namespace

Audio::Audio() : dev_(std::make_unique<Devices>()) {}

Audio::~Audio() {
    listen_stop();
    if (dev_->playback_on) ma_device_uninit(&dev_->playback);
}

void Audio::listen_start(int silence_ms) {
    listen_stop();
    {
        std::lock_guard<std::mutex> lock(mu_);
        vad_ = Vad(silence_ms);
        ready_.clear();
    }
    ma_device_config cfg = ma_device_config_init(ma_device_type_capture);
    cfg.capture.format = ma_format_f32;
    cfg.capture.channels = 1;
    cfg.sampleRate = 16000;  // miniaudio resamples from the device's rate
    cfg.dataCallback = capture_cb;
    cfg.pUserData = this;
    if (ma_device_init(nullptr, &cfg, &dev_->capture) != MA_SUCCESS) throw std::runtime_error("no microphone");
    if (ma_device_start(&dev_->capture) != MA_SUCCESS) {
        ma_device_uninit(&dev_->capture);
        throw std::runtime_error("the microphone did not start");
    }
    dev_->capture_on = true;
    listening_ = true;
}

void Audio::listen_stop() {
    if (!dev_->capture_on) return;
    listening_ = false;
    ma_device_uninit(&dev_->capture);
    dev_->capture_on = false;
    std::lock_guard<std::mutex> lock(mu_);
    auto rest = vad_.flush();
    if (rest.size() > 16000 / 4) ready_.push_back(std::move(rest));
}

void Audio::on_capture(const float* in, uint32_t frames) {
    if (!listening_ || !in) return;
    std::lock_guard<std::mutex> lock(mu_);
    for (auto& u : vad_.feed(in, static_cast<int>(frames))) ready_.push_back(std::move(u));
}

float Audio::level(bool& speaking) {
    std::lock_guard<std::mutex> lock(mu_);
    speaking = vad_.speaking();
    return vad_.level();
}

bool Audio::take(std::vector<float>& utterance) {
    std::lock_guard<std::mutex> lock(mu_);
    if (ready_.empty()) return false;
    utterance = std::move(ready_.front());
    ready_.pop_front();
    return true;
}

void Audio::play(const int16_t* pcm, int samples, int rate) {
    if (dev_->playback_on && dev_->playback_rate != rate) {
        ma_device_uninit(&dev_->playback);
        dev_->playback_on = false;
    }
    {
        std::lock_guard<std::mutex> lock(play_mu_);
        play_buf_.assign(pcm, pcm + samples);
        play_pos_ = 0;
    }
    if (!dev_->playback_on) {
        ma_device_config cfg = ma_device_config_init(ma_device_type_playback);
        cfg.playback.format = ma_format_s16;
        cfg.playback.channels = 1;
        cfg.sampleRate = static_cast<ma_uint32>(rate);
        cfg.dataCallback = playback_cb;
        cfg.pUserData = this;
        if (ma_device_init(nullptr, &cfg, &dev_->playback) != MA_SUCCESS) throw std::runtime_error("no audio output");
        if (ma_device_start(&dev_->playback) != MA_SUCCESS) throw std::runtime_error("the audio output did not start");
        dev_->playback_on = true;
        dev_->playback_rate = rate;
    }
}

void Audio::on_playback(int16_t* out, uint32_t frames) {
    std::lock_guard<std::mutex> lock(play_mu_);
    size_t n = std::min<size_t>(frames, play_buf_.size() - play_pos_);
    if (n) std::memcpy(out, play_buf_.data() + play_pos_, n * sizeof(int16_t));
    std::memset(out + n, 0, (frames - n) * sizeof(int16_t));
    play_pos_ += n;
}

bool Audio::playing() const {
    auto* self = const_cast<Audio*>(this);
    std::lock_guard<std::mutex> lock(self->play_mu_);
    return play_pos_ < play_buf_.size();
}

}  // namespace parrot
