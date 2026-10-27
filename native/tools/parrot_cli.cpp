// parrot-cli: the engine from the command line (what the app does, for testing and evaluation).
//
//   parrot-cli --models DIR [--llm FILE] transcribe in.wav
//   parrot-cli --models DIR score in.wav "the sentence that was read"
//   parrot-cli --models DIR say "text" out.wav
//   parrot-cli --models DIR phonemes "text"
//   parrot-cli --models DIR --llm FILE chat '{"level":"B1","said":"..."}'
//   parrot-cli --models DIR listen            (the microphone: utterances, transcribed, until Ctrl+C)
//
// Every command prints JSON.
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <vector>

#include "parrot.h"
#include "wav.hpp"

static int fail(const char* what) {
    std::fprintf(stderr, "error: %s\n", what);
    return 1;
}

static int print(char* s) {
    if (!s) return fail(parrot_last_error());
    std::printf("%s\n", s);
    parrot_free(s);
    return 0;
}

int main(int argc, char** argv) {
    std::string models = ".", llm, whisper, acoustic;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--models" && i + 1 < argc) models = argv[++i];
        else if (a == "--llm" && i + 1 < argc) llm = argv[++i];
        else if (a == "--whisper" && i + 1 < argc) whisper = argv[++i];
        else if (a == "--acoustic" && i + 1 < argc) acoustic = argv[++i];
        else args.push_back(a);
    }
    if (args.empty()) return fail("usage: parrot-cli --models DIR transcribe|score|say|phonemes|chat|listen ...");
    nlohmann::json cfg = {{"models", models}};
    if (!llm.empty()) cfg["llm"] = llm;
    if (!whisper.empty()) cfg["whisper"] = whisper;
    if (!acoustic.empty()) cfg["acoustic"] = acoustic;
    parrot_engine* e = parrot_open(cfg.dump().c_str());
    if (!e) return fail(parrot_last_error());
    const std::string& cmd = args[0];
    int rc = 0;
    try {
        if (cmd == "transcribe" && args.size() == 2) {
            auto a = wav::read16k(args[1]);
            rc = print(parrot_transcribe(e, a.data(), static_cast<int>(a.size())));
        } else if (cmd == "score" && args.size() == 3) {
            auto a = wav::read16k(args[1]);
            rc = print(parrot_score(e, a.data(), static_cast<int>(a.size()), args[2].c_str()));
        } else if (cmd == "say" && args.size() == 3) {
            int16_t* pcm = nullptr;
            int n = 0;
            auto t0 = std::chrono::steady_clock::now();
            if (parrot_speak(e, args[1].c_str(), &pcm, &n) != 0) return fail(parrot_last_error());
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            wav::write16(args[2], pcm, static_cast<size_t>(n), static_cast<uint32_t>(parrot_voice_rate(e)));
            std::printf("{\"samples\":%d,\"rate\":%d,\"seconds\":%.2f,\"ms\":%.0f}\n", n, parrot_voice_rate(e),
                        n / static_cast<double>(parrot_voice_rate(e)), ms);
            parrot_free_pcm(pcm);
        } else if (cmd == "phonemes" && args.size() == 2) {
            rc = print(parrot_phonemes(e, args[1].c_str()));
        } else if (cmd == "chat" && args.size() == 2) {
            rc = print(parrot_chat(e, args[1].c_str()));
        } else if (cmd == "listen") {
            if (parrot_listen_start(e, 700) != 0) return fail(parrot_last_error());
            std::fprintf(stderr, "listening (Ctrl+C to stop)\n");
            for (;;) {
                float* a = nullptr;
                int n = 0;
                if (parrot_listen_take(e, &a, &n)) {
                    rc = print(parrot_transcribe(e, a, n));
                    parrot_free_audio(a);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        } else {
            rc = fail("unknown command or wrong arguments");
        }
    } catch (const std::exception& ex) {
        rc = fail(ex.what());
    }
    parrot_close(e);
    return rc;
}
