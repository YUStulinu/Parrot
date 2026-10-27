/* Parrot's C API: what the Flutter app calls through dart:ffi and the training code through ctypes.
 *
 * Every function is thread-compatible (one engine per thread), returns 0 on success, and on failure
 * a negative code with a message in parrot_last_error(). Strings are UTF-8; JSON in and out.
 */
#ifndef PARROT_H
#define PARROT_H

#include <stdint.h>

#ifdef _WIN32
#define PARROT_API __declspec(dllexport)
#else
#define PARROT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct parrot_engine parrot_engine;

/* The last error of this thread. */
PARROT_API const char* parrot_last_error(void);

/* Frees a string the library returned. */
PARROT_API void parrot_free(char* s);

/* --- alignment and features (used by training, and inside scoring) -------------------------- */

/* Loads the acoustic model's vocabulary (vocab.json); once per process. */
PARROT_API int parrot_load_vocab(const char* vocab_json);

/* Folds T x V log-probabilities, aligns the phonemes (space-separated ARPAbet, "W IY1 K AO1 L") and
 * writes n x parrot_feature_count() features and n x 3 ints (phone index, start frame, end frame).
 * `n` is the number of phonemes. */
PARROT_API int parrot_feature_count(void);
PARROT_API int parrot_align_features(const float* logprobs, int frames, int vocab, const char* phones, float* features,
                                     int32_t* segments);

/* --- the engine: models loaded once ------------------------------------------------------------ */

/* `config_json`: {"models": dir, "whisper": file, "acoustic": file, "scorer": file, "voice": file,
 * "llm": file (optional), "threads": n}. Missing optional models leave their feature off. */
PARROT_API parrot_engine* parrot_open(const char* config_json);
PARROT_API void parrot_close(parrot_engine* e);

/* Transcribes 16 kHz mono float audio. Returns {"text", "words":[{"text","start","end","p"}], "ms"}. */
PARROT_API char* parrot_transcribe(parrot_engine* e, const float* audio, int samples);

/* Scores how `text` was pronounced in the audio: {"words":[{"text","score","phones":[{"phone",
 * "expected","heard","score","start","end"}]}], "score", "ms"}. Scores are 0–100. */
PARROT_API char* parrot_score(parrot_engine* e, const float* audio, int samples, const char* text);

/* The tutor's turn: `request_json` {"level":"B1","topic":..., "history":[{"role","text"}],
 * "said": the learner's last utterance}. Returns {"reply", "corrections":[{"original","corrected",
 * "explanation","kind"}], "ms"}. */
PARROT_API char* parrot_chat(parrot_engine* e, const char* request_json);

/* Speaks `text`: 16-bit mono PCM at parrot_voice_rate() written to a buffer the caller frees with
 * parrot_free_pcm. */
PARROT_API int parrot_voice_rate(parrot_engine* e);
PARROT_API int parrot_speak(parrot_engine* e, const char* text, int16_t** pcm, int* samples);
PARROT_API void parrot_free_pcm(int16_t* pcm);

/* The phonemes the dictionary expects for `text`: {"words":[{"text","phones":[...],"guessed"}]}. */
PARROT_API char* parrot_phonemes(parrot_engine* e, const char* text);

/* --- live audio --------------------------------------------------------------------------------- */

/* Starts the microphone (16 kHz mono). Utterances are cut by voice activity: speech, then
 * `silence_ms` of quiet. */
PARROT_API int parrot_listen_start(parrot_engine* e, int silence_ms);
PARROT_API void parrot_listen_stop(parrot_engine* e);
/* The input level (0–1) and whether speech is going on, for the UI. */
PARROT_API float parrot_listen_level(parrot_engine* e, int* speaking);
/* Takes the next finished utterance, if any: 1 and the samples (free with parrot_free_audio), or 0. */
PARROT_API int parrot_listen_take(parrot_engine* e, float** audio, int* samples);
PARROT_API void parrot_free_audio(float* audio);

/* Plays 16-bit mono PCM (non-blocking; a new call replaces what is playing). */
PARROT_API int parrot_play(parrot_engine* e, const int16_t* pcm, int samples, int rate);
PARROT_API int parrot_playing(parrot_engine* e);

#ifdef __cplusplus
}
#endif

#endif
