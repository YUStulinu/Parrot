"""Evaluation of the engine through its C API (the library the app uses).

  python -m parrot_ml.eval_engine asr      Whisper base and small on speechocean762 test (learners' speech): WER
  python -m parrot_ml.eval_engine voice    the voice read back by Whisper: WER, speed
  python -m parrot_ml.eval_engine tutor    corrections on typical learner mistakes and on correct sentences;
                                           how hard the replies are at each level; latency
  (--llm FILE for the tutor; results in ml/eval/engine-*.json, the tutor's named after the model)
"""

from __future__ import annotations

import argparse
import ctypes
import io
import json
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .paths import DATA, MODELS, RESULTS, ROOT, native_library


class Engine:
    def __init__(self, **config):
        self.lib = native_library(engine=True)
        L = self.lib
        L.parrot_open.restype = ctypes.c_void_p
        L.parrot_open.argtypes = [ctypes.c_char_p]
        L.parrot_last_error.restype = ctypes.c_char_p
        L.parrot_free.argtypes = [ctypes.c_void_p]
        for f in ("parrot_transcribe",):
            getattr(L, f).restype = ctypes.c_void_p
            getattr(L, f).argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
        L.parrot_chat.restype = ctypes.c_void_p
        L.parrot_chat.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        L.parrot_voice_rate.argtypes = [ctypes.c_void_p]
        L.parrot_speak.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_int16)),
                                   ctypes.POINTER(ctypes.c_int)]
        L.parrot_free_pcm.argtypes = [ctypes.c_void_p]
        L.parrot_close.argtypes = [ctypes.c_void_p]
        self.e = L.parrot_open(json.dumps({"models": str(MODELS), **config}).encode())
        if not self.e:
            raise SystemExit(L.parrot_last_error().decode())

    def _json(self, p) -> dict:
        if not p:
            raise RuntimeError(self.lib.parrot_last_error().decode())
        try:
            return json.loads(ctypes.string_at(p).decode("utf-8"))
        finally:
            self.lib.parrot_free(p)

    def transcribe(self, audio: np.ndarray) -> dict:
        a = np.ascontiguousarray(audio, dtype=np.float32)
        return self._json(self.lib.parrot_transcribe(self.e, a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(a)))

    def chat(self, request: dict) -> dict:
        return self._json(self.lib.parrot_chat(self.e, json.dumps(request).encode()))

    def speak(self, text: str) -> tuple[np.ndarray, int]:
        pcm = ctypes.POINTER(ctypes.c_int16)()
        n = ctypes.c_int()
        if self.lib.parrot_speak(self.e, text.encode(), ctypes.byref(pcm), ctypes.byref(n)) != 0:
            raise RuntimeError(self.lib.parrot_last_error().decode())
        out = np.ctypeslib.as_array(pcm, shape=(n.value,)).copy()
        self.lib.parrot_free_pcm(pcm)
        return out, self.lib.parrot_voice_rate(self.e)

    def close(self):
        self.lib.parrot_close(self.e)


# --- text measures --------------------------------------------------------------------------------

def words_of(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("-", " "))


NUMBERS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
           "8": "eight", "9": "nine", "10": "ten"}


# Whisper .en writes American spelling; a British spelling of the same word is not an error.
SPELLING = {"favourite": "favorite", "colour": "color", "centre": "center", "theatre": "theater", "travelling": "traveling",
            "realise": "realize", "organise": "organize", "neighbour": "neighbor", "programme": "program"}


def normalize(text: str) -> list[str]:
    return [SPELLING.get(w, w) for w in (NUMBERS.get(w, w) for w in words_of(text))]


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """(edits, reference length)"""
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)], len(ref)


_syllables: dict[str, int] | None = None


def syllables(word: str) -> int:
    """From CMUdict (vowels carry a stress digit), or counted vowel groups for unknown words."""
    global _syllables
    if _syllables is None:
        _syllables = {}
        d = MODELS / "cmudict.dict"
        for line in d.read_text(encoding="utf-8").splitlines() if d.exists() else []:
            parts = line.split("#")[0].split()
            if parts and "(" not in parts[0]:
                _syllables[parts[0]] = sum(p[-1].isdigit() for p in parts[1:])
    return _syllables.get(word) or max(1, len(re.findall(r"[aeiouy]+", word)))


def readability(text: str) -> dict:
    sentences = [s for s in re.split(r"[.!?]+", text) if words_of(s)]
    ws = words_of(text)
    if not ws or not sentences:
        return {"words_per_sentence": 0.0, "fk_grade": 0.0}
    syl = sum(syllables(w) for w in ws)
    wps = len(ws) / len(sentences)
    return {"words_per_sentence": wps, "fk_grade": 0.39 * wps + 11.8 * syl / len(ws) - 15.59}


# --- evaluations ----------------------------------------------------------------------------------

def eval_asr(every: int) -> dict:
    import pandas as pd

    d = pd.read_parquet(DATA / "speechocean762-test.parquet").iloc[::every]
    audio = [sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")[0] for r in d.itertuples()]
    out = {"utterances": len(d), "audio_hours": sum(len(a) for a in audio) / 16000 / 3600}
    for model in ["ggml-base.en-q5_1.bin", "ggml-small.en-q5_1.bin"]:
        e = Engine(whisper=model)
        edits = total = 0
        t0 = time.time()
        for r, a in zip(d.itertuples(), audio):
            x, n = wer(normalize(r.text), normalize(e.transcribe(a)["text"]))
            edits, total = edits + x, total + n
        seconds = time.time() - t0
        e.close()
        out[model] = {"wer": edits / total, "real_time_factor": seconds / (out["audio_hours"] * 3600)}
        print(f"{model}: WER {edits / total:.3f}, {seconds / len(d) * 1000:.0f} ms per utterance")
    return out


VOICE_SENTENCES = [
    "Hello! What did you do at the weekend?",
    "I think this is a great idea, but we should ask the others first.",
    "The weather was terrible, so we stayed at home and watched a film.",
    "Could you tell me the way to the nearest train station?",
    "She has been learning to play the piano for three years.",
    "That sounds wonderful. Did you take any photos?",
    "My favourite food is pasta with tomatoes and fresh basil.",
    "Although it was raining, they decided to go for a walk.",
    "What kind of music do you usually listen to?",
    "I bought a new bicycle last month and I ride it to work every day.",
    "The museum opens at nine in the morning and closes at six.",
    "If you could live anywhere in the world, where would you go?",
    "He was surprised by how quickly the situation changed.",
    "Let's meet in front of the library at half past four.",
    "Thank you so much for your help yesterday.",
    "Do you prefer reading books or watching films?",
    "We are planning a trip to the mountains in July.",
    "Thoroughly tested theories are rarely thrown away.",
    "Is there anything else you would like to talk about?",
    "I usually drink a cup of coffee before I start working.",
]


def eval_voice() -> dict:
    e = Engine(whisper="ggml-small.en-q5_1.bin")
    edits = total = 0
    audio_s = synth_s = 0.0
    errors = []
    for s in VOICE_SENTENCES:
        t0 = time.time()
        pcm, rate = e.speak(s)
        synth_s += time.time() - t0
        audio_s += len(pcm) / rate
        x = np.interp(np.arange(0, len(pcm), rate / 16000), np.arange(len(pcm)), pcm.astype(np.float32) / 32768)
        heard = e.transcribe(x)["text"]
        n_e, n = wer(normalize(s), normalize(heard))
        edits, total = edits + n_e, total + n
        if n_e:
            errors.append({"said": s, "heard": heard.strip()})
    e.close()
    out = {"sentences": len(VOICE_SENTENCES), "wer": edits / total, "synthesis_real_time_factor": synth_s / audio_s,
           "misheard": errors}
    print(f"voice: WER {out['wer']:.3f} (read back by Whisper small), synthesis {out['synthesis_real_time_factor']:.3f}x real time")
    for x in errors:
        print(f"  said {x['said']!r}, heard {x['heard']!r}")
    return out


def _norm(s: str) -> str:
    return " ".join(words_of(s))


def eval_tutor(llm: str) -> dict:
    cases = json.loads((RESULTS / "tutor_cases.json").read_text(encoding="utf-8"))
    e = Engine(llm=llm)
    found = n_errors = right_fix = kept = matched = 0
    ms = []
    details = []
    for c in cases["mistakes"]:
        r = e.chat({"level": "B1", "said": c["said"]})
        ms.append(r["ms"])
        kept += len(r["corrections"])
        hits = set()
        for k in r["corrections"]:
            o, f = _norm(k["original"]), _norm(k["corrected"])
            for i, err in enumerate(c["errors"]):
                if _norm(err["wrong"]) in o:
                    matched += 1
                    hits.add(i)
                    rights = err["right"] if isinstance(err["right"], list) else [err["right"]]
                    if any(_norm(x) in f for x in rights):
                        right_fix += 1
                    break
        n_errors += len(c["errors"])
        found += len(hits)
        details.append({"said": c["said"], "corrections": r["corrections"], "dropped": r["dropped"]})
    false_alarms = 0
    for s in cases["correct"]:
        r = e.chat({"level": "B1", "said": s})
        ms.append(r["ms"])
        if r["corrections"]:
            false_alarms += 1
            details.append({"said": s, "correct": True, "corrections": r["corrections"]})
    out = {
        "mistakes": n_errors,
        "recall": found / n_errors,
        "precision": matched / kept if kept else 0.0,
        "right_fix": right_fix / matched if matched else 0.0,
        "correct_sentences": len(cases["correct"]),
        "false_alarm_rate": false_alarms / len(cases["correct"]),
    }
    print(f"tutor: recall {out['recall']:.2f}, precision {out['precision']:.2f}, right fix {out['right_fix']:.2f}, "
          f"false alarms on correct sentences {out['false_alarm_rate']:.2f}")

    # How hard the replies are, per level: openings and answers on the same topics.
    level_out = {}
    for level in ["A2", "B1", "B2", "C1"]:
        texts = []
        for topic in ["Travel", "Food", "Work and study", "Films and books", "Free time"]:
            r = e.chat({"level": level, "topic": topic, "said": ""})
            texts.append(r["reply"])
            ms.append(r["ms"])
            r = e.chat({"level": level, "topic": topic, "history": [{"role": "partner", "text": r["reply"]}],
                        "said": "I like it a lot, but I don't have much time for it these days."})
            texts.append(r["reply"])
            ms.append(r["ms"])
        rs = [readability(t) for t in texts]
        level_out[level] = {"words_per_sentence": statistics.mean(r["words_per_sentence"] for r in rs),
                            "fk_grade": statistics.mean(r["fk_grade"] for r in rs), "example": texts[0]}
        print(f"  {level}: {level_out[level]['words_per_sentence']:.1f} words per sentence, "
              f"Flesch-Kincaid grade {level_out[level]['fk_grade']:.1f}")
    out["levels"] = level_out
    out["latency_ms"] = {"median": statistics.median(ms), "p90": float(np.percentile(ms, 90))}
    out["model"] = Path(llm).name
    out["details"] = details
    print(f"  latency: median {out['latency_ms']['median'] / 1000:.1f} s, p90 {out['latency_ms']['p90'] / 1000:.1f} s")
    e.close()
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["asr", "voice", "tutor"])
    ap.add_argument("--llm", help="the language model (.gguf), for the tutor")
    ap.add_argument("--every", type=int, default=5, help="asr: every n-th test utterance")
    a = ap.parse_args()
    if a.what == "asr":
        r = eval_asr(a.every)
    elif a.what == "voice":
        r = eval_voice()
    else:
        if not a.llm:
            raise SystemExit("--llm FILE")
        r = eval_tutor(a.llm)
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"engine-tutor-{Path(a.llm).stem}" if a.what == "tutor" else f"engine-{a.what}"
    (RESULTS / f"{name}.json").write_text(json.dumps(r, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
