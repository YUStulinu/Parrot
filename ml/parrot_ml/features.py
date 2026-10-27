"""Pronunciation features of speechocean762 (5,000 utterances read by 250 learners of English, every
phoneme scored 0–2 by five experts), as the app computes them: the phoneme model's log-probabilities,
then the C++ aligner (the same code as the app, through the C API) for the features.

    python -m parrot_ml.features              # PyTorch on the GPU (fp32), for training
    python -m parrot_ml.features --onnx int8  # the app's int8 model through onnxruntime (the test set)
"""

from __future__ import annotations

import argparse
import ctypes
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .paths import DATA, MODELS, WORK, native_library


def load_split(split: str) -> pd.DataFrame:
    return pd.read_parquet(DATA / f"speechocean762-{split}.parquet")


class Aligner:
    def __init__(self, vocab: Path | None = None):
        self.lib = native_library()
        self.lib.parrot_last_error.restype = ctypes.c_char_p
        if self.lib.parrot_load_vocab(str(vocab or MODELS / "acoustic.vocab.json").encode()) != 0:
            raise RuntimeError(self.lib.parrot_last_error().decode())
        self.F = self.lib.parrot_feature_count()

    def features(self, logprobs: np.ndarray, phones: list[str]):
        lp = np.ascontiguousarray(logprobs, dtype=np.float32)
        n = len(phones)
        feats = np.zeros((n, self.F), dtype=np.float32)
        segs = np.zeros((n, 3), dtype=np.int32)
        rc = self.lib.parrot_align_features(lp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), lp.shape[0], lp.shape[1],
                                            " ".join(phones).encode(), feats.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                            segs.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
        if rc != 0:
            raise RuntimeError(self.lib.parrot_last_error().decode())
        return feats, segs


def extract(split: str, backend: str, every: int = 1) -> None:
    """Features of every utterance of the split, or of every `every`-th (a deterministic subset)."""
    d = load_split(split)
    if every > 1:
        d = d.iloc[::every].reset_index(drop=True)
    aligner = Aligner()
    if backend == "torch":
        import torch

        from .export import load

        acoustic, _ = load("cuda" if torch.cuda.is_available() else "cpu")
        device = next(acoustic.parameters()).device

        def logprobs(wav):
            with torch.no_grad():
                return acoustic(torch.from_numpy(wav)[None].to(device)).float().cpu().numpy()[0]
    else:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(MODELS / ("acoustic.onnx" if backend == "fp32" else f"acoustic-{backend}.onnx")),
                                    providers=["CPUExecutionProvider"])

        def logprobs(wav):
            return sess.run(None, {"audio": wav[None]})[0][0]

    rows = {k: [] for k in ["utt", "word", "phone", "label", "features", "start", "end"]}
    utts, words = [], []
    t0 = time.time()
    audio_seconds = 0.0
    skipped = 0
    for i, r in enumerate(d.itertuples()):
        wav, sr = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
        assert sr == 16000
        audio_seconds += len(wav) / 16000
        phones = [p for w in r.words for p in w["phones"]]
        try:
            feats, segs = aligner.features(logprobs(wav), phones)
        except RuntimeError:
            skipped += 1
            continue
        k = 0
        for wi, w in enumerate(r.words):
            for p, a in zip(w["phones"], w["phones-accuracy"]):
                rows["utt"].append(len(utts))
                rows["word"].append(len(words))
                rows["phone"].append(segs[k, 0])
                rows["label"].append(a)
                rows["features"].append(feats[k])
                rows["start"].append(segs[k, 1])
                rows["end"].append(segs[k, 2])
                k += 1
            words.append((len(utts), w["text"], w["accuracy"], w["total"]))
        utts.append((r.text, r.speaker, r.accuracy, r.fluency, r.completeness, r.prosodic, r.total))
        if (i + 1) % 250 == 0:
            print(f"{split} {backend}: {i + 1}/{len(d)} ({audio_seconds / (time.time() - t0):.0f}x real time)")
    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / f"features-{split}-{backend}{f'-every{every}' if every > 1 else ''}.npz"
    np.savez_compressed(
        out,
        utt=np.array(rows["utt"]), word=np.array(rows["word"]), phone=np.array(rows["phone"]),
        label=np.array(rows["label"], dtype=np.float32), features=np.stack(rows["features"]),
        start=np.array(rows["start"]), end=np.array(rows["end"]),
        word_utt=np.array([w[0] for w in words]), word_text=np.array([w[1] for w in words]),
        word_accuracy=np.array([w[2] for w in words], dtype=np.float32),
        utt_text=np.array([u[0] for u in utts]), utt_speaker=np.array([u[1] for u in utts]),
        utt_accuracy=np.array([u[2] for u in utts], dtype=np.float32), utt_fluency=np.array([u[3] for u in utts], dtype=np.float32),
        utt_total=np.array([u[6] for u in utts], dtype=np.float32),
        seconds=np.float32(time.time() - t0), audio_seconds=np.float32(audio_seconds),
    )
    print(f"{out.name}: {len(utts)} utterances, {len(rows['phone'])} phonemes, {skipped} skipped, "
          f"{audio_seconds / 3600:.1f} h of speech in {(time.time() - t0) / 60:.1f} min")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", choices=["fp32", "int8"], help="the app's ONNX model instead of PyTorch (test set)")
    ap.add_argument("--every", type=int, default=1, help="only every n-th test utterance (with --onnx or --subset)")
    ap.add_argument("--subset", action="store_true", help="PyTorch features of the --every subset of the test set")
    a = ap.parse_args()
    if a.onnx or a.subset:
        extract("test", a.onnx or "torch", a.every)
    else:
        for split in ["train", "test"]:
            extract(split, "torch")


if __name__ == "__main__":
    main()
