"""Exports the phoneme model to ONNX for the app: the waveform in (normalized inside the graph), CTC
log-probabilities over the model's 392 IPA tokens out, 50 frames a second.

Two files: fp32 (the desktop) and int8 (dynamic quantization of the matrix products, a quarter of
the size, for phones). Both are checked against PyTorch on real speech: the frames' best tokens must
agree, and the log-probabilities must be close.

    python -m parrot_ml.export
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import time

import numpy as np
import soundfile as sf
import torch

from .paths import ACOUSTIC_REPO, ACOUSTIC_REVISION, DATA, MODELS


class Acoustic(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, audio):  # (1, samples) float32 at 16 kHz
        x = (audio - audio.mean(dim=-1, keepdim=True)) / torch.sqrt(audio.var(dim=-1, keepdim=True) + 1e-7)
        return torch.log_softmax(self.model(x).logits, dim=-1)


def load(device: str = "cpu"):
    from huggingface_hub import hf_hub_download
    from transformers import Wav2Vec2ForCTC

    model = Wav2Vec2ForCTC.from_pretrained(ACOUSTIC_REPO, revision=ACOUSTIC_REVISION).eval().to(device)
    # The vocabulary as it is (the tokenizer class would want espeak installed, for nothing here).
    vocab = json.loads(open(hf_hub_download(ACOUSTIC_REPO, "vocab.json", revision=ACOUSTIC_REVISION), encoding="utf-8").read())
    return Acoustic(model).eval(), vocab


def sample_audio(n: int = 3) -> list[np.ndarray]:
    import pandas as pd

    d = pd.read_parquet(DATA / "speechocean762-test.parquet", columns=["audio"]).head(n)
    out = []
    for a in d["audio"]:
        wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        assert sr == 16000
        out.append(wav)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    MODELS.mkdir(parents=True, exist_ok=True)
    acoustic, vocab = load()
    (MODELS / "acoustic.vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    fp32 = MODELS / "acoustic.onnx"
    dummy = torch.zeros(1, 32000)
    torch.onnx.export(acoustic, (dummy,), str(fp32), input_names=["audio"], output_names=["logprobs"],
                      dynamic_axes={"audio": {1: "samples"}, "logprobs": {1: "frames"}}, opset_version=17, dynamo=False)
    # Weights may land in a side file: fold them into one file the app can copy.
    import onnx

    m = onnx.load(str(fp32))
    onnx.save(m, str(fp32), save_as_external_data=False)
    for extra in MODELS.glob("acoustic.onnx.data"):
        extra.unlink()
    int8 = MODELS / "acoustic-int8.onnx"
    # Per-channel weights, matrix products only (the convolutional front end stays fp32): plain dynamic
    # int8 of every op cost the scorer 0.04 phone PCC (ml/eval/scorer.json, "int8-all-ops"); this costs none.
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=["MatMul"])
    print(f"acoustic.onnx {fp32.stat().st_size / 1e6:.0f} MB, acoustic-int8.onnx {int8.stat().st_size / 1e6:.0f} MB")

    report = []
    for name, path in [("fp32", fp32), ("int8", int8)]:
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        for wav in sample_audio():
            with torch.no_grad():
                ref = acoustic(torch.from_numpy(wav)[None]).numpy()[0]
            t = time.time()
            got = sess.run(None, {"audio": wav[None]})[0][0]
            secs = time.time() - t
            agree = float((ref.argmax(-1) == got.argmax(-1)).mean())
            diff = float(np.abs(np.exp(ref) - np.exp(got)).max())
            report.append((name, len(wav) / 16000, secs, agree, diff))
            print(f"{name}: {len(wav) / 16000:.1f} s of speech in {secs:.2f} s; best token agrees on {agree:.1%} of frames, "
                  f"max probability difference {diff:.3f}")
    worst = min(r[3] for r in report if r[0] == "fp32")
    if worst < 0.999:
        raise SystemExit("the fp32 export does not match PyTorch")


if __name__ == "__main__":
    main()
