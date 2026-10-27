"""Trains the pronunciation scorer on speechocean762 and evaluates it against the experts' scores.

The scorer reads, for each phoneme of an utterance, the aligner's features (how well each phoneme
explains its frames, how long it lasts) and the phoneme's identity, and predicts the score an expert
would give: 0 (wrong or missing), 1 (accented), 2 (right). A small transformer over the utterance's
phonemes (in the spirit of GOPT, Gong et al. 2022) lets a phoneme's score depend on its neighbours,
and also predicts word and utterance accuracy (0–10).

Compared with:
- GOP: a linear regression on the expected phoneme's mean log-probability and its margin (LPR),
  the classic "goodness of pronunciation";
- an MLP on one phoneme's features, without context.

Metric: Pearson correlation with the experts on the official test set (2,500 utterances, speakers
not seen in training), as the literature reports it. Three seeds; mean and spread.

    python -m parrot_ml.train
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .paths import MODELS, RESULTS, WORK

N_PHONES = 39


def load(split: str, backend: str = "torch") -> dict:
    d = np.load(WORK / f"features-{split}-{backend}.npz")
    return {k: d[k] for k in d.files}


def utterances(d: dict) -> list[dict]:
    """Groups phonemes by utterance."""
    out = []
    bounds = np.flatnonzero(np.diff(d["utt"])) + 1
    for idx in np.split(np.arange(len(d["utt"])), bounds):
        u = d["utt"][idx[0]]
        w = d["word"][idx]
        out.append({
            "features": d["features"][idx], "phone": d["phone"][idx], "label": d["label"][idx],
            "word": w - w[0], "word_label": d["word_accuracy"][np.unique(w)], "utt_label": d["utt_accuracy"][u],
            "speaker": str(d["utt_speaker"][u]),
        })
    return out


def pcc(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


class Scorer(nn.Module):
    def __init__(self, n_features: int, mean: np.ndarray, std: np.ndarray, dim: int = 64, layers: int = 3, context: bool = True):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32))
        self.phone_embed = nn.Embedding(N_PHONES, 24)
        self.inp = nn.Linear(n_features + 24, dim)
        self.pos = nn.Embedding(128, dim)
        self.context = context
        layer = nn.TransformerEncoderLayer(dim, 4, dim * 2, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers) if context else nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.phone_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.word_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.utt_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def forward(self, features, phones, words, mask=None):
        """features (B, n, F), phones (B, n), words (B, n): word index within the utterance.
        Returns phone scores (B, n) 0–2, word scores (B, n) 0–10 (one per word index), utterance (B,)."""
        x = (features - self.mean) / self.std
        n = features.shape[1]
        h = self.inp(torch.cat([x, self.phone_embed(phones)], -1)) + self.pos(torch.arange(n, device=features.device).clamp(max=127))[None]
        if self.context:
            h = self.encoder(h, src_key_padding_mask=None if mask is None else ~mask)
        else:
            h = self.encoder(h)
        phone = self.phone_head(h).squeeze(-1)
        valid = torch.ones_like(phone) if mask is None else mask.float()
        onehot = (words[:, :, None] == torch.arange(n, device=features.device)[None, None, :]).float() * valid[:, :, None]
        word_h = torch.einsum("bnw,bnd->bwd", onehot, h) / onehot.sum(1).clamp(min=1)[:, :, None]
        word = self.word_head(word_h).squeeze(-1)
        utt_h = (h * valid[:, :, None]).sum(1) / valid.sum(1, keepdim=True).clamp(min=1)
        utt = self.utt_head(utt_h).squeeze(-1)
        return phone, word, utt


def batches(us: list[dict], size: int, shuffle: bool, rng: np.random.Generator):
    order = rng.permutation(len(us)) if shuffle else np.arange(len(us))
    for i in range(0, len(us), size):
        group = [us[j] for j in order[i : i + size]]
        n = max(len(u["phone"]) for u in group)
        B = len(group)
        F = group[0]["features"].shape[1]
        feats = np.zeros((B, n, F), np.float32)
        phones = np.zeros((B, n), np.int64)
        words = np.zeros((B, n), np.int64)
        labels = np.zeros((B, n), np.float32)
        mask = np.zeros((B, n), bool)
        wl = np.zeros((B, n), np.float32)
        wmask = np.zeros((B, n), bool)
        ul = np.zeros(B, np.float32)
        for b, u in enumerate(group):
            k = len(u["phone"])
            feats[b, :k], phones[b, :k], words[b, :k], labels[b, :k], mask[b, :k] = u["features"], u["phone"], u["word"], u["label"], True
            w = len(u["word_label"])
            wl[b, :w], wmask[b, :w] = u["word_label"], True
            ul[b] = u["utt_label"]
        yield [torch.from_numpy(x) for x in (feats, phones, words, labels, mask, wl, wmask, ul)]


def predict(model: Scorer, us: list[dict], device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    ph, wd, ut = [], [], []
    with torch.no_grad():
        for feats, phones, words, _, mask, _, wmask, _ in batches(us, 64, False, np.random.default_rng(0)):
            p, w, u = model(feats.to(device), phones.to(device), words.to(device), mask.to(device))
            ph.append(p.cpu().numpy()[mask.numpy()])
            wd.append(w.cpu().numpy()[wmask.numpy()])
            ut.append(u.cpu().numpy())
    return np.concatenate(ph), np.concatenate(wd), np.concatenate(ut)


def train(train_us, test_us, seed: int, context: bool, epochs: int = 60, device="cuda") -> tuple[Scorer, dict]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    allf = np.concatenate([u["features"] for u in train_us])
    model = Scorer(allf.shape[1], allf.mean(0), allf.std(0) + 1e-5, context=context).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    for _ in range(epochs):
        model.train()
        for feats, phones, words, labels, mask, wl, wmask, ul in batches(train_us, 32, True, rng):
            feats, phones, words, labels, mask, wl, wmask, ul = [t.to(device) for t in (feats, phones, words, labels, mask, wl, wmask, ul)]
            p, w, u = model(feats, phones, words, mask)
            loss = ((p - labels) ** 2)[mask].mean() + 0.02 * ((w - wl) ** 2)[wmask].mean() + 0.02 * ((u - ul) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model, evaluate(model, test_us, device)


def evaluate(model: Scorer, us: list[dict], device) -> dict:
    ph, wd, ut = predict(model, us, device)
    truth_p = np.concatenate([u["label"] for u in us])
    truth_w = np.concatenate([u["word_label"] for u in us])
    truth_u = np.array([u["utt_label"] for u in us])
    return {
        "phone_pcc": pcc(ph, truth_p), "phone_mse": float(((ph - truth_p) ** 2).mean()),
        "word_pcc": pcc(wd, truth_w), "utt_pcc": pcc(ut, truth_u),
    }


def gop_baseline(train_d: dict, test_d: dict) -> dict:
    """Linear regression of the phoneme score on GOP and LPR."""
    F = train_d["features"].shape[1]
    gi, li = F - 5, F - 4
    X = np.c_[train_d["features"][:, [gi, li]], np.ones(len(train_d["label"]))]
    w, *_ = np.linalg.lstsq(X, train_d["label"], rcond=None)
    pred = np.c_[test_d["features"][:, [gi, li]], np.ones(len(test_d["label"]))] @ w
    return {"phone_pcc": pcc(pred, test_d["label"]), "phone_mse": float(((pred - test_d["label"]) ** 2).mean())}


def export(model: Scorer, n_features: int, out: Path = MODELS / "scorer.onnx") -> None:
    model = model.cpu().eval()

    class Wrapped(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, features, phones, words):
            p, w, u = self.m(features, phones, words)
            return p.clamp(0, 2), w.clamp(0, 10), u.clamp(0, 10)

    def example(n: int):
        g = torch.Generator().manual_seed(n)
        return (torch.randn(1, n, n_features, generator=g), torch.randint(0, N_PHONES, (1, n), generator=g),
                torch.arange(n)[None] // 3)

    # The torch.export-based exporter keeps the length symbolic through attention (the TorchScript one
    # bakes it in).
    n = torch.export.Dim("n", min=1, max=512)
    torch.onnx.export(Wrapped(model), example(12), str(out), input_names=["features", "phones", "words"],
                      output_names=["phone_scores", "word_scores", "utterance_score"],
                      dynamic_shapes=({1: n}, {1: n}, {1: n}), opset_version=18, dynamo=True, external_data=False,
                      verbose=False)
    # Parity at other lengths than the traced one.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    worst = 0.0
    for n in (1, 3, 12, 41, 90):
        args = example(n)
        with torch.no_grad():
            want = Wrapped(model)(*args)
        got = sess.run(None, {"features": args[0].numpy(), "phones": args[1].numpy(), "words": args[2].numpy()})
        worst = max(worst, *(float(np.abs(g - w.numpy()).max()) for g, w in zip(got, want)))
    if worst > 1e-3:
        raise RuntimeError(f"the ONNX scorer differs from PyTorch by {worst}")
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB), ONNX matches PyTorch to {worst:.1e} at lengths 1-90")


def skew() -> dict:
    """The exported scorer (ONNX, as the app runs it) on the same test subset through each phoneme
    model: PyTorch, ONNX fp32, and the int8 variants."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(MODELS / "scorer.onnx"), providers=["CPUExecutionProvider"])
    out = {}
    for f in sorted(WORK.glob("features-test-*-every5.npz")):
        backend = f.name[len("features-test-"):-len("-every5.npz")]
        d = load("test", f"{backend}-every5")
        ph, wd, ut = [], [], []
        for u in utterances(d):
            p, w, t = sess.run(None, {"features": u["features"][None].astype(np.float32), "phones": u["phone"][None].astype(np.int64),
                                      "words": u["word"][None].astype(np.int64)})
            ph.append(p[0])
            wd.append(w[0][: len(u["word_label"])])
            ut.append(float(np.ravel(t)[0]))
        out[backend] = {"phone_pcc": pcc(np.concatenate(ph), d["label"]), "word_pcc": pcc(np.concatenate(wd), d["word_accuracy"]),
                        "utt_pcc": pcc(np.array(ut), d["utt_accuracy"])}
        print(f"{backend:>7}: phone {out[backend]['phone_pcc']:.3f}, word {out[backend]['word_pcc']:.3f}, "
              f"utterance {out[backend]['utt_pcc']:.3f}")
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if "--skew" in sys.argv:
        r = json.loads((RESULTS / "scorer.json").read_text(encoding="utf-8"))
        r["skew_onnx_scorer"] = skew()
        (RESULTS / "scorer.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_d, test_d = load("train"), load("test")
    train_all, test_us = utterances(train_d), utterances(test_d)
    # Model selection on held-out speakers of the training set (10%), never on the test set.
    speakers = sorted({u["speaker"] for u in train_all})
    held = set(np.random.default_rng(0).choice(speakers, max(1, len(speakers) // 10), replace=False))
    train_us = [u for u in train_all if u["speaker"] not in held]
    val_us = [u for u in train_all if u["speaker"] in held]
    print(f"{len(train_us)} training and {len(val_us)} validation utterances ({len(held)} held-out speakers), {len(test_us)} test")
    results = {"gop_linear": gop_baseline(train_d, test_d)}
    print(f"GOP (linear): phone PCC {results['gop_linear']['phone_pcc']:.3f}")
    best = None
    for name, context in [("mlp", False), ("transformer", True)]:
        runs = []
        for seed in range(3):
            t = time.time()
            model, r = train(train_us, test_us, seed, context, device=device)
            v = evaluate(model, val_us, device)
            runs.append(r)
            print(f"{name} seed {seed}: test phone PCC {r['phone_pcc']:.3f}, word {r['word_pcc']:.3f}, utterance {r['utt_pcc']:.3f}"
                  f" (validation phone {v['phone_pcc']:.3f}, {time.time() - t:.0f} s)")
            if best is None or v["phone_pcc"] > best[2]["phone_pcc"]:
                best = (model, r, v, f"{name} seed {seed}")
        results[name] = {k: {"mean": float(np.mean([r[k] for r in runs])), "std": float(np.std([r[k] for r in runs]))} for k in runs[0]}
    export(best[0], train_d["features"].shape[1])
    results["exported"] = {"model": best[3], "validation": best[2], "test": best[1]}
    print(f"exported {best[3]} (best validation phone PCC)")
    # Train/serve skew: the app runs the ONNX phoneme models (int8 on phones); the scorer was trained on
    # PyTorch's posteriors. The same test subset through each, if extracted (features --every 5 ...).
    for backend in ["torch", "fp32", "int8"]:
        try:
            d = load("test", f"{backend}-every5")
        except FileNotFoundError:
            continue
        results[f"exported_on_{backend}_subset"] = r = evaluate(best[0].to(device), utterances(d), device)
        print(f"exported scorer on {backend} features (every 5th test utterance): phone {r['phone_pcc']:.3f}, "
              f"word {r['word_pcc']:.3f}, utterance {r['utt_pcc']:.3f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "scorer.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
