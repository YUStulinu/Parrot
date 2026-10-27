"""Tests of the ML side: alignment features through the C library, the scorer and its export, the
evaluation's text measures. No models or data needed (the C library must be built)."""

import json

import numpy as np
import pytest
import torch

from parrot_ml import eval_engine as ev
from parrot_ml.features import Aligner
from parrot_ml.train import N_PHONES, Scorer, evaluate, export, gop_baseline, pcc, utterances

TOKENS = ["<pad>", "w", "iː", "k", "ɔː", "l", "ɑːɹ", "x"]


def frames(tokens):
    V = len(TOKENS)
    lp = np.full((len(tokens), V), np.log(0.1 / (V - 1)), np.float32)
    for t, k in enumerate(tokens):
        lp[t, k] = np.log(0.9)
    return lp


@pytest.fixture(scope="module")
def aligner(tmp_path_factory):
    vocab = tmp_path_factory.mktemp("vocab") / "vocab.json"
    vocab.write_text(json.dumps({t: i for i, t in enumerate(TOKENS)}), encoding="utf-8")
    return Aligner(vocab)


def test_alignment_features(aligner):
    # "walk" W AO K with blanks between: w w _ ɔː ɔː ɔː _ k
    lp = frames([1, 1, 0, 4, 4, 4, 0, 3])
    feats, segs = aligner.features(lp, ["W", "AO1", "K"])
    assert feats.shape == (3, aligner.F) and aligner.F == 41 + 5
    assert list(segs[:, 1]) == [0, 3, 7]  # each phoneme starts where its token does
    gop = feats[:, -5]
    assert (gop > np.log(0.8)).all()  # well pronounced


def test_alignment_wrong_expectation(aligner):
    lp = frames([1, 1, 0, 4, 4, 4, 0, 3])
    feats, _ = aligner.features(lp, ["W", "IY1", "K"])  # expected "week", heard "walk"
    assert feats[1, -5] < -3  # the vowel scores low


def test_alignment_too_short(aligner):
    with pytest.raises(RuntimeError):
        aligner.features(frames([1]), ["W", "AO", "K"])


def synthetic(n_utts=40, seed=0):
    """Features whose GOP column predicts the label, grouped in utterances of 2-word."""
    rng = np.random.default_rng(seed)
    F = 46
    rows = {k: [] for k in ["utt", "word", "phone", "label", "features"]}
    word_acc, utt_acc, speakers = [], [], []
    w = 0
    for u in range(n_utts):
        labels = []
        for _ in range(2):
            for _ in range(3):
                y = float(rng.integers(0, 3))
                f = rng.normal(size=F).astype(np.float32)
                f[F - 5] = y * 2 - 3 + rng.normal(scale=0.3)
                f[F - 4] = y + rng.normal(scale=0.3)
                for k, v in zip(rows, [u, w, int(rng.integers(0, N_PHONES)), y, f]):
                    rows[k].append(v)
                labels.append(y)
            word_acc.append(np.mean(labels[-3:]) * 5)
            w += 1
        utt_acc.append(np.mean(labels) * 5)
        speakers.append(str(u % 4))
    d = {k: np.array(v) for k, v in rows.items()}
    d["features"] = np.stack(rows["features"])
    d["label"] = d["label"].astype(np.float32)
    d["word_accuracy"] = np.array(word_acc, np.float32)
    d["utt_accuracy"] = np.array(utt_acc, np.float32)
    d["utt_speaker"] = np.array(speakers)
    return d


def test_gop_baseline_learns_the_gop():
    r = gop_baseline(synthetic(seed=0), synthetic(seed=1))
    assert r["phone_pcc"] > 0.8


def test_scorer_shapes_and_word_pooling():
    us = utterances(synthetic(8))
    assert len(us) == 8 and list(us[0]["word"]) == [0, 0, 0, 1, 1, 1]
    allf = np.concatenate([u["features"] for u in us])
    m = Scorer(allf.shape[1], allf.mean(0), allf.std(0) + 1e-5)
    r = evaluate(m, us, "cpu")
    assert set(r) == {"phone_pcc", "phone_mse", "word_pcc", "utt_pcc"}
    x = torch.randn(2, 6, allf.shape[1])
    p, w, u = m.eval()(x, torch.zeros(2, 6, dtype=torch.long), torch.tensor([[0, 0, 0, 1, 1, 1]] * 2))
    assert p.shape == (2, 6) and w.shape == (2, 6) and u.shape == (2,)


@pytest.mark.parametrize("context", [False, True])
def test_export_matches_at_any_length(tmp_path, context):
    m = Scorer(46, np.zeros(46), np.ones(46), context=context)
    out = tmp_path / "scorer.onnx"
    export(m, 46, out)  # raises if ONNX and PyTorch differ at lengths other than the traced one
    assert out.stat().st_size > 10_000


def test_pcc():
    a = np.arange(10.0)
    assert pcc(a, 2 * a + 1) == pytest.approx(1.0)
    assert pcc(a, -a) == pytest.approx(-1.0)


def test_wer_and_normalize():
    assert ev.wer("a b c".split(), "a b c".split()) == (0, 3)
    assert ev.wer("a b c".split(), "a x c d".split()) == (2, 3)
    assert ev.normalize("I have 3 cats, don't I?") == ["i", "have", "three", "cats", "don't", "i"]


def test_readability_orders_simple_and_hard_text():
    easy = ev.readability("I like my cat. It is big. We play a lot.")
    hard = ev.readability("Notwithstanding considerable institutional resistance, the administration implemented "
                          "comprehensive regulatory modifications throughout the subsequent legislative period.")
    assert easy["words_per_sentence"] < 5 < hard["words_per_sentence"]
    assert easy["fk_grade"] < hard["fk_grade"]
