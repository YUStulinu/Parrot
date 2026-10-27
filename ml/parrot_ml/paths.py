"""Where things are."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "ml"
DATA = ML / "data"
WORK = ML / "work"  # features, checkpoints (ignored by git)
MODELS = ROOT / "native" / "third_party" / "models"
RESULTS = ML / "eval"

# The phoneme model: wav2vec2 large (LV-60) fine-tuned on espeak phonemes (Apache 2.0).
ACOUSTIC_REPO = "facebook/wav2vec2-lv-60-espeak-cv-ft"
ACOUSTIC_REVISION = "ae45363bf3413b374fecd9dc8bc1df0e24c3b7f4"


def native_library(engine: bool = False) -> ctypes.CDLL:
    """The C API (native/include/parrot.h), from the alignment-only build or the full one (engine=True:
    the full one; PARROT_BUILD names another build folder, e.g. build-cuda)."""
    names = ["parrot.dll"] if os.name == "nt" else ["libparrot.so", "libparrot.dylib"]
    full = [os.environ["PARROT_BUILD"]] if os.environ.get("PARROT_BUILD") else []
    full += ["build/Release", "build"]
    for build in full if engine else ["build-core/Release", "build-core"] + full:
        for n in names:
            p = ROOT / "native" / build / n
            if p.exists():
                if os.name == "nt":
                    os.add_dll_directory(str(p.parent))
                    # A GPU build needs the CUDA runtime (in bind since CUDA 13).
                    cuda = os.environ.get("CUDA_PATH")
                    for d in [Path(cuda) / "bin" / "x64", Path(cuda) / "bin"] if cuda else []:
                        if d.is_dir():
                            os.add_dll_directory(str(d))
                return ctypes.CDLL(str(p))
    raise SystemExit("build native/ first (see the README): the parrot library is missing")
