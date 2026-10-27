#!/usr/bin/env bash
# Downloads the pinned sources, libraries and models the native code and the app use, checks their
# SHA-256, and unpacks them into native/third_party (ignored by git).
#   scripts/fetch-deps.sh            sources, onnxruntime for this OS and Android, the models
#   scripts/fetch-deps.sh --ci       the same without Whisper small (the desktop's larger speech model)
set -euo pipefail
ci=0
[ "${1:-}" = --ci ] && ci=1
root="$(cd "$(dirname "$0")/.." && pwd)"
tp="$root/native/third_party"
dl="$tp/downloads"
mkdir -p "$dl" "$tp/models"

fetch() { # url sha256 file
  local url="$1" sha="$2" file="$dl/$3"
  if [ ! -f "$file" ] || ! echo "$sha  $file" | sha256sum -c - >/dev/null 2>&1; then
    echo "downloading $url"
    curl -fsSL --retry 3 -o "$file.part" "$url" || { echo "download failed: $url" >&2; exit 1; }
    echo "$sha  $file.part" | sha256sum -c - >/dev/null || { echo "checksum mismatch: $url" >&2; rm -f "$file.part"; exit 1; }
    mv "$file.part" "$file"
  fi
}

unpack() { # file dir-inside-archive target
  local file="$dl/$1" target="$tp/$3"
  [ -d "$target" ] && return
  local tmp; tmp="$(mktemp -d)"
  case "$file" in
    *.zip | *.aar) unzip -q "$file" -d "$tmp" ;;
    *) tar -xzf "$file" -C "$tmp" ;;
  esac
  if [ "$2" = . ]; then mv "$tmp" "$target"; else mv "$tmp/$2" "$target"; rm -rf "$tmp"; fi
}

fetch https://github.com/ggml-org/whisper.cpp/archive/refs/tags/v1.9.5.tar.gz \
  ff1a9053feb509ff9d7729703355541ae9690073a6b1c40eb692c962e0dc1720 whisper.cpp-1.9.5.tar.gz
unpack whisper.cpp-1.9.5.tar.gz whisper.cpp-1.9.5 whisper.cpp

fetch https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11429.tar.gz \
  0487b2e58d0f87c50def234ebda5dd8c1ab6c7afbc440ebef0080203110db6d7 llama.cpp-b11429.tar.gz
unpack llama.cpp-b11429.tar.gz llama.cpp-b11429 llama.cpp

# onnxruntime runs the phoneme model, the pronunciation scorer and the Piper voice.
case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    fetch https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-win-x64-1.31.1.zip \
      5869e26bb25314375f2d5ef1877e3ea06b1543d6c710e6e4a1ec2f7177ab4b2c onnxruntime-win-x64-1.31.1.zip
    unpack onnxruntime-win-x64-1.31.1.zip onnxruntime-win-x64-1.31.1 onnxruntime-host ;;
  Linux)
    fetch https://github.com/microsoft/onnxruntime/releases/download/v1.31.1/onnxruntime-linux-x64-1.31.1.tgz \
      c849e36a1ef3e22d4ff217b9d1ce4a3cf5b9467b975610913ae959ad5b16bdc9 onnxruntime-linux-x64-1.31.1.tgz
    unpack onnxruntime-linux-x64-1.31.1.tgz onnxruntime-linux-x64-1.31.1 onnxruntime-host ;;
esac
fetch https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/1.31.1/onnxruntime-android-1.31.1.aar \
  d4d50660300ed2c4f38ffdfd5ab5bfa3a22f995f282dda24f42ea3ea55467b49 onnxruntime-android-1.31.1.aar
unpack onnxruntime-android-1.31.1.aar . onnxruntime-android

fetch https://raw.githubusercontent.com/mackron/miniaudio/0.11.25/miniaudio.h \
  ac7af4de748b7e26b777f37e01cee313a308a7296a3eb080e2906b320cc55c89 miniaudio-0.11.25.h
mkdir -p "$tp/miniaudio" && cp "$dl/miniaudio-0.11.25.h" "$tp/miniaudio/miniaudio.h"

# The CMU pronouncing dictionary: the expected phonemes of what you say, and the voice's input.
fetch https://raw.githubusercontent.com/cmusphinx/cmudict/74790861f652b15e4ac49015a90074ad62a27690/cmudict.dict \
  81917843c7f44ce2b094ac63873c2c7a4cf802040792c455ba3ca406891c3d22 cmudict.dict

# Models: Whisper (base.en for phones, small.en for the desktop), the Piper voice (LJSpeech, public domain).
fetch https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin \
  4baf70dd0d7c4247ba2b81fafd9c01005ac77c2f9ef064e00dcf195d0e2fdd2f ggml-base.en-q5_1.bin
models="cmudict.dict ggml-base.en-q5_1.bin en_US-ljspeech-medium.onnx en_US-ljspeech-medium.onnx.json"
if [ $ci = 0 ]; then
  fetch https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q5_1.bin \
    bfdff4894dcb76bbf647d56263ea2a96645423f1669176f4844a1bf8e478ad30 ggml-small.en-q5_1.bin
  models="$models ggml-small.en-q5_1.bin"
fi
fetch https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx \
  6f52a751e2349abe7a76735eb09dc1875298c77ea2342ffd2fef79ff81b87f22 en_US-ljspeech-medium.onnx
fetch https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json \
  141d612cc0a95ed7efc1ca936b845c2364967f2e9217c5dbfcf69fc4d6c65860 en_US-ljspeech-medium.onnx.json
for f in $models; do
  cp -u "$dl/$f" "$tp/models/$f"
done
echo "dependencies are in $tp"
