// The native engine (libparrot) through dart:ffi.
//
// Every slow call (recognition, scoring, the tutor, the voice) runs on a short-lived isolate with the
// engine's address, so the UI never waits; they are queued, one at a time, because the engine's
// models are not meant to run concurrently. The microphone's level and finished utterances are
// polled from the UI isolate: those calls only take a lock and copy.
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import 'results.dart';

typedef _OpenC = Pointer<Void> Function(Pointer<Utf8>);
typedef _CloseC = Void Function(Pointer<Void>);
typedef _Close = void Function(Pointer<Void>);
typedef _ErrorC = Pointer<Utf8> Function();
typedef _FreeC = Void Function(Pointer<Utf8>);
typedef _Free = void Function(Pointer<Utf8>);
typedef _AudioStrC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Float>, Int32);
typedef _AudioStr = Pointer<Utf8> Function(Pointer<Void>, Pointer<Float>, int);
typedef _ScoreC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Float>, Int32, Pointer<Utf8>);
typedef _Score = Pointer<Utf8> Function(Pointer<Void>, Pointer<Float>, int, Pointer<Utf8>);
typedef _StrStrC = Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>);
typedef _IntC = Int32 Function(Pointer<Void>);
typedef _Int = int Function(Pointer<Void>);
typedef _SpeakC = Int32 Function(Pointer<Void>, Pointer<Utf8>, Pointer<Pointer<Int16>>, Pointer<Int32>);
typedef _Speak = int Function(Pointer<Void>, Pointer<Utf8>, Pointer<Pointer<Int16>>, Pointer<Int32>);
typedef _FreePcmC = Void Function(Pointer<Int16>);
typedef _FreePcm = void Function(Pointer<Int16>);
typedef _ListenStartC = Int32 Function(Pointer<Void>, Int32);
typedef _ListenStart = int Function(Pointer<Void>, int);
typedef _LevelC = Float Function(Pointer<Void>, Pointer<Int32>);
typedef _Level = double Function(Pointer<Void>, Pointer<Int32>);
typedef _TakeC = Int32 Function(Pointer<Void>, Pointer<Pointer<Float>>, Pointer<Int32>);
typedef _Take = int Function(Pointer<Void>, Pointer<Pointer<Float>>, Pointer<Int32>);
typedef _FreeAudioC = Void Function(Pointer<Float>);
typedef _FreeAudio = void Function(Pointer<Float>);
typedef _PlayC = Int32 Function(Pointer<Void>, Pointer<Int16>, Int32, Int32);
typedef _Play = int Function(Pointer<Void>, Pointer<Int16>, int, int);

class EngineException implements Exception {
  EngineException(this.message);
  final String message;
  @override
  String toString() => message;
}

/// The functions of the library (looked up once per isolate).
class _Lib {
  _Lib._(DynamicLibrary l)
      : open = l.lookupFunction<_OpenC, _OpenC>('parrot_open'),
        close = l.lookupFunction<_CloseC, _Close>('parrot_close'),
        lastError = l.lookupFunction<_ErrorC, _ErrorC>('parrot_last_error'),
        free = l.lookupFunction<_FreeC, _Free>('parrot_free'),
        transcribe = l.lookupFunction<_AudioStrC, _AudioStr>('parrot_transcribe'),
        score = l.lookupFunction<_ScoreC, _Score>('parrot_score'),
        chat = l.lookupFunction<_StrStrC, _StrStrC>('parrot_chat'),
        phonemes = l.lookupFunction<_StrStrC, _StrStrC>('parrot_phonemes'),
        voiceRate = l.lookupFunction<_IntC, _Int>('parrot_voice_rate'),
        speak = l.lookupFunction<_SpeakC, _Speak>('parrot_speak'),
        freePcm = l.lookupFunction<_FreePcmC, _FreePcm>('parrot_free_pcm'),
        listenStart = l.lookupFunction<_ListenStartC, _ListenStart>('parrot_listen_start'),
        listenStop = l.lookupFunction<_CloseC, _Close>('parrot_listen_stop'),
        listenLevel = l.lookupFunction<_LevelC, _Level>('parrot_listen_level'),
        listenTake = l.lookupFunction<_TakeC, _Take>('parrot_listen_take'),
        freeAudio = l.lookupFunction<_FreeAudioC, _FreeAudio>('parrot_free_audio'),
        play = l.lookupFunction<_PlayC, _Play>('parrot_play'),
        playing = l.lookupFunction<_IntC, _Int>('parrot_playing');

  static _Lib? _instance;
  static _Lib get instance => _instance ??= _Lib._(_load());

  static DynamicLibrary _load() {
    if (Platform.isAndroid || Platform.isLinux) return DynamicLibrary.open('libparrot.so');
    if (Platform.isWindows) return DynamicLibrary.open('parrot.dll');
    if (Platform.isMacOS) return DynamicLibrary.open('libparrot.dylib');
    throw UnsupportedError('no engine for ${Platform.operatingSystem}');
  }

  final Pointer<Void> Function(Pointer<Utf8>) open;
  final _Close close;
  final Pointer<Utf8> Function() lastError;
  final _Free free;
  final _AudioStr transcribe;
  final _Score score;
  final Pointer<Utf8> Function(Pointer<Void>, Pointer<Utf8>) chat, phonemes;
  final _Int voiceRate;
  final _Speak speak;
  final _FreePcm freePcm;
  final _ListenStart listenStart;
  final _Close listenStop;
  final _Level listenLevel;
  final _Take listenTake;
  final _FreeAudio freeAudio;
  final _Play play;
  final _Int playing;

  String error() => lastError().toDartString();

  /// Reads and frees a JSON string the engine returned, or throws its error.
  Map<String, dynamic> json(Pointer<Utf8> s) {
    if (s == nullptr) throw EngineException(error());
    try {
      return jsonDecode(s.toDartString()) as Map<String, dynamic>;
    } finally {
      free(s);
    }
  }
}

Pointer<Float> _copyAudio(Float32List audio, Arena arena) {
  final p = arena<Float>(audio.length);
  p.asTypedList(audio.length).setAll(0, audio);
  return p;
}

class Engine {
  Engine._(this._address, this.voiceRate);

  final int _address;
  final int voiceRate; // 0 without a voice
  Future<void> _queue = Future.value();
  bool _closed = false;

  Pointer<Void> get _ptr => Pointer<Void>.fromAddress(_address);

  /// Loads the models in [modelsDir] (and the language model [llm], loaded on the first chat).
  static Future<Engine> open({required String modelsDir, String? llm, String? whisper, int threads = 0}) async {
    final config = jsonEncode({
      'models': modelsDir,
      'llm': ?llm,
      'whisper': ?whisper,
      'threads': threads,
    });
    final (address, rate) = await Isolate.run(() {
      final lib = _Lib.instance;
      final c = config.toNativeUtf8();
      try {
        final e = lib.open(c);
        if (e == nullptr) throw EngineException(lib.error());
        return (e.address, lib.voiceRate(e));
      } finally {
        malloc.free(c);
      }
    });
    return Engine._(address, rate);
  }

  /// Runs [job] on its own isolate after the jobs before it.
  Future<T> _run<T>(T Function(_Lib lib, Pointer<Void> e) job) {
    if (_closed) return Future.error(EngineException('the engine is closed'));
    final address = _address;
    final result = _queue.then((_) => Isolate.run(() => job(_Lib.instance, Pointer<Void>.fromAddress(address))));
    _queue = result.then((_) {}, onError: (_) {});
    return result;
  }

  Future<Transcript> transcribe(Float32List audio) => _run((lib, e) => using((arena) {
        return Transcript.fromJson(lib.json(lib.transcribe(e, _copyAudio(audio, arena), audio.length)));
      }));

  Future<PronunciationResult> score(Float32List audio, String text) => _run((lib, e) => using((arena) {
        return PronunciationResult.fromJson(
            lib.json(lib.score(e, _copyAudio(audio, arena), audio.length, text.toNativeUtf8(allocator: arena))));
      }));

  Future<TutorTurn> chat(TutorRequest request) => _run((lib, e) => using((arena) {
        return TutorTurn.fromJson(lib.json(lib.chat(e, jsonEncode(request.toJson()).toNativeUtf8(allocator: arena))));
      }));

  Future<List<WordPhones>> phonemes(String text) => _run((lib, e) => using((arena) {
        final j = lib.json(lib.phonemes(e, text.toNativeUtf8(allocator: arena)));
        return [for (final w in j['words'] as List) WordPhones.fromJson(w as Map<String, dynamic>)];
      }));

  /// The voice's 16-bit samples at [voiceRate].
  Future<Int16List> speak(String text) => _run((lib, e) => using((arena) {
        final pcm = arena<Pointer<Int16>>();
        final n = arena<Int32>();
        if (lib.speak(e, text.toNativeUtf8(allocator: arena), pcm, n) != 0) throw EngineException(lib.error());
        final out = Int16List.fromList(pcm.value.asTypedList(n.value));
        lib.freePcm(pcm.value);
        return out;
      }));

  // --- live audio (UI isolate; quick calls) ---

  void listenStart({int silenceMs = 700}) {
    final lib = _Lib.instance;
    if (lib.listenStart(_ptr, silenceMs) != 0) throw EngineException(lib.error());
  }

  void listenStop() => _Lib.instance.listenStop(_ptr);

  /// The microphone's level (0–1) and whether someone is speaking.
  (double, bool) level() => using((arena) {
        final s = arena<Int32>();
        final l = _Lib.instance.listenLevel(_ptr, s);
        return (l, s.value != 0);
      });

  /// A finished utterance (16 kHz), if there is one.
  Float32List? take() => using((arena) {
        final lib = _Lib.instance;
        final a = arena<Pointer<Float>>();
        final n = arena<Int32>();
        if (lib.listenTake(_ptr, a, n) == 0) return null;
        final out = Float32List.fromList(a.value.asTypedList(n.value));
        lib.freeAudio(a.value);
        return out;
      });

  void play(Int16List pcm, int rate) => using((arena) {
        final lib = _Lib.instance;
        final p = arena<Int16>(pcm.length);
        p.asTypedList(pcm.length).setAll(0, pcm);
        if (lib.play(_ptr, p, pcm.length, rate) != 0) throw EngineException(lib.error());
      });

  bool get playing => _Lib.instance.playing(_ptr) != 0;

  Future<void> close() async {
    if (_closed) return;
    _closed = true;
    await _queue;
    _Lib.instance.close(_ptr);
  }
}
