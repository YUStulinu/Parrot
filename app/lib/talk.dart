// A conversation, hands-free: the partner speaks, the microphone listens until you stop talking, what
// you said is transcribed and answered; corrections go beside your words and into the review cards.
import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';

import 'engine/engine.dart';
import 'engine/results.dart';
import 'srs/mistakes.dart';
import 'srs/store.dart';

enum TalkPhase { idle, thinking, speaking, listening, hearing }

class Message {
  Message(this.role, this.text, {this.corrections = const []});
  final String role; // partner, learner
  final String text;
  List<Correction> corrections;
}

/// What the recognizer writes for non-speech: "[BLANK_AUDIO]", "(wind blowing)", "*cough*".
bool isNonSpeech(String text) {
  final t = text.replaceAll(RegExp(r'\[[^\]]*\]|\([^)]*\)|\*[^*]*\*'), '').trim();
  return t.isEmpty || !RegExp(r'[A-Za-z]').hasMatch(t);
}

/// Plays [pcm] and waits for the end.
Future<void> playAndWait(Engine e, Int16List pcm) async {
  e.play(pcm, e.voiceRate);
  await Future<void>.delayed(const Duration(milliseconds: 100));
  while (e.playing) {
    await Future<void>.delayed(const Duration(milliseconds: 50));
  }
}

/// Listens for one utterance (or null after [timeout]).
Future<Float32List?> listenOnce(Engine e, {Duration timeout = const Duration(seconds: 20), void Function(double, bool)? onLevel}) async {
  e.listenStart(silenceMs: 800);
  final end = DateTime.now().add(timeout);
  try {
    while (DateTime.now().isBefore(end)) {
      final a = e.take();
      if (a != null) return a;
      final (l, s) = e.level();
      onLevel?.call(l, s);
      await Future<void>.delayed(const Duration(milliseconds: 50));
    }
    return null;
  } finally {
    e.listenStop();
  }
}

class TalkSession extends ChangeNotifier {
  TalkSession(this.engine, this.cards, {required this.level, this.topic = '', this.speak = true, this.onCards});

  final Engine engine;
  final CardStore cards;
  final String level, topic;
  final bool speak;
  final VoidCallback? onCards;

  final messages = <Message>[];
  TalkPhase phase = TalkPhase.idle;
  double micLevel = 0;
  bool speaking = false;
  String? error;
  bool _running = false;
  int newCards = 0;

  bool get running => _running;

  /// Starts the conversation (the partner opens it) and, with [listen], the microphone.
  Future<void> start({bool listen = true}) async {
    if (_running) return;
    _running = true;
    error = null;
    notifyListeners();
    try {
      if (messages.isEmpty) await _partnerTurn('');
      if (listen) await _loop();
    } catch (e) {
      error = '$e';
    }
    _running = false;
    _set(TalkPhase.idle);
  }

  void stop() {
    _running = false;
    engine.listenStop();
    notifyListeners();
  }

  /// Typed instead of spoken (no microphone, or a noisy room).
  Future<void> send(String text) async {
    if (text.trim().isEmpty || phase == TalkPhase.thinking) return;
    final wasRunning = _running;
    stop();
    try {
      await _learnerTurn(text.trim());
    } catch (e) {
      error = '$e';
    }
    _set(TalkPhase.idle);
    if (wasRunning) unawaited(start());
  }

  Future<void> _loop() async {
    while (_running) {
      _set(TalkPhase.listening);
      final audio = await listenOnce(engine, timeout: const Duration(seconds: 60), onLevel: (l, s) {
        micLevel = l;
        speaking = s;
        notifyListeners();
      });
      if (!_running) return;
      if (audio == null) continue;
      _set(TalkPhase.hearing);
      final t = await engine.transcribe(audio);
      if (!_running) return;
      if (isNonSpeech(t.text)) continue;
      await _learnerTurn(t.text);
    }
  }

  Future<void> _learnerTurn(String said) async {
    final m = Message('learner', said);
    messages.add(m);
    notifyListeners();
    await _partnerTurn(said, learner: m);
  }

  Future<void> _partnerTurn(String said, {Message? learner}) async {
    _set(TalkPhase.thinking);
    final history = [
      for (final m in messages.take(learner == null ? messages.length : messages.length - 1)) Turn(m.role, m.text),
    ];
    final turn = await engine.chat(TutorRequest(level: level, topic: topic, history: history, said: said));
    if (learner != null && turn.corrections.isNotEmpty) {
      learner.corrections = turn.corrections;
      newCards += addCorrections(cards, turn.corrections);
      onCards?.call();
    }
    messages.add(Message('partner', turn.reply));
    notifyListeners();
    if (speak && engine.voiceRate > 0) {
      _set(TalkPhase.speaking);
      await playAndWait(engine, await engine.speak(turn.reply));
    }
  }

  void _set(TalkPhase p) {
    phase = p;
    notifyListeners();
  }

  bool _disposed = false;

  @override
  void notifyListeners() {
    if (!_disposed) super.notifyListeners();
  }

  @override
  void dispose() {
    stop();
    _disposed = true;
    super.dispose();
  }
}
