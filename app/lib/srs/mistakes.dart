// Turning mistakes into review cards.
import '../engine/results.dart';
import '../phones.dart';
import 'store.dart';

/// A word is practised when its score is under this (0–100), a sound when under [weakSound].
const weakWord = 60.0;
const weakSound = 50.0;

/// The weak sounds of a scored word, as "θ as in think (you said s)".
List<String> weakSounds(WordScore w) => [
      for (final p in w.phones)
        if (p.score < weakSound)
          '${describe(p.phone)}${p.heard != p.phone && phoneIpa.containsKey(p.heard) ? ' (sounded like ${phoneIpa[p.heard]})' : ''}',
    ];

/// Adds a card for each correction; returns how many were new.
int addCorrections(CardStore store, List<Correction> corrections, {DateTime? now}) {
  var added = 0;
  for (final c in corrections) {
    if (store.addMistake(kind: c.kind, front: c.original, back: c.corrected, note: c.explanation, now: now)) added++;
  }
  return added;
}

/// Adds a card for each badly pronounced word (whose pronunciation is known, not guessed).
int addPronunciation(CardStore store, PronunciationResult r, {DateTime? now}) {
  var added = 0;
  for (final w in r.words) {
    if (w.guessed || w.score >= weakWord) continue;
    final sounds = weakSounds(w);
    final front = w.text.toLowerCase();
    final back = '/${w.phones.map((p) => phoneIpa[p.phone] ?? p.phone).join()}/';
    if (store.addMistake(kind: 'pronunciation', front: front, back: back, note: sounds.join('; '), now: now)) added++;
  }
  return added;
}
