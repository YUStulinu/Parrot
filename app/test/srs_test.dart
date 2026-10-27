import 'package:flutter_test/flutter_test.dart';
import 'package:parrot/engine/results.dart';
import 'package:parrot/srs/fsrs.dart';
import 'package:parrot/srs/mistakes.dart';
import 'package:parrot/srs/store.dart';
import 'package:parrot/talk.dart';

void main() {
  group('FSRS', () {
    final f = Fsrs();

    test('recall is 90% after S days', () {
      expect(f.recall(10, 10), closeTo(0.9, 1e-9));
      expect(f.recall(0, 10), 1.0);
      expect(f.recall(30, 10), lessThan(0.9));
    });

    test('the interval at 90% retention is the stability', () {
      expect(f.interval(10), 10);
      expect(f.interval(0.2), 1); // at least a day
      expect(f.interval(10000), 365); // at most the maximum
    });

    test('first grades: better grade, longer memory, easier card', () {
      final m = [for (final g in Grade.values) f.first(g)];
      for (var i = 1; i < 4; i++) {
        expect(m[i].stability, greaterThan(m[i - 1].stability));
        expect(m[i].difficulty, lessThan(m[i - 1].difficulty));
      }
      expect(m[2].stability, closeTo(3.173, 1e-9)); // w2
    });

    test('a successful review on time grows the stability; a lapse shrinks it', () {
      const m = Memory(10, 5);
      final good = f.next(m, Grade.good, 10);
      final again = f.next(m, Grade.again, 10);
      expect(good.stability, greaterThan(20));
      expect(again.stability, lessThan(10));
      expect(again.difficulty, greaterThan(m.difficulty));
      expect(f.next(m, Grade.easy, 10).stability, greaterThan(good.stability));
      expect(f.next(m, Grade.hard, 10).stability, lessThan(good.stability));
    });

    test('a late success grows stability more than an early one', () {
      const m = Memory(10, 5);
      expect(f.next(m, Grade.good, 20).stability, greaterThan(f.next(m, Grade.good, 5).stability));
    });

    test('difficulty stays in 1..10', () {
      var m = f.first(Grade.again);
      for (var i = 0; i < 30; i++) {
        m = f.next(m, Grade.again, 1);
      }
      expect(m.difficulty, lessThanOrEqualTo(10));
      for (var i = 0; i < 60; i++) {
        m = f.next(m, Grade.easy, 1);
      }
      expect(m.difficulty, greaterThanOrEqualTo(1));
    });
  });

  group('cards', () {
    late CardStore store;
    final t0 = DateTime(2026, 1, 1, 12);
    setUp(() => store = CardStore.memory());
    tearDown(() => store.close());

    test('a mistake becomes a due card; the same mistake again does not duplicate it', () {
      expect(store.addMistake(kind: 'grammar', front: 'I goed', back: 'I went', now: t0), isTrue);
      expect(store.addMistake(kind: 'grammar', front: 'I goed', back: 'I went', now: t0), isFalse);
      expect(store.due(now: t0).length, 1);
      expect(store.stats(now: t0).total, 1);
    });

    test('reviews schedule the card and are recorded', () {
      store.addMistake(kind: 'grammar', front: 'I goed', back: 'I went', now: t0);
      final id = store.due(now: t0).first.id;
      final c = store.review(id, Grade.good, now: t0);
      expect(c.reps, 1);
      expect(c.due.difference(t0).inDays, 3); // FSRS-5's first "good" stability, 3.17 days
      expect(store.due(now: t0.add(const Duration(days: 2))), isEmpty);
      expect(store.due(now: t0.add(const Duration(days: 3))).length, 1);
      final again = store.review(id, Grade.again, now: t0.add(const Duration(days: 3)));
      expect(again.lapses, 1);
      expect(again.due.difference(t0.add(const Duration(days: 3))).inMinutes, 10);
      expect(store.stats(now: t0).reviews, 2);
    });

    test('a repeated mistake brings a scheduled card back today', () {
      store.addMistake(kind: 'grammar', front: 'I goed', back: 'I went', now: t0);
      store.review(store.due(now: t0).first.id, Grade.easy, now: t0);
      final later = t0.add(const Duration(days: 1));
      expect(store.due(now: later), isEmpty);
      store.addMistake(kind: 'grammar', front: 'I goed', back: 'I went', now: later);
      expect(store.due(now: later).length, 1);
    });

    test('the buttons preview longer intervals for better grades', () {
      store.addMistake(kind: 'vocabulary', front: 'make a photo', back: 'take a photo', now: t0);
      final p = store.preview(store.due(now: t0).first, now: t0);
      expect(p[Grade.again]!.inMinutes, 10);
      expect(p[Grade.easy]!.inDays, greaterThan(p[Grade.good]!.inDays));
    });

    test('corrections and badly pronounced words become cards', () {
      final n = addCorrections(store, [Correction('I goed', 'I went', 'go → went in the past', 'grammar')], now: t0);
      expect(n, 1);
      final r = PronunciationResult([
        WordScore('think', 30, false, [PhoneScore('TH', 'S', 10, 0, 0.1), PhoneScore('IH', 'IH', 90, 0.1, 0.2)]),
        WordScore('well', 95, false, [PhoneScore('W', 'W', 95, 0, 0.1)]),
        WordScore('zorp', 10, true, [PhoneScore('Z', 'S', 10, 0, 0.1)]),
      ], 60, 0);
      expect(addPronunciation(store, r, now: t0), 1); // the good word and the guessed one are not cards
      final card = store.all().firstWhere((c) => c.kind == 'pronunciation');
      expect(card.front, 'think');
      expect(card.back, '/θɪ/');
      expect(card.note, contains('θ as in think'));
      expect(card.note, contains('sounded like s'));
    });
  });

  test('non-speech transcripts are ignored', () {
    expect(isNonSpeech('[BLANK_AUDIO]'), isTrue);
    expect(isNonSpeech(' (wind blowing) '), isTrue);
    expect(isNonSpeech('...'), isTrue);
    expect(isNonSpeech('I went home.'), isFalse);
  });

  test('engine JSON is read', () {
    final t = TutorTurn.fromJson({
      'reply': 'Nice!',
      'corrections': [
        {'original': 'I goed', 'corrected': 'I went', 'explanation': 'x', 'kind': 'grammar'}
      ],
      'dropped': [],
      'ms': 1200.5,
    });
    expect(t.corrections.single.corrected, 'I went');
    final p = PronunciationResult.fromJson({
      'score': 72,
      'ms': 300,
      'words': [
        {
          'text': 'bear',
          'score': 80,
          'guessed': false,
          'phones': [
            {'phone': 'B', 'expected': 'B', 'heard': 'B', 'score': 90, 'start': 0.1, 'end': 0.2}
          ]
        }
      ],
    });
    expect(p.words.single.phones.single.score, 90);
    expect(TutorRequest(level: 'A2', history: [Turn('partner', 'Hi')]).toJson()['history'], [
      {'role': 'partner', 'text': 'Hi'}
    ]);
  });
}
