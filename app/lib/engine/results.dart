// What the engine returns (its JSON) and what the tutor is asked.

double _d(Object? v) => (v as num?)?.toDouble() ?? 0;

class Word {
  Word(this.text, this.start, this.end, this.p);
  final String text;
  final double start, end, p;
}

class Transcript {
  Transcript(this.text, this.words, this.ms);
  final String text;
  final List<Word> words;
  final double ms;

  factory Transcript.fromJson(Map<String, dynamic> j) => Transcript(
        (j['text'] as String).trim(),
        [
          for (final w in j['words'] as List)
            Word(w['text'] as String, _d(w['start']), _d(w['end']), _d(w['p'])),
        ],
        _d(j['ms']),
      );
}

class PhoneScore {
  PhoneScore(this.phone, this.heard, this.score, this.start, this.end);
  final String phone, heard; // ARPAbet
  final double score; // 0–100
  final double start, end; // seconds
}

class WordScore {
  WordScore(this.text, this.score, this.guessed, this.phones);
  final String text;
  final double score; // 0–100
  final bool guessed; // the pronunciation was guessed from the spelling
  final List<PhoneScore> phones;
}

class PronunciationResult {
  PronunciationResult(this.words, this.score, this.ms);
  final List<WordScore> words;
  final double score; // 0–100
  final double ms;

  factory PronunciationResult.fromJson(Map<String, dynamic> j) => PronunciationResult(
        [
          for (final w in j['words'] as List)
            WordScore(w['text'] as String, _d(w['score']), w['guessed'] as bool? ?? false, [
              for (final p in w['phones'] as List)
                PhoneScore(p['expected'] as String, p['heard'] as String, _d(p['score']), _d(p['start']), _d(p['end'])),
            ]),
        ],
        _d(j['score']),
        _d(j['ms']),
      );
}

class WordPhones {
  WordPhones(this.text, this.phones, this.guessed);
  final String text;
  final List<String> phones;
  final bool guessed;

  factory WordPhones.fromJson(Map<String, dynamic> j) =>
      WordPhones(j['text'] as String, [for (final p in j['phones'] as List) p as String], j['guessed'] as bool? ?? false);
}

class Correction {
  Correction(this.original, this.corrected, this.explanation, this.kind);
  final String original, corrected, explanation, kind;

  factory Correction.fromJson(Map<String, dynamic> j) =>
      Correction(j['original'] as String, j['corrected'] as String, j['explanation'] as String, j['kind'] as String);
}

class TutorTurn {
  TutorTurn(this.reply, this.corrections, this.ms);
  final String reply;
  final List<Correction> corrections;
  final double ms;

  factory TutorTurn.fromJson(Map<String, dynamic> j) => TutorTurn(
        j['reply'] as String,
        [for (final c in j['corrections'] as List) Correction.fromJson(c as Map<String, dynamic>)],
        _d(j['ms']),
      );
}

class Turn {
  Turn(this.role, this.text);
  final String role; // 'partner' or 'learner'
  final String text;
  Map<String, dynamic> toJson() => {'role': role, 'text': text};
}

class TutorRequest {
  TutorRequest({required this.level, this.topic = '', this.history = const [], this.said = ''});
  final String level; // A2, B1, B2, C1
  final String topic;
  final List<Turn> history;
  final String said;

  Map<String, dynamic> toJson() =>
      {'level': level, 'topic': topic, 'history': [for (final t in history) t.toJson()], 'said': said};
}
