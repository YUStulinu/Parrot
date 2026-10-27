// Review cards and their history in SQLite.
//
// A card comes from a mistake: a correction from the conversation (what you said → what is right)
// or a word you mispronounced (the word and its weak sounds). Making the same mistake again brings
// its card back today. Every review is kept, with the memory it led to.
import 'package:sqlite3/sqlite3.dart';

import 'fsrs.dart';

class ReviewCard {
  ReviewCard({
    required this.id,
    required this.kind,
    required this.front,
    required this.back,
    required this.note,
    required this.due,
    required this.stability,
    required this.difficulty,
    required this.reps,
    required this.lapses,
    required this.lastReview,
  });

  final int id;
  final String kind; // grammar, vocabulary, word_choice, pronunciation
  final String front; // the prompt: what you said, or the word to say
  final String back; // the answer: the correction, or the sounds to watch
  final String note; // the explanation
  final DateTime due;
  final double stability, difficulty; // 0 before the first review
  final int reps, lapses;
  final DateTime? lastReview;

  bool get isNew => reps == 0;
}

class CardStore {
  CardStore(this.db, {Fsrs? fsrs}) : fsrs = fsrs ?? Fsrs() {
    db.execute('PRAGMA foreign_keys = ON');
    db.execute('''
      CREATE TABLE IF NOT EXISTS cards (
        id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL,
        front TEXT NOT NULL,
        back TEXT NOT NULL,
        note TEXT NOT NULL DEFAULT '',
        created INTEGER NOT NULL,
        due INTEGER NOT NULL,
        stability REAL NOT NULL DEFAULT 0,
        difficulty REAL NOT NULL DEFAULT 0,
        reps INTEGER NOT NULL DEFAULT 0,
        lapses INTEGER NOT NULL DEFAULT 0,
        seen INTEGER NOT NULL DEFAULT 1,
        last_review INTEGER,
        UNIQUE (kind, front)
      )''');
    db.execute('''
      CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY,
        card INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
        at INTEGER NOT NULL,
        grade INTEGER NOT NULL,
        elapsed_days REAL NOT NULL,
        stability REAL NOT NULL,
        difficulty REAL NOT NULL
      )''');
    db.execute('CREATE INDEX IF NOT EXISTS cards_due ON cards(due)');
  }

  factory CardStore.open(String path) => CardStore(sqlite3.open(path));
  factory CardStore.memory() => CardStore(sqlite3.openInMemory());

  final Database db;
  final Fsrs fsrs;

  /// Adds the card for a mistake, or brings its card back today if it was already there.
  /// Returns true for a new card.
  bool addMistake({required String kind, required String front, required String back, String note = '', DateTime? now}) {
    final t = (now ?? DateTime.now()).millisecondsSinceEpoch;
    final existing = db.select('SELECT id FROM cards WHERE kind = ? AND front = ?', [kind, front]);
    if (existing.isNotEmpty) {
      db.execute('UPDATE cards SET due = min(due, ?), seen = seen + 1, back = ?, note = ? WHERE id = ?',
          [t, back, note, existing.first['id']]);
      return false;
    }
    db.execute('INSERT INTO cards (kind, front, back, note, created, due) VALUES (?, ?, ?, ?, ?, ?)',
        [kind, front, back, note, t, t]);
    return true;
  }

  ReviewCard _card(Row r) => ReviewCard(
        id: r['id'] as int,
        kind: r['kind'] as String,
        front: r['front'] as String,
        back: r['back'] as String,
        note: r['note'] as String,
        due: DateTime.fromMillisecondsSinceEpoch(r['due'] as int),
        stability: (r['stability'] as num).toDouble(),
        difficulty: (r['difficulty'] as num).toDouble(),
        reps: r['reps'] as int,
        lapses: r['lapses'] as int,
        lastReview: r['last_review'] == null ? null : DateTime.fromMillisecondsSinceEpoch(r['last_review'] as int),
      );

  List<ReviewCard> due({DateTime? now, int limit = 50}) => [
        for (final r in db.select('SELECT * FROM cards WHERE due <= ? ORDER BY due LIMIT ?',
            [(now ?? DateTime.now()).millisecondsSinceEpoch, limit]))
          _card(r),
      ];

  List<ReviewCard> all() => [for (final r in db.select('SELECT * FROM cards ORDER BY created DESC')) _card(r)];

  ReviewCard? byId(int id) {
    final r = db.select('SELECT * FROM cards WHERE id = ?', [id]);
    return r.isEmpty ? null : _card(r.first);
  }

  /// Records a review and schedules the card's next one. "Again" brings it back in ten minutes.
  ReviewCard review(int id, Grade grade, {DateTime? now}) {
    final c = byId(id)!;
    final t = now ?? DateTime.now();
    final days = c.lastReview == null ? 0.0 : t.difference(c.lastReview!).inMinutes / (24 * 60);
    final m = c.isNew ? fsrs.first(grade) : fsrs.next(Memory(c.stability, c.difficulty), grade, days);
    final due = grade == Grade.again ? t.add(const Duration(minutes: 10)) : t.add(Duration(days: fsrs.interval(m.stability)));
    db.execute('BEGIN');
    try {
      db.execute('INSERT INTO reviews (card, at, grade, elapsed_days, stability, difficulty) VALUES (?, ?, ?, ?, ?, ?)',
          [id, t.millisecondsSinceEpoch, grade.value, days, m.stability, m.difficulty]);
      db.execute(
          'UPDATE cards SET stability = ?, difficulty = ?, due = ?, reps = reps + 1, lapses = lapses + ?, last_review = ? '
          'WHERE id = ?',
          [m.stability, m.difficulty, due.millisecondsSinceEpoch, grade == Grade.again && !c.isNew ? 1 : 0, t.millisecondsSinceEpoch, id]);
      db.execute('COMMIT');
    } catch (_) {
      db.execute('ROLLBACK');
      rethrow;
    }
    return byId(id)!;
  }

  /// The next due date for each grade, for the buttons.
  Map<Grade, Duration> preview(ReviewCard c, {DateTime? now}) {
    final t = now ?? DateTime.now();
    final days = c.lastReview == null ? 0.0 : t.difference(c.lastReview!).inMinutes / (24 * 60);
    return {
      for (final g in Grade.values)
        g: g == Grade.again
            ? const Duration(minutes: 10)
            : Duration(
                days: fsrs.interval((c.isNew ? fsrs.first(g) : fsrs.next(Memory(c.stability, c.difficulty), g, days)).stability)),
    };
  }

  ({int total, int due, int reviews}) stats({DateTime? now}) {
    final t = (now ?? DateTime.now()).millisecondsSinceEpoch;
    final r = db.select('SELECT (SELECT count(*) FROM cards) AS total, (SELECT count(*) FROM cards WHERE due <= ?) AS due, '
        '(SELECT count(*) FROM reviews) AS reviews', [t]);
    return (total: r.first['total'] as int, due: r.first['due'] as int, reviews: r.first['reviews'] as int);
  }

  void delete(int id) => db.execute('DELETE FROM cards WHERE id = ?', [id]);

  void close() => db.close();
}
