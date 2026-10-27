import 'package:flutter/material.dart';

import '../app_state.dart';
import '../srs/fsrs.dart';
import '../srs/store.dart';
import '../talk.dart';

String _when(Duration d) {
  if (d.inMinutes < 60) return '${d.inMinutes} min';
  if (d.inDays < 1) return '${d.inHours} h';
  if (d.inDays < 31) return '${d.inDays} d';
  if (d.inDays < 365) return '${(d.inDays / 30).round()} mo';
  return '${(d.inDays / 365).toStringAsFixed(1)} y';
}

class ReviewScreen extends StatefulWidget {
  const ReviewScreen({super.key, required this.state});
  final AppState state;

  @override
  State<ReviewScreen> createState() => _ReviewScreenState();
}

class _ReviewScreenState extends State<ReviewScreen> {
  bool shown = false;

  Future<void> _say(String text) async {
    final e = widget.state.engine!;
    if (e.voiceRate > 0) await playAndWait(e, await e.speak(text));
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.state;
    final due = s.cards.due(limit: 1);
    final stats = s.cards.stats();
    if (due.isEmpty) {
      return Center(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          const Icon(Icons.check_circle_outline, size: 48),
          const SizedBox(height: 12),
          Text(stats.total == 0 ? 'No cards yet.' : 'All done for now.', style: Theme.of(context).textTheme.titleMedium),
          const SizedBox(height: 4),
          Text(stats.total == 0
              ? 'Your mistakes in conversations and practice become cards here.'
              : '${stats.total} cards, ${stats.reviews} reviews so far.'),
        ]),
      );
    }
    final c = due.first;
    final preview = s.cards.preview(c);
    final pron = c.kind == 'pronunciation';
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 560),
        child: ListView(shrinkWrap: true, padding: const EdgeInsets.all(24), children: [
          Text('${stats.due} due · ${c.kind.replaceAll('_', ' ')}${c.isNew ? ' · new' : ''}',
              style: Theme.of(context).textTheme.labelLarge),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(pron ? 'Say this word:' : 'You said this. How is it said correctly?'),
                const SizedBox(height: 8),
                Row(children: [
                  Expanded(child: Text(c.front, style: const TextStyle(fontSize: 24))),
                  if (pron) IconButton(icon: const Icon(Icons.volume_up), onPressed: () => _say(c.front)),
                ]),
                if (shown) ...[
                  const Divider(height: 28),
                  Row(children: [
                    Expanded(
                      child: Text(c.back,
                          style: TextStyle(fontSize: 24, color: Theme.of(context).colorScheme.primary, fontWeight: FontWeight.bold)),
                    ),
                    if (!pron) IconButton(icon: const Icon(Icons.volume_up), onPressed: () => _say(c.back)),
                  ]),
                  if (c.note.isNotEmpty) Padding(padding: const EdgeInsets.only(top: 8), child: Text(c.note)),
                ],
              ]),
            ),
          ),
          const SizedBox(height: 16),
          if (!shown)
            FilledButton(onPressed: () => setState(() => shown = true), child: const Text('Show answer'))
          else
            Row(children: [
              for (final g in Grade.values)
                Expanded(
                  child: Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 3),
                    child: _GradeButton(grade: g, after: _when(preview[g]!), onPressed: () => _grade(c, g)),
                  ),
                ),
            ]),
        ]),
      ),
    );
  }

  void _grade(ReviewCard c, Grade g) {
    widget.state.cards.review(c.id, g);
    setState(() => shown = false);
    widget.state.cardsChanged();
  }
}

class _GradeButton extends StatelessWidget {
  const _GradeButton({required this.grade, required this.after, required this.onPressed});
  final Grade grade;
  final String after;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    final label = switch (grade) { Grade.again => 'Again', Grade.hard => 'Hard', Grade.good => 'Good', Grade.easy => 'Easy' };
    final child = Column(mainAxisSize: MainAxisSize.min, children: [Text(label), Text(after, style: const TextStyle(fontSize: 11))]);
    return grade == Grade.good
        ? FilledButton(onPressed: onPressed, child: child)
        : OutlinedButton(onPressed: onPressed, child: child);
  }
}
