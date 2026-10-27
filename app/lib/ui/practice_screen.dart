import 'package:flutter/material.dart';

import '../app_state.dart';
import '../engine/results.dart';
import '../phones.dart';
import '../sentences.dart';
import '../srs/mistakes.dart';
import '../talk.dart';

/// 0–100 → red, amber, green.
Color scoreColor(double score, ColorScheme cs) {
  if (score >= 75) return Colors.green.shade600;
  if (score >= 50) return Colors.orange.shade700;
  return cs.error;
}

class PracticeScreen extends StatefulWidget {
  const PracticeScreen({super.key, required this.state});
  final AppState state;

  @override
  State<PracticeScreen> createState() => _PracticeScreenState();
}

class _PracticeScreenState extends State<PracticeScreen> {
  int index = 0;
  final custom = TextEditingController();
  String? sentence;
  bool recording = false, busy = false;
  double level = 0;
  PronunciationResult? result;
  WordScore? selected;
  String? message;

  List<String> get list => practiceSentences[widget.state.level]!;
  String get current => sentence ?? list[index % list.length];

  @override
  void dispose() {
    custom.dispose();
    super.dispose();
  }

  Future<void> _listen() async {
    final e = widget.state.engine!;
    if (!await widget.state.microphone()) return;
    setState(() {
      recording = true;
      result = null;
      selected = null;
      message = null;
    });
    final audio = await listenOnce(e, timeout: const Duration(seconds: 15), onLevel: (l, _) => setState(() => level = l));
    setState(() {
      recording = false;
      busy = audio != null;
      if (audio == null) message = 'Nothing heard. Try again, a little closer to the microphone.';
    });
    if (audio == null) return;
    try {
      final r = await e.score(audio, current);
      final added = addPronunciation(widget.state.cards, r);
      if (added > 0) widget.state.cardsChanged();
      setState(() {
        result = r;
        message = added > 0 ? '$added word${added > 1 ? 's' : ''} added to your review cards.' : null;
      });
    } catch (err) {
      setState(() => message = '$err');
    }
    setState(() => busy = false);
  }

  Future<void> _hear(String text) async {
    final e = widget.state.engine!;
    if (e.voiceRate == 0) return;
    setState(() => busy = true);
    try {
      await playAndWait(e, await e.speak(text));
    } finally {
      if (mounted) setState(() => busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final r = result;
    return ListView(padding: const EdgeInsets.all(16), children: [
      Text('Read this aloud', style: Theme.of(context).textTheme.titleMedium),
      const SizedBox(height: 8),
      Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: r == null
              ? Text(current, style: const TextStyle(fontSize: 22))
              : Wrap(spacing: 6, runSpacing: 6, children: [
                  for (final w in r.words)
                    InkWell(
                      onTap: () => setState(() => selected = w),
                      borderRadius: BorderRadius.circular(6),
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                        decoration: BoxDecoration(
                          border: Border(bottom: BorderSide(color: scoreColor(w.score, cs), width: 3)),
                          color: selected == w ? cs.secondaryContainer : null,
                        ),
                        child: Text(w.text, style: TextStyle(fontSize: 22, color: scoreColor(w.score, cs))),
                      ),
                    ),
                ]),
        ),
      ),
      const SizedBox(height: 12),
      Row(children: [
        FilledButton.icon(
          onPressed: recording || busy ? null : _listen,
          icon: Icon(recording ? Icons.graphic_eq : Icons.mic),
          label: Text(recording ? 'Listening…' : 'Read it'),
        ),
        const SizedBox(width: 8),
        OutlinedButton.icon(
          onPressed: recording || busy ? null : () => _hear(current),
          icon: const Icon(Icons.volume_up),
          label: const Text('Hear it'),
        ),
        const Spacer(),
        IconButton(
          tooltip: 'Next sentence',
          onPressed: recording || busy
              ? null
              : () => setState(() {
                    sentence = null;
                    index++;
                    result = null;
                    selected = null;
                    message = null;
                  }),
          icon: const Icon(Icons.skip_next),
        ),
      ]),
      if (recording) Padding(padding: const EdgeInsets.only(top: 8), child: LinearProgressIndicator(value: level)),
      if (busy && !recording) const Padding(padding: EdgeInsets.only(top: 8), child: LinearProgressIndicator()),
      if (message != null) Padding(padding: const EdgeInsets.only(top: 8), child: Text(message!)),
      if (r != null) ...[
        const SizedBox(height: 16),
        Row(children: [
          Text('Score ', style: Theme.of(context).textTheme.titleMedium),
          Text('${r.score.round()}', style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: scoreColor(r.score, cs))),
          const Text(' / 100'),
          const Spacer(),
          Text('tap a word for its sounds', style: Theme.of(context).textTheme.bodySmall),
        ]),
      ],
      if (selected != null) _WordDetail(word: selected!, onHear: () => _hear(selected!.text)),
      const SizedBox(height: 24),
      TextField(
        controller: custom,
        decoration: const InputDecoration(labelText: 'Or practise your own sentence', border: OutlineInputBorder()),
        onSubmitted: (v) => setState(() {
          sentence = v.trim().isEmpty ? null : v.trim();
          result = null;
          selected = null;
        }),
      ),
    ]);
  }
}

class _WordDetail extends StatelessWidget {
  const _WordDetail({required this.word, required this.onHear});
  final WordScore word;
  final VoidCallback onHear;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final weak = weakSounds(word);
    return Card(
      margin: const EdgeInsets.only(top: 12),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Row(children: [
            Text(word.text, style: Theme.of(context).textTheme.titleLarge),
            const SizedBox(width: 8),
            Text('${word.score.round()}', style: TextStyle(color: scoreColor(word.score, cs), fontWeight: FontWeight.bold)),
            const Spacer(),
            IconButton(icon: const Icon(Icons.volume_up), onPressed: onHear),
          ]),
          if (word.guessed) const Text('This word is not in the dictionary: its sounds are a guess.'),
          const SizedBox(height: 8),
          Wrap(spacing: 4, runSpacing: 4, children: [
            for (final p in word.phones)
              Tooltip(
                message: '${describe(p.phone)} · ${p.score.round()}',
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                  decoration: BoxDecoration(
                    color: scoreColor(p.score, cs).withValues(alpha: 0.15),
                    border: Border.all(color: scoreColor(p.score, cs)),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text('/${phoneIpa[p.phone] ?? p.phone}/', style: const TextStyle(fontSize: 18)),
                ),
              ),
          ]),
          if (weak.isNotEmpty) ...[
            const SizedBox(height: 8),
            for (final w in weak) Text('• $w'),
          ],
        ]),
      ),
    );
  }
}
