import 'package:flutter/material.dart';

import '../app_state.dart';
import '../engine/results.dart';
import '../sentences.dart';
import '../talk.dart';

class TalkScreen extends StatefulWidget {
  const TalkScreen({super.key, required this.state});
  final AppState state;

  @override
  State<TalkScreen> createState() => _TalkScreenState();
}

class _TalkScreenState extends State<TalkScreen> {
  TalkSession? session;
  String topic = '';
  final input = TextEditingController();
  final scroll = ScrollController();

  @override
  void dispose() {
    session?.dispose();
    input.dispose();
    scroll.dispose();
    super.dispose();
  }

  Future<void> _begin({bool listen = true}) async {
    final s = widget.state;
    if (listen && !await s.microphone()) {
      if (mounted) ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Parrot needs the microphone to listen.')));
    }
    session?.dispose();
    final t = TalkSession(s.engine!, s.cards, level: s.level, topic: topic, speak: s.speakReplies, onCards: s.cardsChanged);
    t.addListener(_follow);
    setState(() => session = t);
    t.start(listen: listen);
  }

  void _follow() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (scroll.hasClients) scroll.animateTo(scroll.position.maxScrollExtent, duration: const Duration(milliseconds: 200), curve: Curves.easeOut);
    });
  }

  @override
  Widget build(BuildContext context) {
    final s = widget.state;
    if (!s.hasTutor) {
      return const _Empty(
        icon: Icons.psychology_alt_outlined,
        text: 'No language model found.\nPut a .gguf model (for example Qwen2.5-3B-Instruct, Q4_K_M) in the models folder.',
      );
    }
    final t = session;
    if (t == null) return _start(context);
    return ListenableBuilder(
      listenable: t,
      builder: (context, _) => Column(children: [
        Expanded(
          child: ListView(
            controller: scroll,
            padding: const EdgeInsets.all(16),
            children: [
              for (final m in t.messages) _Bubble(message: m),
              if (t.error != null) Text(t.error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
            ],
          ),
        ),
        _StatusBar(session: t, onNew: () => setState(() => session = null)),
        Padding(
          padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
          child: TextField(
            controller: input,
            decoration: InputDecoration(
              hintText: 'Or type what you want to say',
              border: const OutlineInputBorder(),
              isDense: true,
              suffixIcon: IconButton(icon: const Icon(Icons.send), onPressed: _send),
            ),
            onSubmitted: (_) => _send(),
          ),
        ),
      ]),
    );
  }

  void _send() {
    final text = input.text;
    input.clear();
    session?.send(text);
  }

  Widget _start(BuildContext context) {
    final s = widget.state;
    return Center(
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 480),
        child: ListView(shrinkWrap: true, padding: const EdgeInsets.all(24), children: [
          Text('Talk with Parrot', style: Theme.of(context).textTheme.headlineSmall, textAlign: TextAlign.center),
          const SizedBox(height: 8),
          Text(
            'A conversation at level ${s.level}. Just speak: Parrot listens until you pause, answers, and notes your '
            'mistakes quietly beside what you said. They become review cards.',
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 20),
          Wrap(spacing: 8, runSpacing: 8, alignment: WrapAlignment.center, children: [
            for (final x in topics)
              ChoiceChip(label: Text(x.isEmpty ? 'Anything' : x), selected: topic == x, onSelected: (_) => setState(() => topic = x)),
          ]),
          const SizedBox(height: 12),
          SwitchListTile(
            title: const Text('Speak the replies'),
            value: s.speakReplies,
            onChanged: (v) => setState(() => s.speakReplies = v),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(onPressed: _begin, icon: const Icon(Icons.mic), label: const Text('Start talking')),
          const SizedBox(height: 8),
          TextButton.icon(
            onPressed: () => _begin(listen: false),
            icon: const Icon(Icons.keyboard),
            label: const Text('Type instead'),
          ),
        ]),
      ),
    );
  }
}

class _StatusBar extends StatelessWidget {
  const _StatusBar({required this.session, required this.onNew});
  final TalkSession session;
  final VoidCallback onNew;

  @override
  Widget build(BuildContext context) {
    final t = session;
    final cs = Theme.of(context).colorScheme;
    final label = switch (t.phase) {
      TalkPhase.idle => 'Paused',
      TalkPhase.thinking => 'Parrot is thinking…',
      TalkPhase.speaking => 'Parrot is speaking',
      TalkPhase.listening => t.speaking ? 'Listening…' : 'Your turn: speak',
      TalkPhase.hearing => 'Writing down what you said…',
    };
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 8),
      child: Row(children: [
        IconButton.filled(
          icon: Icon(t.running ? Icons.pause : Icons.mic),
          tooltip: t.running ? 'Pause' : 'Listen',
          onPressed: t.running ? t.stop : t.start,
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(label),
            const SizedBox(height: 4),
            LinearProgressIndicator(
              value: t.phase == TalkPhase.listening ? t.micLevel : (t.phase == TalkPhase.idle ? 0 : null),
              color: t.speaking ? cs.primary : cs.outline,
            ),
          ]),
        ),
        if (t.newCards > 0) Padding(padding: const EdgeInsets.only(left: 12), child: Chip(label: Text('+${t.newCards} cards'))),
        IconButton(icon: const Icon(Icons.refresh), tooltip: 'New conversation', onPressed: onNew),
      ]),
    );
  }
}

class _Bubble extends StatefulWidget {
  const _Bubble({required this.message});
  final Message message;

  @override
  State<_Bubble> createState() => _BubbleState();
}

class _BubbleState extends State<_Bubble> {
  bool open = false;

  @override
  Widget build(BuildContext context) {
    final m = widget.message;
    final cs = Theme.of(context).colorScheme;
    final mine = m.role == 'learner';
    return Align(
      alignment: mine ? Alignment.centerRight : Alignment.centerLeft,
      child: ConstrainedBox(
        constraints: const BoxConstraints(maxWidth: 560),
        child: Column(crossAxisAlignment: mine ? CrossAxisAlignment.end : CrossAxisAlignment.start, children: [
          Container(
            margin: const EdgeInsets.symmetric(vertical: 4),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: mine ? cs.primaryContainer : cs.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Text(m.text, style: const TextStyle(fontSize: 16)),
          ),
          if (m.corrections.isNotEmpty)
            InkWell(
              onTap: () => setState(() => open = !open),
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
                child: Text(
                  open ? 'Hide tips' : '${m.corrections.length} tip${m.corrections.length > 1 ? 's' : ''}',
                  style: TextStyle(color: cs.tertiary, fontSize: 13),
                ),
              ),
            ),
          if (open) for (final c in m.corrections) _CorrectionView(c),
        ]),
      ),
    );
  }
}

class _CorrectionView extends StatelessWidget {
  const _CorrectionView(this.c);
  final Correction c;

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    return Card(
      margin: const EdgeInsets.only(bottom: 6),
      child: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text.rich(TextSpan(children: [
            TextSpan(text: c.original, style: TextStyle(decoration: TextDecoration.lineThrough, color: cs.error)),
            const TextSpan(text: '  →  '),
            TextSpan(text: c.corrected, style: TextStyle(color: cs.primary, fontWeight: FontWeight.bold)),
          ])),
          if (c.explanation.isNotEmpty) Padding(padding: const EdgeInsets.only(top: 4), child: Text(c.explanation)),
          Text(c.kind.replaceAll('_', ' '), style: Theme.of(context).textTheme.labelSmall),
        ]),
      ),
    );
  }
}

class _Empty extends StatelessWidget {
  const _Empty({required this.icon, required this.text});
  final IconData icon;
  final String text;

  @override
  Widget build(BuildContext context) => Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Icon(icon, size: 48),
            const SizedBox(height: 12),
            Text(text, textAlign: TextAlign.center),
          ]),
        ),
      );
}
