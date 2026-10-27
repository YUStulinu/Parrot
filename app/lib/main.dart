import 'package:flutter/material.dart';

import 'app_state.dart';
import 'ui/practice_screen.dart';
import 'ui/review_screen.dart';
import 'ui/talk_screen.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final state = await AppState.start();
  runApp(ParrotApp(state: state));
}

class ParrotApp extends StatelessWidget {
  const ParrotApp({super.key, required this.state});
  final AppState state;

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF2E7D5B);
    return MaterialApp(
      title: 'Parrot',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: seed, useMaterial3: true),
      darkTheme: ThemeData(colorSchemeSeed: seed, brightness: Brightness.dark, useMaterial3: true),
      home: Home(state: state),
    );
  }
}

class Home extends StatefulWidget {
  const Home({super.key, required this.state});
  final AppState state;

  @override
  State<Home> createState() => _HomeState();
}

class _HomeState extends State<Home> {
  int tab = 0;

  @override
  Widget build(BuildContext context) {
    final s = widget.state;
    return ListenableBuilder(
      listenable: s,
      builder: (context, _) {
        final due = s.cards.stats().due;
        return Scaffold(
          appBar: AppBar(
            title: const Text('Parrot'),
            actions: [
              Padding(
                padding: const EdgeInsets.only(right: 8),
                child: SegmentedButton<String>(
                  segments: [for (final l in levels) ButtonSegment(value: l, label: Text(l))],
                  selected: {s.level},
                  showSelectedIcon: false,
                  onSelectionChanged: (v) => s.level = v.first,
                ),
              ),
            ],
          ),
          body: !s.ready
              ? _Loading(error: s.engineError)
              : IndexedStack(index: tab, children: [
                  TalkScreen(state: s),
                  PracticeScreen(state: s),
                  ReviewScreen(state: s),
                ]),
          bottomNavigationBar: NavigationBar(
            selectedIndex: tab,
            onDestinationSelected: (i) => setState(() => tab = i),
            destinations: [
              const NavigationDestination(icon: Icon(Icons.forum_outlined), selectedIcon: Icon(Icons.forum), label: 'Talk'),
              const NavigationDestination(
                  icon: Icon(Icons.record_voice_over_outlined), selectedIcon: Icon(Icons.record_voice_over), label: 'Practice'),
              NavigationDestination(
                icon: Badge(isLabelVisible: due > 0, label: Text('$due'), child: const Icon(Icons.style_outlined)),
                selectedIcon: const Icon(Icons.style),
                label: 'Review',
              ),
            ],
          ),
        );
      },
    );
  }
}

class _Loading extends StatelessWidget {
  const _Loading({this.error});
  final String? error;

  @override
  Widget build(BuildContext context) {
    if (error != null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(mainAxisSize: MainAxisSize.min, children: [
            Icon(Icons.error_outline, size: 48, color: Theme.of(context).colorScheme.error),
            const SizedBox(height: 12),
            const Text('The models could not be loaded.', style: TextStyle(fontWeight: FontWeight.bold)),
            const SizedBox(height: 8),
            SelectableText(error!, textAlign: TextAlign.center),
          ]),
        ),
      );
    }
    return const Center(
      child: Column(mainAxisSize: MainAxisSize.min, children: [
        CircularProgressIndicator(),
        SizedBox(height: 16),
        Text('Loading the speech models…'),
      ]),
    );
  }
}
