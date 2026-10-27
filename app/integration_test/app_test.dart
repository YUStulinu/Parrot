// The app with the real engine and models, driven through its screens: a typed conversation with the
// tutor, its correction becoming a review card, and the review.
//
//   PARROT_MODELS=<models> PARROT_LLM=<model.gguf> PARROT_DATA=<empty folder> flutter test integration_test -d windows
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:parrot/app_state.dart';
import 'package:parrot/main.dart';

/// Pumps until [finder] finds something (the engine's calls take seconds).
Future<void> waitFor(WidgetTester t, Finder finder, {Duration timeout = const Duration(minutes: 3)}) async {
  final end = DateTime.now().add(timeout);
  while (DateTime.now().isBefore(end)) {
    await t.pump(const Duration(milliseconds: 200));
    if (finder.evaluate().isNotEmpty) return;
  }
  throw TestFailure('timed out waiting for $finder');
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('a typed conversation, its correction as a card, and the review', (t) async {
    expect(Platform.environment['PARROT_DATA'], isNotNull, reason: 'use an empty data folder, not your own');
    final state = await AppState.start();
    state.speakReplies = false;
    await t.pumpWidget(ParrotApp(state: state));
    await waitFor(t, find.text('Type instead'));

    await t.tap(find.text('Review'));
    await t.pumpAndSettle();
    expect(find.text('No cards yet.'), findsOneWidget);
    await t.tap(find.text('Talk'));
    await t.pumpAndSettle();

    await t.tap(find.text('Type instead'));
    final input = find.byType(TextField);
    await waitFor(t, input);
    // The partner opens the conversation.
    await waitFor(t, find.byIcon(Icons.mic)); // the status bar's listen button, once the turn is over
    await t.enterText(input, 'Yesterday I goed to the cinema with my friends.');
    await t.testTextInput.receiveAction(TextInputAction.done);
    await waitFor(t, find.textContaining(RegExp(r'^\d tips?$')));
    await t.tap(find.textContaining(RegExp(r'^\d tips?$')));
    await t.pumpAndSettle();
    expect(find.textContaining('went'), findsWidgets);
    expect(state.cards.stats().total, greaterThanOrEqualTo(1));

    await t.tap(find.text('Review'));
    await t.pumpAndSettle();
    expect(find.text('You said this. How is it said correctly?'), findsOneWidget);
    await t.tap(find.text('Show answer'));
    await t.pumpAndSettle();
    while (find.text('Good').evaluate().isNotEmpty) {
      await t.tap(find.text('Good'));
      await t.pumpAndSettle();
      if (find.text('Show answer').evaluate().isNotEmpty) {
        await t.tap(find.text('Show answer'));
        await t.pumpAndSettle();
      }
    }
    expect(find.text('All done for now.'), findsOneWidget);
    expect(state.cards.stats().reviews, state.cards.stats().total);
  });
}
