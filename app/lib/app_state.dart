// The app's state: where things are, the engine (loading in the background), the cards, settings.
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:path/path.dart' as p;
import 'package:sqlite3/sqlite3.dart';

import 'engine/engine.dart';
import 'srs/store.dart';

const levels = ['A2', 'B1', 'B2', 'C1'];

class Folders {
  Folders(this.data, this.models);
  final String data; // the database
  final String models; // the models, the language model among them

  static const _platform = MethodChannel('parrot/platform');

  /// Windows: %APPDATA%\Parrot (or PARROT_DATA), models from PARROT_MODELS, a `models` folder next to the program, or the
  /// data folder's. Android: the app's files; models in its external folder (where `adb push` can put
  /// them) or its internal one.
  static Future<Folders> find() async {
    final env = Platform.environment;
    if (Platform.isAndroid) {
      final f = await _platform.invokeMapMethod<String, String?>('folders');
      final data = f!['data']!;
      // The app makes the external models folder itself: one that `adb shell mkdir` made belongs to the
      // shell user, and the app could not list it.
      if (f['external'] != null) {
        try {
          Directory(p.join(f['external']!, 'models')).createSync(recursive: true);
        } on FileSystemException {
          // not writable: the internal folder below
        }
      }
      final candidates = [if (f['external'] != null) p.join(f['external']!, 'models'), p.join(data, 'models')];
      return Folders(data, candidates.firstWhere((c) => Directory(c).existsSync(), orElse: () => candidates.first));
    }
    final base = env['APPDATA'] ?? env['XDG_DATA_HOME'] ?? p.join(env['HOME'] ?? '.', '.local', 'share');
    final data = env['PARROT_DATA'] ?? p.join(base, 'Parrot');
    Directory(data).createSync(recursive: true);
    final candidates = [
      if (env['PARROT_MODELS'] != null) env['PARROT_MODELS']!,
      p.join(p.dirname(Platform.resolvedExecutable), 'models'),
      p.join(data, 'models'),
    ];
    return Folders(data, candidates.firstWhere((c) => Directory(c).existsSync(), orElse: () => candidates.last));
  }

  /// The language model: PARROT_LLM, or the first .gguf file among the models.
  String? languageModel() {
    final env = Platform.environment['PARROT_LLM'];
    if (env != null && File(env).existsSync()) return env;
    final dir = Directory(models);
    if (!dir.existsSync()) return null;
    final ggufs = dir.listSync().whereType<File>().where((f) => f.path.endsWith('.gguf')).map((f) => f.path).toList()..sort();
    return ggufs.isEmpty ? null : ggufs.first;
  }

  /// The speech model: the small one on a desktop if present, the base one otherwise.
  String whisper() {
    final small = p.join(models, 'ggml-small.en-q5_1.bin');
    return !Platform.isAndroid && File(small).existsSync() ? small : p.join(models, 'ggml-base.en-q5_1.bin');
  }
}

class Settings {
  Settings(this._db) {
    _db.execute('CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)');
  }
  final Database _db;

  String get(String key, String fallback) {
    final r = _db.select('SELECT value FROM settings WHERE key = ?', [key]);
    return r.isEmpty ? fallback : r.first['value'] as String;
  }

  void set(String key, String value) =>
      _db.execute('INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value', [key, value]);
}

class AppState extends ChangeNotifier {
  AppState._(this.folders, this.cards, this.settings);

  final Folders folders;
  final CardStore cards;
  final Settings settings;
  Engine? engine;
  String? engineError;
  bool get ready => engine != null;
  bool get hasTutor => folders.languageModel() != null;

  static Future<AppState> start() async {
    final folders = await Folders.find();
    final db = sqlite3.open(p.join(folders.data, 'parrot.db'));
    final s = AppState._(folders, CardStore(db), Settings(db));
    s._load();
    return s;
  }

  Future<void> _load() async {
    try {
      engine = await Engine.open(modelsDir: folders.models, llm: folders.languageModel(), whisper: folders.whisper());
    } catch (e) {
      engineError = '$e\n\nModels folder: ${folders.models}';
    }
    notifyListeners();
  }

  String get level => settings.get('level', 'B1');
  set level(String v) {
    settings.set('level', v);
    notifyListeners();
  }

  bool get speakReplies => settings.get('speak', '1') == '1';
  set speakReplies(bool v) {
    settings.set('speak', v ? '1' : '0');
    notifyListeners();
  }

  /// Cards changed (added or reviewed).
  void cardsChanged() => notifyListeners();

  /// Asks for the microphone where the system needs it (Android).
  Future<bool> microphone() async {
    if (!Platform.isAndroid) return true;
    return await Folders._platform.invokeMethod<bool>('microphone') ?? false;
  }
}
