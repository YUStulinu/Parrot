// FSRS-5 (Free Spaced Repetition Scheduler, https://github.com/open-spaced-repetition), with its
// published default parameters.
//
// A card's memory is two numbers: stability S (days until recall falls to 90%) and difficulty D
// (1–10). Recall after t days is R = (1 + F·t/S)^-0.5, F = 19/81. Each review updates S and D from
// the grade; the next review is when R would fall to the desired retention.
import 'dart:math';

enum Grade { again, hard, good, easy }

extension GradeValue on Grade {
  int get value => index + 1; // FSRS grades are 1–4
}

class Memory {
  const Memory(this.stability, this.difficulty);
  final double stability, difficulty;
}

class Fsrs {
  Fsrs({this.retention = 0.9, this.maxInterval = 365, List<double>? weights}) : w = weights ?? defaultWeights;

  static const defaultWeights = [
    0.40255, 1.18385, 3.173, 15.69105, 7.1949, 0.5345, 1.4604, 0.0046, 1.54575, 0.1192, //
    1.01925, 1.9395, 0.11, 0.29605, 2.2698, 0.2315, 2.9898, 0.51655, 0.6621,
  ];
  static const decay = -0.5;
  static const factor = 19 / 81; // so that R(S, S) = 0.9

  final double retention;
  final int maxInterval; // days
  final List<double> w;

  /// Probability of recall after [days] with stability [s].
  double recall(double days, double s) => pow(1 + factor * days / s, decay).toDouble();

  /// Days until recall falls to [retention].
  int interval(double s) {
    final days = s / factor * (pow(retention, 1 / decay) - 1);
    return days.round().clamp(1, maxInterval);
  }

  double _d0(Grade g) => (w[4] - exp(w[5] * (g.value - 1)) + 1).clamp(1, 10).toDouble();

  Memory first(Grade g) => Memory(max(w[g.value - 1], 0.1), _d0(g));

  /// The memory after a review [days] after the last one.
  Memory next(Memory m, Grade g, double days) {
    final s = m.stability, d = m.difficulty;
    // Difficulty: moves with the grade, less as it nears 10, and reverts a little to the "easy" start.
    final dd = d - w[6] * (g.value - 3) * (10 - d) / 9;
    final nd = (w[7] * _d0(Grade.easy) + (1 - w[7]) * dd).clamp(1, 10).toDouble();
    double ns;
    if (days < 1) {
      // A review on the same day: stability changes only a little.
      ns = s * exp(w[17] * (g.value - 3 + w[18]));
    } else {
      final r = recall(days, s);
      if (g == Grade.again) {
        ns = min(s, w[11] * pow(d, -w[12]) * (pow(s + 1, w[13]) - 1) * exp(w[14] * (1 - r)));
      } else {
        final hard = g == Grade.hard ? w[15] : 1.0;
        final easy = g == Grade.easy ? w[16] : 1.0;
        ns = s * (exp(w[8]) * (11 - d) * pow(s, -w[9]) * (exp(w[10] * (1 - r)) - 1) * hard * easy + 1);
      }
    }
    return Memory(max(ns, 0.1), nd);
  }
}
