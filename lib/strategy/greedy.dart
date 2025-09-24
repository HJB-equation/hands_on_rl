import 'dart:math';

class EpsilonGreedy {
  final int n;
  final List<(int, double)> hatQs;
  final double epsilon;
  final Random random;

  EpsilonGreedy(this.n, {this.epsilon = 0.1, int? seed})
    : hatQs = List.filled(n, (0, 1.0)),
      random = Random(seed);

  int call(int reward, int action) {
    hatQs[action] = () {
      final counts = hatQs[action].$1 + 1;
      return (
        counts,
        // 期望增量公式，与直接求期望一样
        hatQs[action].$2 + 1.0 / counts * (reward - hatQs[action].$2),
      );
    }();

    return random.nextDouble() < epsilon
        ? random.nextInt(n)
        : argMax(hatQs.map((e) => e.$2));
  }
}

int argMax(Iterable<double> xs) {
  var result = (-1, 0.0);
  for (final times in xs.indexed) {
    if (times.$2 > result.$2) {
      result = times;
    }
  }
  return result.$1;
}
