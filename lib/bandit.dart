import 'dart:math';

class Bandit {
  final int n;
  final List<double> qs;
  final Random random;
  late final double bestQ;

  Bandit(this.n, [int? seed])
    : qs = List.generate(n, (_) => Random().nextDouble()),
      random = Random(seed) {
    bestQ = qs.reduce(max);
  }

  int call(int a) => random.nextDouble() < qs[a] ? 1 : 0;
}
