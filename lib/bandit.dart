import 'random.dart' as r;
import 'dart:math';

class Bandit {
  final int n;
  final List<double> qs;
  late final double bestQ;

  Bandit(this.n) : qs = List.generate(n, (_) => r.random.nextDouble()) {
    bestQ = qs.reduce(max);
  }

  int call(int a) => r.random.nextDouble() < qs[a] ? 1 : 0;
}
