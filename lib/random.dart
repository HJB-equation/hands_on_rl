import 'dart:math' as math;

class Random {
  math.Random? __random;

  set seed(int? seed) {
    __random = math.Random(seed);
  }

  math.Random get _random => __random ??= math.Random();

  int nextInt(int max) => _random.nextInt(max);
  double nextDouble() => _random.nextDouble();
  bool nextBool() => _random.nextBool();
}

final random = Random();
