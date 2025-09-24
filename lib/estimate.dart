import 'package:hands_on_rl/bandit.dart';

List<double> estimate(
  Bandit bandit,
  int Function(int, int) strategy,
  int steps,
) {
  int action = 1;
  int reward = bandit(action);
  final regrets = <double>[];
  for (var t = 0; t < steps; t++) {
    action = strategy(reward, action);
    reward = bandit(action);
    regrets.add(bandit.bestQ - bandit.qs[action]);
  }
  return regrets;
}
