import 'package:hands_on_rl/bandit.dart';

void main() {
  final bandit = Bandit(4);

  print(bandit.qs);

  print(bandit(3));
}
