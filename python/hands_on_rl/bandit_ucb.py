import numpy as np
from typing import List, Tuple


class UCB:
    def __init__(self, n: int, coef: float, init_prob=1.0):
        self.n = n
        self.hat_qs: List[Tuple[int, float]] = [(0, init_prob) for _ in range(n)]
        self.i = 0
        self.coef = coef

    def __call__(self, reward: int, action: int) -> int:
        self.i += 1
        self.hat_qs[action] = self.calc_hat_q(reward, action)
        ucb = self.hat_qs_value + self.coef * np.sqrt(
            np.log(self.i) / (2 * (self.hat_qs_count + 1))
        )  # 计算上置信界
        return int(np.argmax(ucb))

    @property
    def hat_qs_value(self):
        return np.array([q for _, q in self.hat_qs])

    @property
    def hat_qs_count(self):
        return np.array([count for count, _ in self.hat_qs])

    def calc_hat_q(self, reward: int, action: int) -> Tuple[int, float]:
        count, old_estimate = self.hat_qs[action]
        new_count = count + 1
        new_estimate = old_estimate + (1.0 / new_count) * (reward - old_estimate)
        return new_count, new_estimate
