from typing import List, Tuple, Optional, Iterable
import numpy as np


class EpsilonGreedy:
    """Epsilon-greedy strategy for multi-armed bandits."""

    def __init__(self, n: int, epsilon: float = 0.1):
        self.n = n
        self.epsilon = epsilon
        self.hat_qs: List[Tuple[int, float]] = [(0, 1.0) for _ in range(n)]

    def __call__(self, reward: int, action: int) -> int:
        self.hat_qs[action] = self.calc_hat_q(reward, action)

        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n - 1)
        else:
            return int(np.argmax(np.array([q for _, q in self.hat_qs])))

    def calc_hat_q(self, reward: int, action: int) -> Tuple[int, float]:
        count, old_estimate = self.hat_qs[action]
        new_count = count + 1
        new_estimate = old_estimate + (1.0 / new_count) * (reward - old_estimate)
        return new_count, new_estimate


class DecayingEpsilonGreedy:
    """Epsilon-greedy strategy for multi-armed bandits."""

    def __init__(self, n: int):
        self.n = n
        self.hat_qs: List[Tuple[int, float]] = [(0, 1.0) for _ in range(n)]

    def __call__(self, reward: int, action: int) -> int:
        count, old_estimate = self.hat_qs[action]
        new_count = count + 1
        new_estimate = old_estimate + (1.0 / new_count) * (reward - old_estimate)
        self.hat_qs[action] = (new_count, new_estimate)

        if np.random.random() < 1.0 / new_count:
            return np.random.randint(0, self.n - 1)
        else:
            return int(np.argmax(np.array([q for _, q in self.hat_qs])))
