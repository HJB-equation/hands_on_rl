import numpy as np


class Bandit:
    """Multi-armed bandit with Bernoulli rewards."""

    def __init__(self, n: int):
        self.n = n
        self.qs = np.random.uniform(size=n)
        self.best_q = np.max(self.qs)

    def __call__(self, action: int) -> int:
        return 1 if np.random.random() < self.qs[action] else 0
