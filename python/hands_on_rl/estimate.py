from typing import List, Callable
from .bandit import Bandit
import numpy as np


def estimate(
    bandit: Bandit, strategy: Callable[[int, int], int], steps: int
) -> List[float]:
    action = np.random.randint(0, bandit.n - 1)
    reward = bandit(action)
    regrets = []

    for t in range(steps):
        action = strategy(reward, action)
        reward = bandit(action)
        regret = bandit.best_q - bandit.qs[action]
        regrets.append(regret)

    return regrets
