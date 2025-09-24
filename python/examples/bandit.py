from hands_on_rl.bandit import Bandit
from hands_on_rl.greedy import EpsilonGreedy, DecayingEpsilonGreedy
import numpy as np
from hands_on_rl.estimate import estimate
from hands_on_rl.figure import plot_regrets
from hands_on_rl.ucb import UCB
from hands_on_rl.thompson import ThompsonSampling

import matplotlib.pyplot as plt

n_arms = 10
np.random.seed(1)
bandit = Bandit(n=n_arms)

steps = 5000


plot_regrets(
    {
        r"$\epsilon = 0.0001$": estimate(
            bandit, EpsilonGreedy(n=n_arms, epsilon=0.0001), steps=steps
        ),
        r"$\epsilon = 0.01$": estimate(
            bandit, EpsilonGreedy(n=n_arms, epsilon=0.01), steps=steps
        ),
        r"$\epsilon = 0.01$": estimate(
            bandit, EpsilonGreedy(n=n_arms, epsilon=0.01), steps=steps
        ),
        r"$\epsilon = 0.25$": estimate(
            bandit, EpsilonGreedy(n=n_arms, epsilon=0.25), steps=steps
        ),
        r"$\epsilon = 0.5$": estimate(
            bandit, EpsilonGreedy(n=n_arms, epsilon=0.5), steps=steps
        ),
        r"$\epsilon = \frac{1}{t}$": estimate(
            bandit, DecayingEpsilonGreedy(n=n_arms), steps=steps
        ),
    }
)

plot_regrets(
    {
        r"$\epsilon = \frac{1}{t}$": estimate(
            bandit, DecayingEpsilonGreedy(n=n_arms), steps=steps
        ),
    }
)


plot_regrets(
    {
        r"$UCB = (p = \frac{1}{t}, c = 1)$": estimate(
            bandit, UCB(n=n_arms, coef=1.0), steps=steps
        ),
    }
)


plot_regrets(
    {
        r"ThompsonSampling": estimate(bandit, ThompsonSampling(n=n_arms), steps=steps),
    }
)
