import matplotlib.pyplot as plt
import numpy as np


def plot_regrets(datas: dict[str, list[float]]) -> None:
    for label, regrets in datas.items():
        plt.plot(range(len(regrets)), np.cumsum(regrets), label=label)
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Bandit Strategy")
    plt.legend()
    plt.show()
