import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

def plot_regrets(datas: dict[str, list[float]], save_file = 'figures/default.png') -> None:
    for label, regrets in datas.items():
        plt.plot(range(len(regrets)), np.cumsum(regrets), label=label)
    plt.xlabel("Time Steps")
    plt.ylabel("Cumulative Regret")
    plt.title("Bandit Strategy")
    plt.legend()
    plt.savefig(save_file)
    plt.clf()
