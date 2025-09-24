import numpy as np


class ThompsonSampling:
    """汤普森采样算法,继承Solver类"""

    def __init__(self, n: int):
        self._a = np.ones(n)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.ones(n)  # 列表,表示每根拉杆奖励为0的次数

    def __call__(self, reward: int, action: int) -> int:
        self._a[action] += reward
        self._b[action] += 1 - reward
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本

        return int(np.argmax(samples))
