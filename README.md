# 动手学习强化学习

[code](https://github.com/boyu-ai/Hands-on-RL) | [book](https://hrl.boyuai.com/) | [slides](https://hrl.boyuai.com/slides)

参考书：[Reinforcement Learning: An Introduction](http://incompleteideas.net/book/RLbook2020.pdf)  
参考课程：  
- [UCL David Silver RL Course](https://www.davidsilver.uk/teaching/)  
- [Berkeley Sergey Levine Deep RL Course](http://rail.eecs.berkeley.edu/deeprlcourse/)  
- [OpenAI DRL Camp](https://sites.google.com/view/deep-rl-bootcamp/lectures)  
- [RL China Camp](http://rlchina.org/)

## TLDR

### 01

序贯决策（sequential decision making）任务

预测仅仅产生一个针对输入数据的信号，并期望它和未来可观测到的信号一致，这不会使未来情况发生任何改变。

如果决策也是单轮的，那么它可以转化为“判别最优动作”的预测任务。但是当前轮带来最大奖励反馈的动作，在长期来看并不一定是最优的。

> 我是这么理解概率的，如果有个隐变量在其作用，我们是无法刻画它的行为的，只能通过观测，而观测就是采样。有点类似量子力学，然后用概率描述

环境，随机性：一是智能体决策的动作的随机性，二是环境基于当前状态和智能体动作来采样下一刻状态的随机性。（动力学部分）

```math
b_{n+1} = P_n(a_n) \cdot b_n
```

奖励，奖励信号一般是诠释当前状态或动作的好坏的及时反馈信号（梯度）。整个交互过程的每一轮获得的奖励信号可以进行累加，形成智能体的整体回报（return）。我们关注回报的期望，并将其定义为价值（value），这就是强化学习中智能体学习的优化目标。

数据层面观察：
- 有监督学习的任务：数据分布是完全不变的   （可以看作是环境的状态概率分布）
- 强化学习：如果智能体不采取某个决策动作，那么该动作对应的数据就永远无法被观测到，智能体的策略不同，与环境交互所产生的数据分布就不同。为此用占用度量（occupancy measure）刻画，即状态动作对（state-action pair）的概率分布，而非状态概率分布。 （可以理解为状态动作对为泛函）

难点，智能体看到的数据分布是随着智能体的学习而不断发生改变的。

> 相当于不仅可以用概率描述状态，还可以描述函数

预测：

```math
\hat{f}^* = \arg \min_{\hat{f}} \; \mathbb{E}_{(x,y)\sim \mathcal{D}} \big[ \, \ell(y, \hat{f}(x)) \,\big]
```

- $(x,y)\sim \mathcal{D}$：表示训练数据 $(x,y)$ 来自某个分布 $\mathcal{D}$，其中 $x$ 是特征，$y$ 是标签。
- $\mathbb{E}_{(x,y)\sim \mathcal{D}}$：期望，表示在整个数据分布上的平均损失，也称为 **泛化误差**（generalization error）。

强化学习：

```math
\text{最优策略} = \arg\max_{\pi} \; \mathbb{E}_{(s,a)\sim\pi} \left[ R(s,a) \right]
```

> 这很像最优控制，求轨迹

| 概念                  | 最优控制 (PMP)                                                                        | 强化学习 (RL)                                                                    |
| --------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **状态**              | $`x_t`$                                                                               | $`s_t`$                                                                          |
| **控制 / 动作**       | $`u_t`$                                                                               | $`a_t`$                                                                          |
| **动力学**            | $`x_{t+1} = f(x_t,u_t)`$                                                              | $`s_{t+1} \sim P(\cdot \mid s_t,a_t)`$                                           |
| **性能指标**          | $`J(\pi) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r(x_t,u_t)\right]`$             | $`J(\pi) = \mathbb{E}\left[\sum_{t=0}^\infty \gamma^t r(s_t,a_t)\right]`$        |
| **哈密顿函数**        | $`\mathcal{H}(x_t, u_t, \lambda_{t+1}) = r(x_t,u_t) + \lambda_{t+1}^\top f(x_t,u_t)`$ | $`Q^\pi(s_t,a_t) = r(s_t,a_t) + \gamma \, \mathbb{E}_{s_{t+1}}[V^\pi(s_{t+1})]`$ |
| **最优性条件**        | $`u_t^* = \arg\max_{u_t} \mathcal{H}(x_t, u_t, \lambda_{t+1})`$                       | $`a_t^* = \arg\max_{a_t} Q^*(s_t,a_t)`$                                          |
| **协态变量 / 值函数** | $`\lambda_t`$ = 未来约束的影子价格                                                    | $`V^\pi(s_t)`$ = 未来奖励的期望值                                                |

> 在于强化学习中，环境的动力学 $P$ 通常是未知的，因此需要通过与环境的交互来估计值函数或策略。

- 基于价值 (Value-based RL)：先学会“评分表 (score table)”（执行 $a$ 后的长期回报期望），再从表里挑最优动作。

```math
a = \arg\max_{a'} Q^\pi(s,a')
```

- 基于策略 (Policy-based RL)：用万能函数逼近动作，看动作参数对回报的影响
- actor-critic：结合两者


深度强化学习：利用神经网络万能逼近价值函数和策略：[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

eg. DQN: 神经网络逼近 Q 函数（经验回放 + 双网络）

[Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)

[Q-learning](https://link.springer.com/article/10.1007/BF00992698)

### 02

强化学习关注智能体和环境交互过程中的学习，这是一种试错型学习（trial-and-error learning）范式

权衡 （因为环境动力学未知）：
- Exploitation 执行能够获得已知最优收益的决策
- Exploration 尝试更多可能的决策，不一定会是最优收益

多臂老虎机（multi-armed bandit，MAB）：

- 动作：K 个拉杆
- 奖励：r 服从某个分布
- 目标：T次内获得最大累计奖励

```math
\sum_{t=1}^T r_t \quad r_t \sim P(\cdot | a_t)
```

> 由于随机性存在，无法保证复刻的时候会得到同样的结果，所以考虑期望，这个是稳定的（大数定律）

$`Q`$ 当前动作的期望奖励

```math
Q = \mathbb{E}_{r \sim R(\cdot | a)}[r] \\
Q^* = \max_a Q(a) \\
```

懊悔（regret）：拉动当前拉杆的动作$`a`$与最优拉杆的期望奖励差 （上帝视角，用来评估算法的好坏，算法的性能指标）

```math
R(a) = Q^* - Q(a)
```

性能：

如果选择了次优动作 $a$，我们就损失了一部分奖励：

```math
\Delta_a = Q^\star - Q(a)
```

这就是 **次优间隔 (suboptimality gap)**。

Lai & Robbins 下界 （[Asymptotically efficient adaptive allocation rules](https://www.sciencedirect.com/science/article/pii/0196885885900028)）

他们证明了一个基本极限：

$$
\lim_{T \to \infty} \frac{\sigma_R(T)}{\log T} \;\; \geq \; \sum_{a:\Delta_a>0} \frac{\Delta_a}{D_{\mathrm{KL}}\big(\mathcal{R}(r|a) \,\|\, \mathcal{R}^\star(r|a)\big)}
$$

解释：

* $\mathcal{R}(r|a)$：动作 $a$ 的奖励分布。
* $\mathcal{R}^\star(r|a)$：最优动作的奖励分布。
* $D_{\mathrm{KL}}(\cdot \|\cdot)$：KL 散度，衡量两个分布的“可区分性”。
* 含义：**如果一个次优动作和最优动作的分布太相似，就需要更多次探索来分辨它们 → 遗憾下界更高。**

- $`\Delta_a`$ 决定了选错动作会损失多少奖励。
- **KL 散度** 决定了我们多快能分辨出动作好坏。
- Lai & Robbins 定理告诉我们：$`\mathcal{O}(\log T)`$ 是多臂赌博机的最优遗憾增长速度

策略：

- 探索（exploration）是指尝试拉动更多可能的拉杆，这根拉杆不一定会获得最大的奖励，但这种方案能够摸清楚所有拉杆的获奖情况。
- 利用（exploitation）是指拉动已知期望奖励最大的那根拉杆，由于已知的信息仅仅来自有限次的交互观测，所以当前的最优拉杆不一定是全局最优的。

$`\epsilon`$-贪心算法：（初始值全为1,是积极探索，全部为0是保守利用）

评估过去的经验：$`\hat Q`$

```math
a_t =
\begin{cases}
\arg\max\limits_a \hat{Q}(a), & \text{with probability } 1-\epsilon, \\[6pt]
U(0,|\mathcal{A}|), & \text{with probability } \epsilon.
\end{cases}
```

性能（累积懊悔）：

```math
\sigma =
\begin{cases}
\propto T \cdot (Q^\prime - Q^*), & 1-\epsilon, \\[6pt]
\;\;\geq\;\; \frac{\epsilon}{|\mathcal{A}|} \sum_{a \in \mathcal{A}} \Delta_a, & \epsilon
\end{cases}
```

> 第二部分就是计算概率，$`\epsilon`$ 选择探索，选到的概率是$`\frac{1}{|\mathcal{A}|}`$，损失是$`\Delta_a`$

一个改进是，$`\epsilon`$ 随时间衰减。因为探索和利用不是独立的，智能体对环境的了解会越来越多，探索的必要性会降低。

上置信界算法：评估探索和利用，探索本质是不确定性的价值

```math
Q = \hat{Q}_t(a) + \hat{U}_t(a)
```

利用Hoeffding 不等式: 

```math
\mathbb{P}\!\left[\,\mathbb{E}[x] > \bar{x}_t + u\,\right] \leq e^{-2tu^2}, 
\quad x \in [0,1]
```

> 这个描述的是一个下界成立的概率，反面就是上界成立的概率, [probability inequalities for sums of bounded random variables](https://www.jstor.org/stable/2282952)

```math
\mathbb{P}\!\left[\,\mathbb{E}[x] \leq \bar{x}_t + u\,\right] \geq 1 - e^{-2tu^2} = 1 - p \\
\mathbb{P}\!\left[Q \leq \hat{Q}_t(a) + \hat{U}_t(a)\,\right] \geq 1 - p \\
Q \leq \hat{Q}_t(a) + \hat{U}_t(a), \text{with probability} \; 1 - p \\
```

有 $`e^{-2n U^2} = p`$，$`p`$ 是超参，可以取 $`p = \frac{1}{t}`$，则

> 相当于选择上限最大的，潜力最大的

汤普森采样（Thompson sampling）：先假设拉动每根拉杆的奖励服从一个特定的概率分布，然后根据拉动每根拉杆的期望奖励来进行选择。

采用 beta 分布来估计利用和探索，其中不确定性就是宽度

- 不确定性大 → Beta 分布宽 → 偶尔采样出高值 → 促使探索。
- 不确定性小 → Beta 分布窄 → 高概率靠近真实均值 → 算法稳定利用。

> 也就是用 beta 函数来作为价值预测，综合来看，都是有对价值的评估，也是对动力学的估计。$`\epsilon`$-贪心算法估计的是常量，或者修改为随时间衰减的函数；上置信界算法估计利用不确定性估计上界，Thompson 利用的是采用估计

