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
