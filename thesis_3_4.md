# 3.4 通道重要性评估指标设计

基于上一节关于通道冗余与能耗关联的分析，可以看出，面向脉冲神经网络的通道剪枝并不能简单沿用传统卷积网络中基于权值幅值或静态激活的评价方式。对于 SNN 而言，通道是否值得保留，至少涉及两个彼此关联但又不能相互替代的问题：其一，该通道是否在完整时间窗口内真实参与了有效的脉冲传播；其二，该通道所表达的信息相对于同层其他通道是否具有足够差异性。前者决定了通道是否在事件驱动计算中发挥实际作用，后者决定了该通道的保留是否具有必要性。因此，本文将通道重要性评估问题理解为一个同时兼顾“发放有效性”和“表征差异性”的综合判定过程，而不是单一统计量的排序问题[1-4]。

围绕上述认识，本文在指标设计中遵循三个基本原则。第一，指标应能够反映 SNN 的动态特征，即不能只观察静态参数，而应从脉冲输出和神经元状态演化中提取信息。第二，指标应能够区分“活跃但冗余”的通道与“低频但关键”的通道，避免仅按活动强弱作出粗略判断。第三，指标应当具有可实现性，即在训练过程中能够稳定计算，并且不会因额外统计代价过高而削弱方法的实际可用性。基于这三点考虑，本文将通道重要性指标设计为由通道动态表征、差异性度量、脉冲门控和批间平滑四个环节共同构成的统一体系。

首先，本文采用脉冲序列与膜电位响应的联合表征来描述通道状态。设第 $l$ 层在时间窗口 $T$ 内的脉冲输出记为 $x^{(l)} \in \mathbb{R}^{T \times N \times C_l \times H_l \times W_l}$，对应膜电位状态记为 $v^{(l)} \in \mathbb{R}^{T \times N \times C_l \times H_l \times W_l}$。在此基础上，定义通道动态表征张量为

$$
z^{(l)} = x^{(l)} + \left|v^{(l)}\right|.
$$

这样处理的出发点在于：若仅使用脉冲序列，则容易忽略那些尚未发放但已经接近阈值、具有潜在传播作用的神经元状态；若仅使用膜电位，则又难以直接反映事件驱动特征。将二者联合起来，可以同时保留通道的显式发放行为和隐式状态响应，从而获得更完整的动态表征。对第 $c$ 个通道而言，其展开后的表征向量可写为

$$
F_c^{(l)} = \mathrm{vec}\!\left(z_{:,:,c,:,:}^{(l)}\right).
$$

该向量综合了时间维、样本维与空间维上的响应信息，是后续差异性计算的基础。

在获得通道表征之后，本文进一步使用通道间余弦相似度来刻画表征差异性。对任意两个通道 $c$ 和 $j$，先将其表征向量作 $L_2$ 归一化：

$$
\hat{F}_c^{(l)} = \frac{F_c^{(l)}}{\left\|F_c^{(l)}\right\|_2 + \varepsilon},
$$

然后定义它们之间的相似度为

$$
s_{c,j}^{(l)} = \left(\hat{F}_c^{(l)}\right)^{\top}\hat{F}_j^{(l)}.
$$

若某个通道与同层大多数通道都高度相似，则说明其所提供的额外信息有限；反之，若该通道与其他通道之间的相似度整体较低，则意味着其在表征空间中更具独特性。基于这一考虑，本文将第 $l$ 层第 $c$ 个通道的差异性指标定义为

$$
u_c^{(l)}
=
\mathrm{clip}\!\left(
1 - \frac{1}{C_l - 1}\sum_{j \ne c}s_{c,j}^{(l)},
\ 0,\ 2
\right),
$$

其中，$\mathrm{clip}(\cdot)$ 用于对数值范围进行约束，以避免极端值对后续排序造成过度影响。该定义的含义较为直接：当某通道与其他通道的平均相似度较高时，$u_c^{(l)}$ 较小，表示其冗余度较高；当其平均相似度较低时，$u_c^{(l)}$ 较大，表示其更可能携带不可替代的特征信息。

然而，仅以通道差异性作为剪枝依据仍然不足。原因在于，某些通道虽然与其他通道差异较大，但如果在完整时间窗口内几乎不发生有效发放，则其对后续层的实际驱动作用仍可能有限。为避免保留这类“具有差异性但缺乏传播作用”的通道，本文进一步引入基于脉冲发放率的门控机制。设第 $l$ 层第 $c$ 个通道的平均脉冲率为

$$
\rho_c^{(l)}
=
\frac{1}{T N H_l W_l}
\sum_{t=1}^{T}
\sum_{n=1}^{N}
\sum_{h=1}^{H_l}
\sum_{w=1}^{W_l}
x_{t,n,c,h,w}^{(l)},
$$

并记该层平均脉冲率为

$$
\bar{\rho}^{(l)}
=
\frac{1}{C_l}\sum_{c=1}^{C_l}\rho_c^{(l)}.
$$

在此基础上，本文构造脉冲门控因子

$$
g_c^{(l)}
=
\mathrm{clip}\!\left(
\frac{\rho_c^{(l)}}{\bar{\rho}^{(l)} + \varepsilon},
\ 0,\ 2
\right)
\cdot
\mathbb{I}\!\left(\rho_c^{(l)} > \lambda \bar{\rho}^{(l)}\right),
$$

其中，$\mathbb{I}(\cdot)$ 为示性函数，$\lambda$ 为门控阈值系数。在本文实现中，取 $\lambda = 0.1$。这一设计体现了两层考虑：其一，通过相对于层均值的归一化处理，减弱不同层之间脉冲率尺度差异对评分结果的影响；其二，通过阈值门控抑制长期处于极低发放状态的通道，避免其仅凭差异性优势而被错误保留。

在通道差异性和脉冲门控两部分基础上，本文将第 $l$ 层第 $c$ 个通道的综合重要性定义为

$$
I_c^{(l)} = u_c^{(l)} \cdot g_c^{(l)}.
$$

需要强调的是，尽管通道动态表征张量 $z^{(l)}$ 的均值能够在一定程度上反映通道总体活动水平，但在本文的最终实现中，并未直接将该均值作为最终评分项参与排序，而是通过“差异性 × 门控”这一形式完成重要性建模。这样处理的原因在于，若直接以总体活动量参与评分，容易使高活动但高冗余的通道获得过高分值；而采用差异性和脉冲门控的乘积形式，则能够更清晰地保留“有效且独特”的通道，同时压低“活跃但重复”或“独特但几乎不工作”的通道得分。

为了提高指标在训练过程中的稳定性，本文在实现层面对通道重要性和脉冲率都采用了指数滑动平均策略。设第 $k$ 次统计得到的通道重要性和脉冲率分别为 $I_c^{(l,k)}$ 与 $\rho_c^{(l,k)}$，则其平滑更新形式可写为

$$
\tilde{I}_c^{(l,k)}
=
m_I \tilde{I}_c^{(l,k-1)}
+
(1-m_I) I_c^{(l,k)},
$$

$$
\tilde{\rho}_c^{(l,k)}
=
m_{\rho} \tilde{\rho}_c^{(l,k-1)}
+
(1-m_{\rho}) \rho_c^{(l,k)},
$$

其中，$m_I$ 和 $m_{\rho}$ 分别表示重要性与脉冲率的平滑系数。在本文实现中，取 $m_I = 0.95$、$m_{\rho} = 0.95$。此外，考虑到通道间余弦相似度矩阵的计算复杂度与通道数平方成正比，为兼顾统计精度与计算开销，本文并不在每个训练步都重新计算差异性矩阵，而是采用间隔更新与缓存复用的方式：每隔 $K_u$ 个更新步重新计算一次差异性指标，其余时刻复用最近一次结果。在本文实现中，取 $K_u = 8$。这一设计能够在保证指标稳定性的同时，避免通道差异性统计成为训练阶段的主要额外负担。

为更直观地说明指标构造过程，图 3-3 给出了本文通道重要性评估指标的整体流程。可以看出，该指标并非对单一统计量的直接排序，而是从动态表征构造出发，经由差异性计算和脉冲门控两条路径共同形成最终重要性分数，并通过平滑更新用于后续通道排序。

```mermaid
flowchart TD
    A[脉冲序列 x] --> C[构造动态表征 z = x + |v|]
    B[膜电位状态 v] --> C
    C --> D[展开通道表征向量 F_c]
    D --> E[计算余弦相似度矩阵]
    E --> F[得到差异性指标 u_c]
    A --> G[统计通道平均脉冲率 rho_c]
    G --> H[构造脉冲门控因子 g_c]
    F --> I[综合重要性 I_c = u_c * g_c]
    H --> I
    I --> J[指数滑动平均]
    J --> K[用于后续通道排序与剪枝决策]
```

图 3-3 通道重要性评估指标构造流程图

综合而言，本文所设计的通道重要性指标本质上是在脉冲神经网络动态机制约束下，对“是否保留某个通道”这一问题所作的系统化回答。通过引入脉冲输出与膜电位的联合表征、通道差异性度量、基于平均脉冲率的门控机制以及批间平滑更新策略，本文将通道重要性建模为一个既体现时空动态特征、又兼顾实现稳定性的综合评价过程。这一指标设计不仅与前文关于通道冗余和能耗关联的分析保持一致，也为下一节基于重要性排序的结构化剪枝策略奠定了直接基础。

## 参考文献

[1] ROY K, JAISWAL A, PANDA P. Towards spike-based machine intelligence with neuromorphic computing[J]. Nature, 2019, 575(7784): 607-617.

[2] ZHENG H, WU Y, DENG L, et al. Going deeper with directly-trained larger spiking neural networks[C]// Proceedings of the AAAI Conference on Artificial Intelligence. 2021, 35(12): 11062-11070.

[3] FANG W, YU Z, CHEN Y, et al. Deep residual learning in spiking neural networks[C]// Advances in Neural Information Processing Systems. 2021, 34: 21056-21069.

[4] DAVIES M, SRINIVASA N, LIN T H, et al. Loihi: A neuromorphic manycore processor with on-chip learning[J]. IEEE Micro, 2018, 38(1): 82-99.

[5] HAN S, POOL J, TRAN J, et al. Learning both weights and connections for efficient neural networks[C]// Proceedings of the 28th International Conference on Neural Information Processing Systems. 2015: 1135-1143.

[6] HE Y, ZHANG X, SUN J. Channel pruning for accelerating very deep neural networks[C]// Proceedings of the IEEE International Conference on Computer Vision. 2017: 1389-1397.

[7] LIU Z, LI J, SHEN Z, et al. Learning efficient convolutional networks through network slimming[C]// Proceedings of the IEEE International Conference on Computer Vision. 2017: 2736-2744.

[8] HOROWITZ M. 1.1 Computing's energy problem (and what we can do about it)[C]// 2014 IEEE International Solid-State Circuits Conference Digest of Technical Papers. Piscataway: IEEE, 2014: 10-14.
