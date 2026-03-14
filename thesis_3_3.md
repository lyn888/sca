# 3.3 脉冲神经网络通道冗余与能耗关联分析

在第 2 章中，本文已经从模型压缩与近似能耗分析的角度讨论了结构规模、脉冲活动和时间展开长度之间的内在联系。对于本章研究的通道剪枝问题而言，还需要进一步说明的是：在脉冲神经网络中，通道冗余并不只意味着结构不够紧凑，还会直接影响模型的能耗相关表现。只有将这种联系阐释清楚，后续围绕通道重要性评估所建立的剪枝准则才具备明确的理论依据。

与传统卷积神经网络类似，SNN 中卷积通道的基本任务是提取输入信号中的局部特征，并将其传递至后续层以完成更高层次的时空表征。然而，不同通道在网络中的作用并不完全等价。一部分通道能够稳定提取具有判别意义的动态特征，对后续层激活和最终分类结果具有持续影响；另一部分通道在训练结束后则表现出较弱的活动水平，或者虽然具备一定发放能力，但与同层其他通道的时空响应高度相似，其额外贡献相对有限[1-3][5-7]。从表征角度看，这类通道体现为冗余；从压缩角度看，则构成了优先考虑的剪枝对象。

对于卷积层而言，通道冗余首先表现为静态结构开销的冗余。设第 $l$ 层卷积核张量为 $W^{(l)} \in \mathbb{R}^{C_l^{\mathrm{out}} \times C_l^{\mathrm{in}} \times K_l \times K_l}$，则该层参数量可表示为

$$
P_l = C_l^{\mathrm{out}} C_l^{\mathrm{in}} K_l^2.
$$

若将第 $l$ 层输出通道数由 $C_l^{\mathrm{out}}$ 压缩为 $\tilde{C}_l^{\mathrm{out}}$，则压缩后该层参数量可写为

$$
\tilde{P}_l = \tilde{C}_l^{\mathrm{out}} C_l^{\mathrm{in}} K_l^2.
$$

由此可见，输出通道数减少将直接带来本层参数规模下降。更重要的是，卷积通道剪枝具有跨层传递效应：第 $l$ 层被删除的输出通道，同时也是第 $l+1$ 层的输入通道，因此其影响并不局限于当前层，而会进一步削减下一层的连接规模。若忽略偏置项和归一化层参数，则删除第 $l$ 层一个输出通道所引起的静态参数减少量可近似表示为

$$
\Delta P_{l,c}
\approx
C_l^{\mathrm{in}} K_l^2 + C_{l+1}^{\mathrm{out}} K_{l+1}^2.
$$

该式表明，一个冗余通道的存在实际上会同时占用当前层和下一层的参数资源。因此，从静态结构角度看，通道剪枝并不是孤立的局部参数删除，而是会对后续连接拓扑产生连锁影响的结构压缩过程[5-7]。

然而，对于脉冲神经网络而言，仅从参数量讨论通道冗余仍然是不充分的。SNN 的核心特征在于事件驱动计算，即只有当脉冲真正发生时，相关的突触累加与状态更新才会被触发[1][4]。这意味着，某个通道是否应当被保留，不仅取决于其是否占用了静态参数资源，还取决于它在时间窗口内是否真正产生了有价值的脉冲传播。设第 $l$ 层输入平均脉冲率为 $\bar{\rho}_{l,\mathrm{in}}$，则该层在时间窗口 $T$ 内的近似突触操作数可表示为

$$
\mathrm{SynOps}_l
\approx
T \cdot H_l \cdot W_l \cdot C_l^{\mathrm{out}} \cdot C_l^{\mathrm{in}} \cdot K_l^2 \cdot \bar{\rho}_{l,\mathrm{in}},
$$

其中，$H_l$ 和 $W_l$ 分别表示第 $l$ 层输出特征图的空间尺寸[1][4][8]。这一表达式说明，SNN 的动态运行开销同时受到时间展开长度、有效连接规模以及输入脉冲活动水平三类因素影响。对于给定的网络结构而言，若脉冲活动越稀疏，则实际发生的突触事件越少，近似运行开销也越低；反之，若冗余通道持续参与脉冲传播，即便其对最终判别贡献有限，仍会带来额外的事件驱动计算负担。

从通道级角度看，删除冗余通道对动态开销的影响同样具有双重性。设第 $l$ 层第 $c$ 个输出通道的平均脉冲率为 $\bar{\rho}_{l,c}$，则删除该通道至少会带来两部分近似收益。一部分来自当前层少计算一个输出通道，对应的近似突触操作减少量为

$$
\Delta \mathrm{SynOps}_{l,c}^{(1)}
\approx
T \cdot H_l \cdot W_l \cdot C_l^{\mathrm{in}} \cdot K_l^2 \cdot \bar{\rho}_{l,\mathrm{in}};
$$

另一部分来自下一层少接收一个输入通道，对应的近似突触操作减少量可写为

$$
\Delta \mathrm{SynOps}_{l,c}^{(2)}
\approx
T \cdot H_{l+1} \cdot W_{l+1} \cdot C_{l+1}^{\mathrm{out}} \cdot K_{l+1}^2 \cdot \bar{\rho}_{l,c}.
$$

其中，前一项反映的是当前层少生成一个输出特征图所带来的结构性动态收益，后一项反映的是下一层少接收一路脉冲输入所带来的事件性动态收益。二者共同表明，通道剪枝对于 SNN 的影响并不是单层局部的，而是同时作用于结构规模和脉冲传播路径的整体过程。也正因为如此，面向 SNN 的通道压缩不应仅以参数减少量作为判断依据，而应同时考虑通道在事件驱动传播中的实际活动情况。

进一步分析可以发现，通道冗余与能耗相关收益之间并不构成简单的一一对应关系。若仅依据低脉冲率判断通道冗余，虽然有助于减少事件传播，但可能误删那些发放不频繁却具有较强区分性的关键通道；若仅依据高活动度保留通道，则又可能保留大量发放频繁但表征高度相似的冗余通道。换言之，SNN 中真正值得保留的通道，既不能简单理解为活动最强的通道，也不能简单理解为脉冲最少的通道，而应当是那些在有效参与脉冲传播的同时，能够提供相对独特时空表征的通道。这一点与传统 ANN 的通道剪枝存在明显差异，也是本文后续设计通道重要性指标时必须重点考虑的问题。

从方法设计角度看，上述分析可以归纳为两点结论。第一，通道剪枝的压缩收益在 SNN 中具有双重来源，即静态参数规模的缩减和动态突触事件数量的下降；第二，通道冗余的判定不能依赖单一尺度，而应同时考虑通道是否有效参与脉冲传播，以及其所表达的信息是否具有足够差异性。前者对应脉冲发放有效性，后者对应通道表征差异性。基于这两方面认识，本文在后续章节中将不再沿用单纯依赖权值幅值或活动强度的评分方式，而是围绕脉冲输出、膜电位状态以及通道间动态差异构建更符合 SNN 机制的通道重要性评估指标。

综合而言，脉冲神经网络中的通道冗余不仅意味着结构资源浪费，更意味着事件驱动传播路径中的额外计算负担。由于卷积通道在参数连接和脉冲传播两个层面同时发挥作用，删除冗余通道能够同时带来参数压缩和近似 SynOps 下降的双重收益；但为了避免由过度压缩引发明显的精度损失，又必须在结构收益与表征保持之间建立更合理的权衡关系。这一分析构成了本文后续通道重要性评估指标设计的直接理论基础。

## 参考文献

[1] ROY K, JAISWAL A, PANDA P. Towards spike-based machine intelligence with neuromorphic computing[J]. Nature, 2019, 575(7784): 607-617.

[2] ZHENG H, WU Y, DENG L, et al. Going deeper with directly-trained larger spiking neural networks[C]// Proceedings of the AAAI Conference on Artificial Intelligence. 2021, 35(12): 11062-11070.

[3] FANG W, YU Z, CHEN Y, et al. Deep residual learning in spiking neural networks[C]// Advances in Neural Information Processing Systems. 2021, 34: 21056-21069.

[4] DAVIES M, SRINIVASA N, LIN T H, et al. Loihi: A neuromorphic manycore processor with on-chip learning[J]. IEEE Micro, 2018, 38(1): 82-99.

[5] HAN S, POOL J, TRAN J, et al. Learning both weights and connections for efficient neural networks[C]// Proceedings of the 28th International Conference on Neural Information Processing Systems. 2015: 1135-1143.

[6] HE Y, ZHANG X, SUN J. Channel pruning for accelerating very deep neural networks[C]// Proceedings of the IEEE International Conference on Computer Vision. 2017: 1389-1397.

[7] LIU Z, LI J, SHEN Z, et al. Learning efficient convolutional networks through network slimming[C]// Proceedings of the IEEE International Conference on Computer Vision. 2017: 2736-2744.

[8] HOROWITZ M. 1.1 Computing's energy problem (and what we can do about it)[C]// 2014 IEEE International Solid-State Circuits Conference Digest of Technical Papers. Piscataway: IEEE, 2014: 10-14.
