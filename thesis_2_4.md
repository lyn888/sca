# 2.4 神经网络剪枝技术基础

神经网络剪枝（Pruning）是模型压缩研究中最具代表性的技术路线之一，其核心思想是在尽量保持任务性能的前提下，识别并移除网络中的冗余连接、权值、卷积核、通道或层级结构，从而降低模型参数规模、计算复杂度以及实际部署成本[1-6]。从更深层次看，剪枝并不只是简单意义上的“删减参数”，而是围绕网络冗余识别、重要性评估、压缩策略设计以及性能恢复机制展开的一类综合优化问题。随着深度神经网络规模持续扩大，剪枝技术也逐步由早期面向浅层网络的权值删除方法，发展为覆盖非结构化稀疏、结构化压缩以及自动化压缩决策等多种形式的成熟研究方向[2-9]。

从数学上看，剪枝过程通常可以理解为在原始参数张量 $W$ 上引入二值掩码 $M$，并通过对 $M$ 的约束控制网络中保留的有效参数。设原始网络输出为 $f(x;W)$，则剪枝后的网络可表示为

$$
f\left(x;W\odot M\right),
$$

其中，$\odot$ 表示逐元素乘法，$M\in\{0,1\}^{|W|}$ 表示掩码矩阵。当 $M_i=0$ 时，对应参数被剪除；当 $M_i=1$ 时，对应参数被保留。于是，剪枝问题可抽象为如下约束优化问题：

$$
\min_{W,M}\ \mathcal{L}\left(W\odot M\right)
\quad
\text{s.t.}\quad
\|M\|_0 \leq K,
$$

其中，$\mathcal{L}(\cdot)$ 为任务损失函数，$\|M\|_0$ 表示保留参数个数，$K$ 为稀疏度预算。若进一步考虑资源约束，则也可写为

$$
\min_{W,M}\ \mathcal{L}\left(W\odot M\right)+\lambda \mathcal{R}(M),
$$

其中，$\mathcal{R}(M)$ 可表示参数量、FLOPs、延迟或能耗等资源开销项，$\lambda$ 为权衡系数。上述表达式表明，剪枝的核心目标是在性能保持与资源压缩之间寻求合理平衡，而不同剪枝方法之间的差异，主要体现在掩码生成方式、冗余结构粒度选择以及重要性评估标准的不同。

按照删除对象粒度和压缩后结构形态的不同，现有剪枝方法通常可分为非结构化剪枝与结构化剪枝两大类；若进一步从决策依据出发，也可以将其统一理解为基于重要性评估的剪枝过程[1-9]。本节将围绕这三个方面展开分析，为后续面向脉冲神经网络的剪枝方法设计提供必要的理论基础。

## 2.4.1 非结构化剪枝方法

非结构化剪枝（Unstructured Pruning）是指以单个权值、连接或参数元素为基本删除对象的剪枝方式。其特点在于压缩粒度细、参数稀疏率高，理论上可以在较小精度损失下移除大量冗余参数[1-4]。早期经典方法如 Optimal Brain Damage（OBD）和 Optimal Brain Surgeon（OBS）就属于这一范畴，它们通过分析删除单个参数对损失函数的影响来决定保留与删除对象[1][2]。在深度学习阶段，Han 等提出基于幅值的迭代剪枝方法，并通过“训练 - 剪枝 - 微调”流程显著提升了大规模深度网络的压缩能力[3]。此后，非结构化剪枝逐渐成为稀疏神经网络研究的重要基础。

### （1）基于二阶信息的经典剪枝

OBD 方法的基本思想是在已训练网络附近对损失函数进行二阶泰勒展开，并假设 Hessian 矩阵近似对角化。设参数向量为 $W=[w_1,w_2,\dots,w_n]^\top$，在局部极小点附近有

$$
\Delta \mathcal{L}
\approx
\frac{1}{2}\Delta W^\top H \Delta W,
$$

其中，$H$ 为损失函数关于参数的 Hessian 矩阵。若删除第 $i$ 个参数，即令 $\Delta w_i=-w_i$，并忽略非对角项，则该参数的敏感度可近似写为

$$
S_i^{\mathrm{OBD}}
=
\frac{1}{2}h_{ii}w_i^2,
$$

其中，$h_{ii}$ 为 Hessian 的第 $i$ 个对角元素。敏感度越小，说明删除该参数对损失函数的影响越小，更适合作为剪枝对象。

OBS 方法进一步考虑 Hessian 的逆矩阵信息，在保留损失变化最小的条件下求解最优参数补偿量，其参数删除代价可写为

$$
S_i^{\mathrm{OBS}}
=
\frac{w_i^2}{2(H^{-1})_{ii}}.
$$

与 OBD 相比，OBS 理论上能够给出更精确的剪枝代价估计，但由于需要显式处理 Hessian 或其逆矩阵，其计算代价较高，难以直接扩展到现代大规模深度网络[1][2]。尽管如此，这两类方法为后续基于重要性评估的剪枝研究奠定了理论基础。

### （2）基于权值幅值的剪枝

在深度学习阶段，最常见的非结构化剪枝方法是基于权值幅值（magnitude）的剪枝策略。其思路较为直接：若某个参数的绝对值较小，则往往认为其对网络输出贡献有限，可优先删除。相应的重要性评分可写为

$$
S_i^{\mathrm{mag}} = |w_i|
$$

或

$$
S_i^{\mathrm{mag2}} = w_i^2.
$$

在给定稀疏率 $p$ 时，可通过阈值 $\tau$ 对参数进行筛选，即

$$
M_i=
\begin{cases}
1, & |w_i|\geq \tau,\\
0, & |w_i|<\tau.
\end{cases}
$$

若保留参数数目为 $N_{\mathrm{keep}}$，总参数数目为 $N_{\mathrm{all}}$，则参数保留率和剪枝率分别可表示为

$$
\eta_{\mathrm{keep}}=\frac{N_{\mathrm{keep}}}{N_{\mathrm{all}}},
\qquad
\eta_{\mathrm{prune}}=1-\eta_{\mathrm{keep}}.
$$

Han 等提出的迭代式权值剪枝通常采用如下流程：首先训练一个稠密网络得到参数 $W$；随后根据幅值准则删除小权值参数，得到掩码 $M$；最后在固定掩码约束下对剩余参数进行微调[3][4]。若第 $k$ 轮迭代的参数为 $W^{(k)}$，掩码为 $M^{(k)}$，则微调过程可表示为

$$
W^{(k+1)} \leftarrow W^{(k)} - \alpha \nabla_{W}\mathcal{L}\left(W^{(k)}\odot M^{(k)}\right),
$$

其中，$\alpha$ 为学习率。该方法实现简洁、效果稳定，因此在实际应用中得到广泛采用。

### （3）训练前与训练中非结构化剪枝

除传统的训练后剪枝外，研究者还提出了训练前剪枝和训练中动态稀疏训练等方法。以 SNIP 为代表的训练前剪枝方法，尝试在网络初始化阶段直接根据连接对损失函数的敏感性进行筛选[7]。设网络参数为 $W$，掩码为 $c$，则 SNIP 的连接敏感度定义为

$$
S_i^{\mathrm{SNIP}}
=
\left|
\frac{\partial \mathcal{L}(c\odot W)}{\partial c_i}
\right|_{c=1}
=
\left|
\frac{\partial \mathcal{L}(W)}{\partial w_i}w_i
\right|.
$$

该评分本质上度量了剪除某一连接对损失函数的瞬时影响，可在训练开始前完成一次性筛选。另一类方法则在训练过程中动态更新稀疏拓扑，其形式可写为

$$
W^{t+1}= \left(W^{t}-\alpha \nabla \mathcal{L}(W^t\odot M^t)\right)\odot M^{t+1},
$$

其中，$M^t$ 随训练迭代动态变化。该类方法强调“边训练、边稀疏化、边重构拓扑”，能够在一定程度上减少对预训练稠密模型的依赖。

如图 2-4 所示，非结构化剪枝通常遵循“评分、删减、掩码约束与继续训练”的基本流程。

```mermaid
flowchart LR
    A[训练或初始化模型] --> B[计算连接重要性]
    B --> C[按阈值删除低重要性权值]
    C --> D[生成稀疏掩码]
    D --> E[固定或动态更新掩码]
    E --> F[微调/继续训练]
```

图 2-4 非结构化剪枝流程示意图

总体而言，非结构化剪枝能够获得较高参数稀疏率，并在理论上展现出较强压缩能力。然而，由于其产生的是不规则稀疏结构，通用硬件和标准深度学习库往往难以直接利用这种稀疏性获得稳定加速，因此其实际部署收益通常低于理论压缩收益[3][5][7]。也正因为如此，结构化剪枝逐渐成为更适合实际部署场景的重要研究方向。

## 2.4.2 结构化剪枝方法

结构化剪枝（Structured Pruning）是指以卷积核、通道、层或模块等更高层级结构单元作为删除对象的剪枝方式。与非结构化剪枝相比，结构化剪枝生成的压缩模型保持较规则的网络结构，因此更容易在通用 CPU、GPU 以及嵌入式推理平台上获得实际加速效果[5][6][8-10]。从部署角度看，结构化剪枝虽然在压缩粒度上不如非结构化剪枝细致，但在工程可实现性方面更具优势。

### （1）卷积层结构化剪枝的基本形式

以卷积层为例，设输入特征图为 $X\in\mathbb{R}^{C_{\mathrm{in}}\times H\times W}$，卷积核权重为

$$
W\in\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times K\times K},
$$

其中，$C_{\mathrm{in}}$ 为输入通道数，$C_{\mathrm{out}}$ 为输出通道数，$K$ 为卷积核尺寸。卷积输出可表示为

$$
Y_j = \sum_{i=1}^{C_{\mathrm{in}}} W_{j,i} * X_i + b_j,
\qquad j=1,2,\dots,C_{\mathrm{out}}.
$$

若删除某个输出通道 $j$，则不仅当前层对应卷积核 $W_{j,:,:,:}$ 会被移除，下一层与该通道相关的输入连接也需同步删除。因此，通道剪枝本质上会改变网络拓扑结构与后续层参数维度。若保留通道索引集合为 $\mathcal{S}$，则剪枝后的卷积层输出可写为

$$
\tilde{Y}_j = \sum_{i\in \mathcal{S}_{\mathrm{in}}}\tilde{W}_{j,i} * \tilde{X}_i + \tilde{b}_j,
\qquad j\in \mathcal{S}_{\mathrm{out}}.
$$

由于通道、卷积核和层级单元都具有明确的结构边界，结构化剪枝后得到的是更紧凑的网络，而不是仅在原网络中引入大量零值参数。

### （2）基于范数的结构化剪枝

最常见的结构化剪枝方法之一是基于卷积核或通道范数的重要性评估。设第 $j$ 个输出通道对应的卷积核权重为 $W_j$，则其重要性可以通过 $L_1$ 或 $L_2$ 范数近似表示为

$$
S_j^{L_1}=\|W_j\|_1=\sum_{i}|w_{j,i}|,
$$

$$
S_j^{L_2}=\|W_j\|_2=\sqrt{\sum_i w_{j,i}^2}.
$$

若通道重要性较小，则认为该通道对最终任务贡献较弱，可优先删除。Li 等提出的基于滤波器范数的结构化剪枝，正是该思路的代表性方法之一[5]。该类方法优点在于实现简单、无需额外复杂计算，但其局限在于仅依据权值本身统计量进行判断，难以充分反映特征图响应和层间协同关系。

### （3）基于重构误差的通道剪枝

He 等提出的 Channel Pruning 方法将通道剪枝问题转化为“在最小重构误差下选择最少通道”的优化问题[8]。设原始卷积层输出为 $Y$，通过选择通道子集和重构权重后得到近似输出 $\hat{Y}$，则其优化目标可写为

$$
\min_{\beta,W}\ \|Y-\hat{Y}\|_F^2
\quad
\text{s.t.}\quad
\|\beta\|_0 \leq K,
$$

其中，$\beta$ 为通道选择向量，$K$ 为允许保留的通道数。若进一步引入稀疏约束，则可转化为 LASSO 形式：

$$
\min_{\beta,W}\ \|Y-\sum_{i}\beta_i X_i * W_i\|_F^2 + \lambda \|\beta\|_1.
$$

该方法的核心在于：不是直接依据权值大小判断通道重要性，而是根据通道被移除后对特征重构误差的影响进行筛选，因此在很多场景下具有更好的精度保持能力。

### （4）基于 BN 缩放因子的结构化剪枝

Liu 等提出的 Network Slimming 将结构化剪枝与训练阶段的稀疏约束结合起来，通过 Batch Normalization（BN）层中的缩放因子 $\gamma$ 来反映通道重要性[9]。对于 BN 层，其输出可写为

$$
y = \gamma \hat{x} + \beta,
$$

其中，$\hat{x}$ 为归一化后特征，$\gamma$ 和 $\beta$ 分别为缩放与平移参数。若在训练过程中对 $\gamma$ 施加 $L_1$ 正则化，则整体目标函数可写为

$$
\mathcal{L}_{\mathrm{total}}
=
\mathcal{L}_{\mathrm{task}}+\lambda \sum_{j} |\gamma_j|.
$$

训练完成后，较小的 $\gamma_j$ 对应的通道通常被视为冗余通道，可按阈值进行删除，即

$$
M_j=
\begin{cases}
1, & |\gamma_j|\geq \tau,\\
0, & |\gamma_j|<\tau.
\end{cases}
$$

该方法将通道选择过程部分融入训练阶段，从而提高了结构化剪枝的自动化程度。由于 BN 参数在现代卷积网络中广泛存在，该方法具有较高实用价值。

### （5）结构化剪枝的压缩效果描述

结构化剪枝后的压缩率通常从参数规模和计算复杂度两个角度衡量。设某卷积层剪枝前参数量为

$$
P_{\mathrm{before}}=C_{\mathrm{out}}C_{\mathrm{in}}K^2,
$$

剪枝后参数量为

$$
P_{\mathrm{after}}=\tilde{C}_{\mathrm{out}}\tilde{C}_{\mathrm{in}}K^2,
$$

则参数压缩率可写为

$$
\eta_P = 1-\frac{P_{\mathrm{after}}}{P_{\mathrm{before}}}.
$$

若以卷积层 FLOPs 近似衡量计算量，则剪枝前后分别可表示为

$$
F_{\mathrm{before}}=2C_{\mathrm{out}}C_{\mathrm{in}}K^2H_{\mathrm{out}}W_{\mathrm{out}},
$$

$$
F_{\mathrm{after}}=2\tilde{C}_{\mathrm{out}}\tilde{C}_{\mathrm{in}}K^2H_{\mathrm{out}}W_{\mathrm{out}},
$$

相应计算压缩率为

$$
\eta_F = 1-\frac{F_{\mathrm{after}}}{F_{\mathrm{before}}}.
$$

如图 2-5 所示，结构化通道剪枝不仅删除当前层冗余通道，还需要同步调整后续层的输入维度。

```mermaid
flowchart LR
    A[原始卷积层<br/>多输入通道 多输出通道] --> B[评估通道重要性]
    B --> C[删除低重要性通道]
    C --> D[同步调整后续层输入维度]
    D --> E[得到紧凑卷积结构]
```

图 2-5 结构化通道剪枝示意图

总体来看，结构化剪枝更适合追求真实部署收益的应用场景。其主要优势在于剪枝后模型结构规则、易于硬件执行；但与此同时，由于其删除对象粒度较大，一旦误删关键通道，模型精度也更容易出现明显下降。因此，如何准确评估结构单元的重要性，便成为结构化剪枝研究中的核心问题。

## 2.4.3 基于重要性评估的剪枝方法

无论是非结构化剪枝还是结构化剪枝，其本质都离不开“重要性评估”这一环节。换言之，剪枝的关键并不在于是否删除，而在于如何判断“哪些参数或结构可以被删、哪些应当保留”。因此，基于重要性评估的剪枝方法可以被视为剪枝研究中的核心逻辑主线。现有工作通常从权值统计、梯度信息、二阶信息、特征图响应以及任务敏感性等多个角度构造重要性评分函数[1-10]。

### （1）一般形式

设待剪枝对象为 $\theta_i$，其重要性评分记为 $S_i$，则一个普遍形式的剪枝决策可写为

$$
S_i = \mathcal{I}(\theta_i,\mathcal{D},f),
$$

其中，$\mathcal{I}(\cdot)$ 表示重要性评估函数，$\mathcal{D}$ 表示数据分布或校准样本集，$f$ 表示当前网络模型。给定评分后，可按照排序方式保留前 $K$ 个对象，即

$$
\mathcal{S}_{\mathrm{keep}}
=
\mathrm{TopK}\left(\{S_i\}_{i=1}^{N}\right).
$$

若采用分层剪枝，则第 $l$ 层的保留集合为

$$
\mathcal{S}_{l}^{\mathrm{keep}}
=
\mathrm{TopK}\left(\{S_{l,i}\}_{i=1}^{N_l},K_l\right),
$$

其中，$K_l$ 表示该层保留对象数。由此可见，不同重要性评估方法之间的核心区别，本质上就在于评分函数 $\mathcal{I}(\cdot)$ 的设计。

### （2）基于梯度和泰勒展开的重要性评估

梯度信息能够反映参数变化对损失函数的敏感程度，因此常被用作重要性评估依据。若考虑删除参数 $\theta_i$ 带来的损失变化，可利用一阶泰勒展开近似：

$$
\Delta \mathcal{L}
\approx
\frac{\partial \mathcal{L}}{\partial \theta_i}\Delta \theta_i.
$$

若令 $\Delta\theta_i=-\theta_i$，即删除该参数，则其重要性可近似表示为

$$
S_i^{\mathrm{Taylor}}
=
\left|
\frac{\partial \mathcal{L}}{\partial \theta_i}\theta_i
\right|.
$$

Molchanov 等进一步将泰勒展开思想用于卷积通道和滤波器的重要性估计，并证明该类指标与真实删除代价之间具有较强相关性[10]。对于通道级对象，若通道输出特征图为 $z_i$，则通道重要性也可写为

$$
S_i^{\mathrm{feat}}
=
\left|
\frac{\partial \mathcal{L}}{\partial z_i} z_i
\right|.
$$

这类方法的优点在于兼顾了参数值和任务损失敏感性，相比单纯权值范数往往更具判别力。

### （3）基于激活统计和特征响应的重要性评估

对于卷积通道、注意力头或中间特征图等结构单元，仅使用权值本身往往难以准确反映其实际贡献，因此研究者常进一步引入激活统计量。设第 $l$ 层第 $i$ 个通道在样本集 $\mathcal{D}$ 上的输出特征图为 $A_{l,i}(x)$，则基于平均激活强度的重要性可写为

$$
S_{l,i}^{\mathrm{act}}
=
\frac{1}{|\mathcal{D}|}
\sum_{x\in\mathcal{D}}
\|A_{l,i}(x)\|_1.
$$

若进一步考虑方差或信息丰富程度，则可构造

$$
S_{l,i}^{\mathrm{var}}
=
\mathrm{Var}_{x\in\mathcal{D}}\left(A_{l,i}(x)\right),
$$

或

$$
S_{l,i}^{\mathrm{energy}}
=
\frac{1}{|\mathcal{D}|}
\sum_{x\in\mathcal{D}}
\|A_{l,i}(x)\|_2^2.
$$

此类指标的直观含义在于：若某个通道长期激活较弱，或者在不同样本间变化不明显，则其可能携带的有效判别信息有限，更适合作为剪枝对象。对于结构化剪枝而言，激活统计方法通常比单纯权值幅值更能反映通道在实际推理中的真实作用。

### （4）基于资源约束的重要性评估

在实际部署场景中，参数的重要性不应只由精度敏感性决定，还应与其资源消耗相关。设第 $i$ 个对象的精度重要性为 $S_i^{\mathrm{acc}}$，资源代价为 $C_i$，则可构造归一化的资源感知评分

$$
S_i^{\mathrm{res}}
=
\frac{S_i^{\mathrm{acc}}}{C_i+\epsilon},
$$

其中，$\epsilon$ 为防止分母为零的微小常数。若 $C_i$ 表示参数量、FLOPs、延迟或能耗，则该式体现出“在单位资源代价下贡献更大的结构应优先保留”的基本思想。进一步地，也可通过多目标形式统一精度与资源约束：

$$
S_i = \alpha S_i^{\mathrm{acc}} - \beta C_i,
$$

其中，$\alpha$ 和 $\beta$ 分别表示精度项和资源项的权重。对边缘部署和能耗优化问题而言，这类资源约束型重要性评估具有特别重要的意义。

### （5）重要性评估与剪枝流程

基于重要性评估的剪枝通常可以概括为“评分、排序、删除与重建/微调”的通用流程。其一般形式如图 2-6 所示。

```mermaid
flowchart LR
    A[待剪枝模型] --> B[计算重要性评分]
    B --> C[按评分排序]
    C --> D[确定剪枝阈值或剪枝率]
    D --> E[删除低重要性对象]
    E --> F[微调或重建模型]
```

图 2-6 基于重要性评估的剪枝流程示意图

若以分层剪枝为例，设第 $l$ 层剪枝率为 $p_l$，该层对象总数为 $N_l$，则保留数目可写为

$$
K_l=(1-p_l)N_l.
$$

随后在该层内选取评分最高的 $K_l$ 个对象保留。若采用全局剪枝，则直接在全网络范围内统一排序并选取前 $K$ 个对象。分层剪枝有助于维持各层结构平衡，而全局剪枝则更容易获得整体最优的压缩分配，但也更可能导致个别层过度削减。因此，具体采用何种策略，通常与网络结构特点和压缩目标密切相关。

为了更直观地比较不同剪枝类型的特点，表 2-2 给出了三类典型方法的对比。

| 方法类别 | 删除粒度 | 典型依据 | 优点 | 局限性 |
| --- | --- | --- | --- | --- |
| 非结构化剪枝 | 单个权值/连接 | 幅值、梯度、二阶信息 | 稀疏率高，理论压缩能力强 | 稀疏结构不规则，实际加速有限 |
| 结构化剪枝 | 卷积核、通道、层 | 范数、重构误差、BN 因子 | 结构规则，便于部署和加速 | 粒度较粗，误删代价较大 |
| 基于重要性评估剪枝 | 依对象而定 | 梯度、激活、资源感知评分 | 决策更有针对性，灵活性强 | 评分设计依赖任务和模型特性 |

综合而言，剪枝技术已经从早期单纯依赖权值删除的参数压缩手段，逐步发展为融合结构设计、重要性评估和资源约束优化的系统性方法。非结构化剪枝强调高稀疏率与理论压缩能力，结构化剪枝强调规则结构与实际部署收益，而基于重要性评估的剪枝则为不同粒度的压缩决策提供了统一的方法论基础。对于后续面向脉冲神经网络的剪枝研究而言，如何结合脉冲动态特征重新定义“结构重要性”，并将精度保持与能耗优化协同纳入剪枝决策，是在上述理论基础上进一步深化的重要方向。

## 参考文献

[1] LECUN Y, DENKER J S, SOLLA S A. Optimal brain damage[C]// Advances in Neural Information Processing Systems. 1990, 2: 598-605.

[2] HASSIBI B, STORK D G. Second order derivatives for network pruning: optimal brain surgeon[C]// Advances in Neural Information Processing Systems. 1993, 5: 164-171.

[3] HAN S, POOL J, TRAN J, et al. Learning both weights and connections for efficient neural networks[C]// Advances in Neural Information Processing Systems. 2015, 28: 1135-1143.

[4] HAN S, MAO H, DALLY W J. Deep compression: compressing deep neural networks with pruning, trained quantization and Huffman coding[EB/OL]. arXiv:1510.00149, 2015.

[5] LI H, KADAV A, DURDANOVIC I, et al. Pruning filters for efficient convnets[EB/OL]. arXiv:1608.08710, 2016.

[6] FRANKLE J, CARBIN M. The lottery ticket hypothesis: finding sparse, trainable neural networks[C]// International Conference on Learning Representations. 2019.

[7] LEE N, AHN T, YOO J, et al. SNIP: single-shot network pruning based on connection sensitivity[C]// International Conference on Learning Representations. 2019.

[8] HE Y, ZHANG X, SUN J. Channel pruning for accelerating very deep neural networks[C]// Proceedings of the IEEE International Conference on Computer Vision. 2017: 1389-1397.

[9] LIU Z, LI J, SHEN Z, et al. Learning efficient convolutional networks through network slimming[C]// Proceedings of the IEEE International Conference on Computer Vision. 2017: 2736-2744.

[10] MOLCHANOV P, MALLYA A, TYREE S, et al. Importance estimation for neural network pruning[C]// Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2019: 11264-11272.
