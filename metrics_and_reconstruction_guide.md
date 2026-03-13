# 指标计算与剪后重建说明

本文档用于回答两个问题：

1. 连接率、参数量和 SynOps 在论文中通常如何计算，以及你当前代码中哪些地方是合理的、哪些地方还需要修正。
2. 剪后重建（post-pruning reconstruction）为什么能提升精度、文献里通常怎么做，以及你当前代码应该如何落地。

---

## 1. 三个指标到底应该怎么算

### 1.1 连接率（connection rate）

如果你的模型没有真正“删除张量维度”，而只是通过 mask 把一部分卷积通道或权重置零，那么最自然的定义是：

`连接保留率 = 非零连接数 / 总连接数 × 100%`

其中“连接”通常只统计真正的突触权重，也就是：

- `Conv2d.weight`
- `Linear.weight`

一般不把 BN 的 `gamma/beta` 计入“连接”，因为它们不是神经元之间的突触连接，而是归一化层参数。

这一定义本质上就是“稀疏率的补数”，在 Han 等的剪枝工作中也是这一类统计口径。对你现在的代码来说，[train.py](/d:/毕业论文/调研/528_Towards_efficient_deep_spi_Supplementary%20Material/sca_code/train.py) 中的 `compute_connection_percent()` 思路是基本合理的，因为它统计的是当前 mask 生效后的非零卷积/全连接权重比例。

但是需要注意：

- 这个指标描述的是“masked model 的有效连接保留率”。
- 它不等于真正部署后 compact model 的参数压缩率。
- 如果论文里同时报告“连接率”和“参数量比例”，两者应该分开解释，不能混用。

建议你在论文里把它表述为：

- `连接保留率（Remaining Connections, %）`
- 或者 `非零连接比例（Non-zero Weight Ratio, %）`

不要直接写成“参数压缩率”。

---

### 1.2 参数量（parameter count）

参数量更适合按“压缩后紧凑模型（compact model）”来统计，而不是按原模型里剩下多少非零值来统计。

如果你做的是结构化通道剪枝，那么论文里更常见的口径是：

`参数保留率 = 压缩后紧凑模型参数量 / 原始模型参数量 × 100%`

对于 `snnvgg16_bn`，更合理的参数量应该包含：

1. 所有卷积层权重
2. 所有 `BatchNorm2d` 的可学习参数
3. 所有分类器线性层权重
4. 所有 `BatchNorm1d` 的可学习参数
5. 最终 `fc` 的权重和偏置

按你当前 [snnvgg.py](/d:/毕业论文/调研/528_Towards_efficient_deep_spi_Supplementary%20Material/sca_code/snnvgg.py) 的定义，`snnvgg16_bn` 的分类头实际上是：

- `classifier1: Linear(C_last, 512, bias=False) + BatchNorm1d(512)`
- `classifier2: Linear(512, 512, bias=False) + BatchNorm1d(512)`
- `fc: Linear(512, num_classes, bias=True)`

因此，更合理的参数量公式应为：

`P_conv_bn = Σ_l [C_out^(l) * C_in^(l) * k_l^2 + 2 * C_out^(l)]`

`P_cls1 = C_last * 512 + 2 * 512`

`P_cls2 = 512 * 512 + 2 * 512`

`P_fc = 512 * N_cls + N_cls`

`P_total = P_conv_bn + P_cls1 + P_cls2 + P_fc`

这里 `2 * C_out` 和 `2 * 512` 对应 BN 的 `weight` 和 `bias`。

### 你当前代码的问题

[train.py](/d:/毕业论文/调研/528_Towards_efficient_deep_spi_Supplementary%20Material/sca_code/train.py) 里的 `compute_compact_vgg_params()` 目前有两个主要问题：

1. 它漏掉了 `classifier2: 512 -> 512` 这一层。
2. 它没有完整计入两个 `BatchNorm1d` 的参数。

所以你当前代码算出来的 compact 参数量会偏小，进而导致“参数保留率”过于乐观。

### 更稳妥的做法

最稳妥的方式不是手写公式，而是：

1. 根据 `cfg.txt` 真正构造一个 compact VGG。
2. 直接对 compact model 执行 `sum(p.numel() for p in model.parameters())`。

这样不会漏掉分类头、BN1d、bias 等项，也最适合论文复现实验。

如果你暂时不想重构代码，那么至少要先把手工公式改正确。

---

### 1.3 SynOps（synaptic operations）

SynOps 是 SNN 里比 FLOPs 更合适的计算量指标。它的核心思想是：

`只有发生脉冲的输入，才会触发后续突触累加。`

因此，SynOps 描述的不是“密集卷积理论乘加数”，而是“在脉冲驱动下真正发生的突触累加次数”。

对卷积层 `l`，更常见的近似写法是：

`SynOps_l ≈ DenseOps_l × r_in^(l)`

其中：

- `DenseOps_l = T × H_out × W_out × C_out × C_in × k × k`
- `r_in^(l)` 是进入该层卷积输入张量的平均脉冲率

对全连接层：

`SynOps_fc ≈ T × N_in × N_out × r_in`

这里的关键是：

`应该使用该层输入的 spike rate，而不是该层输出的 spike rate。`

### 你当前代码的问题

[train.py](/d:/毕业论文/调研/528_Towards_efficient_deep_spi_Supplementary%20Material/sca_code/train.py) 里的 `compute_vgg_synops()` 当前写法是：

- 先取每个 `PruningLayer` 的平均 spike rate
- 然后在第 `i` 层卷积上使用第 `i` 个 spike rate

这个近似有三个问题：

1. `PruningLayer` 的 spike rate 是该卷积块输出脉冲，而不是下一层卷积真正的输入脉冲。
2. 对存在池化的网络，池化会改变脉冲稀疏性，因此“上一层 prune 输出 rate”也未必等于“下一层卷积输入 rate”。
3. 当前实现没有把分类器线性层的 SynOps 算进去。

### 更合理的口径

对于你现在的 `snnvgg16_bn`，推荐这样处理：

#### 方案 A：论文里可接受的近似法

- 第 1 个卷积层：
  - 如果输入是静态图像并在 `T` 个时间步上重复输入，那么令 `r_in = 1`
  - 如果输入是 DVS 事件帧，则直接统计输入事件张量的非零比例
- 第 `l (l >= 2)` 个卷积层：
  - 用“该层卷积实际输入张量”的平均 spike rate
- 全连接层：
  - 用 GAP 或上一层输出后的平均 spike rate

也就是说，最好的办法是：

`直接对每个 Conv2d / Linear 的输入注册 hook，统计真实输入稀疏率。`

#### 方案 B：如果你暂时不改很多代码

可以采用“shift 一位”的近似：

- `conv1`: `r_in = 1`（静态 CIFAR/Tiny-ImageNet）
- `conv_l (l >= 2)`: 使用前一层脉冲率 `spike_rate[l-1]`

这比你当前“本层输出率乘本层 DenseOps”的写法更合理。

但仍需说明：这是近似计算，不是严格逐事件统计。

### 论文里建议怎么写

如果你最终采用近似法，论文中建议写成：

“本文采用基于平均脉冲率的近似 SynOps 统计方法，将每层密集计算量乘以对应层输入脉冲率，以估计事件驱动条件下的实际突触累加开销。”

这样表述是稳妥的，不会把近似值说成严格硬件实测能耗。

---

## 2. 你现在最适合在论文里报告什么

结合你目前代码状态，建议你最终在论文中同时报告下面 4 个指标：

1. `Top-1 Accuracy`
2. `连接保留率（Remaining Connections, %）`
3. `参数保留率（Compact Parameters, %）`
4. `近似 SynOps 保留率（Approximate SynOps, %）`

这样做的好处是：

- 连接率反映 masked model 的实际稀疏程度
- 参数保留率反映 compact model 的压缩效果
- SynOps 反映事件驱动下的近似计算开销

这三个量分别对应不同层面的“压缩收益”，逻辑是完整的。

---

## 3. 剪后重建到底是什么，为什么能提高精度

所谓“剪后重建”，本质上不是简单再训练一次，而是：

`在结构已经被剪掉之后，用额外的约束帮助保留下来的通道重新组织特征表达。`

它之所以能提升精度，是因为剪枝会带来两个直接后果：

1. 原来由被剪通道承担的一部分特征表达能力突然消失。
2. 后续层接收到的特征分布发生偏移，导致整网协同关系被打破。

如果只做普通微调，模型确实有机会慢慢适应，但在高剪枝率下常常不够。于是论文里常见的做法是：

- 剪前先做信息迁移或信息聚合
- 剪后做特征重建
- 再进行固定 mask 的微调

这类思路在 ANN 剪枝中非常常见，比如：

- 用特征图重建指导通道剪枝
- 用教师网络和学生网络做中间层蒸馏
- 用渐进式、分阶段的微调来稳定恢复过程

你的代码其实已经朝这个方向走了，只是还没有完全跑通。

---

## 4. 你当前代码里“剪后重建”现状

你当前代码里已经有三个相关模块：

1. `ptp_information_aggregation()`
2. `ptp_reconstruction()`
3. `adaptive_final_reconstruction()`

其中：

- `ptp_information_aggregation()` 更像“剪前信息聚合（IA）”
- `ptp_reconstruction()` 更像“剪后特征重建（REC）”
- `adaptive_final_reconstruction()` 更像“训练结束后的最后一次自适应恢复”

### 但目前有两个关键问题

#### 问题 1：训练循环里 IA 和 REC 逻辑被注释掉了

在 [train.py](/d:/毕业论文/调研/528_Towards_efficient_deep_spi_Supplementary%20Material/sca_code/train.py) 里，真正执行剪枝的位置附近，`ptp_information_aggregation()` 和 `ptp_reconstruction()` 目前都还是注释状态。

这意味着：

- 你虽然写了这两个函数
- 但训练时其实没有真的调用它们

所以如果你现在感觉“剪后重建没有明显提高准确率”，很可能根本原因是：它现在并没有真正介入训练流程。

#### 问题 2：`adaptive_final_reconstruction()` 里的 teacher 选取不对

你当前在训练结束后做：

`teacher = copy.deepcopy(model)`

也就是：

- teacher 是当前已经剪完、已经训练完的 student 自己的拷贝

这样会导致一个问题：

- teacher 和 student 初始几乎完全一样
- 特征重建损失一开始就几乎为 0

这时所谓“重建”实际上退化成了：

- 固定 mask 条件下再做一点普通微调

它不是无效，但不会有强的“教师监督信号”，因此提升通常有限。

如果你想让重建真正有效，teacher 应该来自：

1. 当前 pruning step 之前的未剪模型
2. 或者训练好的 dense baseline

而不是当前已经剪完的 student 自己。

---

## 5. 推荐你采用的剪后重建方案

下面给你一套最适合你当前代码基础的落地流程。它既参考了论文里的常见做法，也尽量贴合你已经实现的模块。

### 阶段 0：先训练一个 dense teacher

先训练一个不剪枝的 baseline 模型，保存最优 checkpoint。

用途：

- 作为后续所有剪前信息聚合和剪后重建的 teacher
- 也可以作为论文里“未压缩基线模型”的对照

这一步非常关键。没有一个稳定 teacher，重建通常效果有限。

---

### 阶段 1：正常训练到 warmup 结束

也就是你当前的：

- `prune_warmup`
- 正常 CE 训练

目的：

- 先让 SNN 学到比较稳定的时空特征
- 避免过早剪枝导致结构和动态都不稳定

建议：

- CIFAR100 / Tiny-ImageNet 这种任务，`warmup = 20 ~ 40 epochs`
- DVS 数据可以根据收敛情况适当缩短，但不建议直接从 epoch 0 开始大规模剪

---

### 阶段 2：更新 mask，确定将被剪掉的通道

这一步你已经有：

- `mymanager.update_masks(model, alpha, beta)`

它完成的是：

1. 计算各层重要性
2. 得到本轮 mask
3. 同时根据梯度和 spike rate 计算再生得分

这一步结束后，你就已经知道：

- 哪些通道准备被剪掉
- 哪些已剪通道可能值得再生

---

### 阶段 3：剪前信息聚合（IA）

这一阶段应当在真正执行 `do_masks()` 之前做。

目标是：

`在通道还没被真正置零之前，把有用信息尽量转移到保留下来的通道中。`

你代码里的 `ptp_information_aggregation()` 基本就是这个思路。

推荐损失：

`L_IA = L_feat + λ_k * L_pruned`

其中：

- `L_feat = Σ_l ||F_s^l - F_t^l||_2^2`
- `L_pruned = Σ_{c ∈ to_be_pruned} ||W_c||_2^2`
- `λ_k = λ_0 + k * Δλ`

含义：

- 用教师中间特征指导学生保持主要语义表示
- 对即将被剪掉通道逐步施加更强约束，让其权重自然衰减

实践建议：

- hook 最后 2 到 4 个卷积层
- `ptp_calib_batches = 10 ~ 20`
- `ptp_ia_iters = 10 ~ 20`
- `ptp_lr = 1e-4`
- `ptp_reg = 1e-5 ~ 1e-4`
- `ptp_inc = 1e-5 ~ 1e-4`

如果剪枝率较高，可以优先对后面几层做 IA，因为深层特征对分类更敏感。

---

### 阶段 4：真正执行结构化剪枝

执行：

`mymanager.do_masks(model)`

这一步之后：

- 被剪通道对应的卷积权重会被置零
- BN 对应通道也会被置零

从这一步开始，模型结构虽然张量维度还没变，但功能上已经进入剪枝状态。

---

### 阶段 5：剪后特征重建（REC）

这一步才是最典型意义上的“剪后重建”。

目标是：

`在通道已经被剪掉以后，用教师特征引导剩余通道重新组织表达。`

推荐损失：

`L_REC = Σ_l w_l ||F_s^l - F_t^l||_2^2 + α * L_CE`

其中：

- `w_l` 是各层重建权重
- `L_CE` 是真实标签监督

为什么要加少量 CE：

- 只做特征对齐，模型可能过度贴 teacher 中间表示
- 加一点监督损失可以避免语义偏移

你代码里的 `ptp_reconstruction()` 已经非常接近这个结构。

实践建议：

- `ptp_rec_iters = 20 ~ 30`
- `ptp_lr = 1e-4`
- `α = 0.05 ~ 0.1`
- 每次 `optimizer.step()` 后都重新执行一次 `do_masks()`

最后这一点很关键，否则已剪权重会“偷偷长回来”。

---

### 阶段 6：固定 mask 微调（fixed-mask fine-tuning）

重建结束后，不要立刻结束训练，而是继续做若干 epoch 的固定 mask 微调。

这一步的目标是：

- 让剩余通道在新的结构约束下重新适应任务
- 进一步恢复准确率

推荐做法：

1. 学习率降为主训练阶段的 `0.1x` 或 `0.01x`
2. mask 固定，不再更新
3. 每一步优化后重新施加 `do_masks()`
4. 持续 5 到 20 个 epoch

这是很多剪枝论文里最有效也最稳定的一步。

---

### 阶段 7：自适应最终恢复（可选增强）

你现在的 `adaptive_final_reconstruction()` 思路本身是对的：

- 根据层扰动度
- 平均 spike rate
- 深度先验

给不同层分配不同重建权重，再挑最关键的后几层重点恢复。

但要让它真正有效，需要做两点修正：

1. teacher 不能是当前 student 自己的拷贝，而应该是 dense baseline 或本轮剪枝前模型。
2. 最好把 feature reconstruction 和少量 CE 一起保留，并在每一步后重新施加 mask。

如果这两点做对了，这一阶段可以作为最终 accuracy recovery 的增强步骤。

---

## 6. 最推荐你的实际落地顺序

如果你现在只想先做出“确实能涨点”的版本，不要一下子全改太多，建议按下面顺序推进：

### 第一步：把 teacher 改对

先不要用：

`teacher = copy.deepcopy(model)`

而是改成：

- 从磁盘加载最优 dense checkpoint
- 或者在当前 pruning step 之前先复制一份未剪模型

这是最关键的一步。

### 第二步：把 IA 和 REC 在训练循环里真正打开

也就是把 [train.py](/d:/毕业论文/调研/528_Towards_efficient_deep_spi_Supplementary%20Material/sca_code/train.py) 里那两段注释代码恢复到训练流程中：

1. `update_masks()`
2. `ptp_information_aggregation()`
3. `do_masks()`
4. `ptp_reconstruction()`
5. 固定 mask 微调

### 第三步：最后再做 adaptive final reconstruction

只有前面的流程稳定以后，再加最后这一步增强，不然很难判断提升到底来自哪里。

---

## 7. 你可以直接写进论文的方法流程

如果你要把这部分写进论文，可以用下面这条主线：

1. 先基于通道唯一性和脉冲门控构造重要性评分，完成结构化剪枝。
2. 对即将被剪除通道，在剪枝前通过教师特征约束与递增正则项完成信息聚合。
3. 剪枝后以教师网络为参照，对关键中间层特征进行重建，并辅以少量监督损失恢复分类能力。
4. 在固定掩码条件下进行渐进式微调，并根据层扰动度、平均脉冲率和深度先验执行自适应最终恢复。

这条叙述和你当前代码方向是相容的，也比较像论文语言。

---

## 8. 建议引用的文献口径

下面这些文献可作为你写这一部分时的主要依据：

1. Han S, Pool J, Tran J, et al. Learning both weights and connections for efficient neural network[C]//Advances in Neural Information Processing Systems 28. 2015.
   作用：连接稀疏化与非零连接统计的经典来源。

2. He Y, Zhang X, Sun J. Channel pruning for accelerating very deep neural networks[C]//Proceedings of the IEEE International Conference on Computer Vision. 2017: 1389-1397.
   作用：结构化通道剪枝与特征重建思想的经典来源。

3. He Y, Kang G, Dong X, et al. Soft filter pruning for accelerating deep convolutional neural networks[C]//Proceedings of the 27th International Joint Conference on Artificial Intelligence. 2018: 2234-2240.
   作用：渐进式剪枝与恢复思路参考。

4. Roy K, Jaiswal A, Panda P. Towards spike-based machine intelligence with neuromorphic computing[J]. Nature, 2019, 575(7784): 607-617.
   作用：SNN 中 AC/MAC 与能耗讨论的总体依据。

5. Rueckauer B, Lobo J, Takkar A, et al. Neuron-level sparsity and mixed precision for energy-efficient neural network inference on neuromorphic hardware[C]//Proceedings of Machine Learning and Systems. 2023, 5: 1-17.
   作用：脉冲稀疏性、事件驱动计算与能耗估计的参考。

6. Li Y, Xu Q, Shen J, et al. Towards efficient deep spiking neural networks construction with spiking activity based pruning[C]//Proceedings of the 41st International Conference on Machine Learning. 2024, 235: 29063-29073.
   作用：面向 SNN 的脉冲活动剪枝和结构化压缩的直接参考。

7. Han B, Zhao F, Pan W, et al. Adaptive sparse structure development with pruning and regeneration for spiking neural networks[J]. Information Sciences, 2025, 689: 121481.
   作用：再生和恢复机制在 SNN 中的直接参考。

---

## 9. 最后给你的结论

如果你只看一句话，那么结论是：

1. 你现在的“连接率”思路基本对。
2. 你现在的“参数量”对 `snnvgg16_bn` 来说低估了，因为分类头没算完整。
3. 你现在的 `SynOps` 是近似思路，但应改成“按层输入 spike rate 计算”，而不是“按本层输出 spike rate 计算”。
4. 你现在的“剪后重建”方向是对的，但目前真正生效的流程还没完全接起来，尤其 teacher 选取和 IA/REC 调用位置需要调整。

只要把这几处理顺，你这部分论文方法就会扎实很多。
