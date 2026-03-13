# 1.1 研究背景与研究意义

## 1.1.1 脉冲神经网络的发展背景

随着人工智能技术在智能感知、边缘计算、无人系统和物联网等场景中的持续渗透，智能模型的部署形态正在由云端集中计算逐步走向终端侧、边缘侧和在线连续处理。这一变化使模型在获得较高任务性能的同时，还必须满足低时延、低存储占用和低能耗等现实约束。传统深度神经网络在图像识别、语音处理和自然语言理解等任务中已取得显著进展，但其以高密度乘加运算为核心的同步计算范式通常依赖较大的算力与存储资源，在资源受限设备上部署时往往面临功耗较高、响应延迟较大和持续运行能力不足等问题[1-3]。在此背景下，类脑计算与神经形态计算逐渐成为后摩尔时代智能计算的重要研究方向，脉冲神经网络（Spiking Neural Networks, SNNs）因兼具生物合理性、事件驱动性和时空信息处理能力而受到广泛关注[4-6]。

从理论发展脉络看，Maass 早在 1997 年就将脉冲神经网络概括为“第三代神经网络”，指出其相较于感知机模型和传统激活值神经网络，能够利用脉冲时序携带更丰富的信息表达能力[4]。与人工神经网络采用连续值激活不同，SNN 通过离散脉冲在神经元之间传递信息，神经元仅在膜电位达到阈值时才发放脉冲，因此天然具有稀疏计算和异步处理特征[4][7]。随着 IF、LIF 等典型神经元模型的发展，以及频率编码、时间编码和相位编码等信息表达方式的完善，SNN 的建模基础和理论体系逐步成熟[5][7]。

在硬件层面，神经形态芯片的发展进一步强化了 SNN 的研究价值。TrueNorth、Loihi、SpiNNaker 和 Tianjic 等代表性平台表明，事件驱动硬件与脉冲计算模型之间具有较好的协同关系，能够在特定任务中显著降低无效计算与数据搬运开销[8-11]。这种“模型结构 - 编码方式 - 硬件架构”协同演进的趋势，使 SNN 不再只是神经科学启发下的理论模型，而逐步成为低功耗智能计算的重要候选方案。

近年来，随着代理梯度学习、时空反向传播和可学习神经元动力学等方法的提出，深层 SNN 的训练能力明显增强[6][12-14]。研究者已经能够在较深网络结构上直接训练 SNN，并将其应用到图像分类、目标检测、事件视觉和序列决策等任务中[13-17]。这表明 SNN 已经从早期依赖生物启发式局部学习规则的小规模模型，逐步发展为能够承担复杂任务的深层学习模型，也为进一步研究其结构压缩、部署优化和工程应用奠定了基础。

## 1.1.2 脉冲神经网络能耗优化的研究意义

脉冲神经网络之所以被认为适合低功耗场景，关键在于其采用事件驱动的计算机制。对于未发放脉冲的神经元，网络通常无需执行完整的连续值激活计算，因此在理论上能够减少乘加操作、访存次数和冗余更新[1][6][9]。尤其在神经形态硬件上，SNN 的能耗开销更多体现为脉冲触发下的累加操作，而非传统深度网络中大量密集浮点乘法，这使其在边缘感知、持续在线监测和嵌入式智能等应用中展现出明显潜力[1][8-10]。

但应当指出，SNN 具有低功耗潜力，并不意味着其在实际部署中天然达到最优能效。为了追求更高精度，当前深层 SNN 往往采用更深的网络层次、更宽的通道配置和更多的时间步展开，这虽然提升了模型表达能力，却也带来了参数冗余、通道冗余和时序冗余[5][13-15]。从系统实现角度看，模型能耗不仅取决于参数量，还与脉冲发放率、突触操作数（Synaptic Operations, SOPs）、时间步长度以及数据访存行为密切相关[1][18-19]。如果网络中存在大量低贡献通道或低利用率连接，即使模型本身具备脉冲稀疏特性，也仍会产生不必要的计算和能量消耗。

因此，围绕 SNN 开展能耗优化研究具有重要的理论意义和现实价值。就方法研究而言，能耗优化有助于推动 SNN 从“理论上具有低功耗潜力”走向“实际部署中实现高能效”，使其更好适配边缘端和专用芯片的应用需求。就系统应用而言，若能够在保证精度的同时有效减少冗余结构与无效脉冲活动，将进一步提升 SNN 在实时感知、移动终端、智能机器人和无人平台中的部署可行性。就学术价值而言，能耗优化问题天然涉及模型结构、时空动态和硬件执行特性之间的耦合关系，对该问题的深入研究有助于深化对 SNN 内部运行机理的理解，并为类脑模型设计和神经形态计算提供新的理论支撑。

## 1.1.3 模型剪枝在轻量化智能中的应用价值

模型剪枝是深度神经网络压缩的重要技术路线，其基本思想是在尽量保持模型性能的前提下，删除对任务贡献较低的权值、卷积核、通道或层级结构，以降低模型规模、运算复杂度和实际部署成本[20-22]。从早期的 Optimal Brain Damage 和 Optimal Brain Surgeon，到深度学习阶段的迭代剪枝、结构化剪枝与资源约束剪枝，剪枝技术已经成为轻量化智能研究中的核心方法之一[20-21][23-28]。相较于量化、知识蒸馏等模型压缩技术，剪枝在直接减少模型冗余结构、改善推理开销方面具有较强针对性，也更适合与硬件部署需求结合开展联合优化。

对于 SNN 而言，剪枝的意义不仅在于减少参数量，更在于从结构层面抑制冗余脉冲传播与无效时序计算。由于脉冲神经网络的能耗与突触操作次数、脉冲稀疏性和时间步展开密切相关，删除低贡献通道和冗余连接，往往能够同时带来参数压缩、计算量下降以及脉冲活动减少等多方面收益[18-19][29]。特别是结构化剪枝能够直接删除通道、卷积核或层结构，生成更规则的稀疏模型，因而更有利于在实际硬件和软件平台中获得稳定的加速与节能效果[24-28]。

但与传统 ANN 相比，SNN 剪枝面临更复杂的评估与优化问题。一方面，SNN 中结构单元的重要性不仅由静态权值决定，还与脉冲发放行为、时间依赖关系和层间动态协同密切相关；另一方面，剪枝后网络性能恢复也更加困难，过度删减可能破坏脉冲时空特征表达，导致精度显著下降[6][13-14][29-31]。因此，将模型剪枝引入面向能耗优化的脉冲神经网络研究，不仅具有明确的工程应用价值，也具有较强的理论探索意义。

# 1.2 国内外研究现状

## 1.2.1 脉冲神经网络研究现状

脉冲神经网络研究起源于对生物神经元放电机制的抽象建模，早期工作主要关注脉冲神经元模型、时序编码方式以及 Hebbian 学习、STDP 等局部学习规则[4-5][7]。这一阶段的研究强调生物合理性，但由于脉冲发放函数不可导、时间维度依赖强、优化过程不稳定，深层网络训练长期存在困难，SNN 在复杂任务上的性能也一度明显落后于传统深度神经网络。

近年来，随着代理梯度方法的发展，SNN 的训练瓶颈得到明显缓解。Neftci 等系统总结了代理梯度学习的基本思想，指出利用可导近似替代脉冲函数真实梯度，是将反向传播成功引入 SNN 的关键技术[6]。Wu 等提出时空反向传播方法，在时间维度展开网络并联合优化空间与时间参数，使 SNN 能够直接完成端到端训练[12]。Fang 等进一步引入可学习膜时间常数，使神经元动力学参数与权重共同更新，增强了网络对复杂时序信息的建模能力[13]。这些工作推动了 SNN 从浅层、低复杂度模型向深层、高性能模型演化。

在应用层面，深层 SNN 已逐步扩展到图像分类、目标检测、事件相机视觉、文本处理和机器人控制等任务。直接训练的大规模 SpikingVGG、SpikingResNet 等模型证明了深层 SNN 在静态视觉任务上的可行性[14-16]；Spiking-YOLO 等工作则表明 SNN 具备向更复杂视觉理解任务拓展的潜力[17]。总体来看，现有研究已经基本证明 SNN 不再局限于“低复杂度、低精度”的实验模型，而正在向高性能类脑智能模型迈进。

尽管如此，现阶段 SNN 研究仍存在两个突出问题。其一，为了弥补训练难度和表达能力不足，许多方法倾向于增加网络深度、宽度和时间步，导致模型结构持续膨胀[13-16]。其二，SNN 的能耗优势往往需要在特定硬件或特定稀疏条件下才能充分体现，若模型脉冲活动过于密集、时间展开过长，其综合效率未必优于经过充分优化的 ANN 模型[1][18-19]。因此，如何在保持性能的同时减少冗余结构和无效计算，已经成为当前 SNN 研究的重要方向。

## 1.2.2 神经网络剪枝方法研究现状

剪枝研究最早可追溯至 LeCun 等提出的 Optimal Brain Damage 和 Hassibi 等提出的 Optimal Brain Surgeon。这两类方法通过分析参数扰动对损失函数的影响来判断连接的重要性，为后续“基于重要性评估的剪枝”奠定了理论基础[20-21]。进入深度学习时代后，随着模型规模急剧增长，剪枝重新受到广泛重视。Han 等提出迭代式权值剪枝方法，通过“训练 - 剪枝 - 微调”流程删除大量低重要性连接，在不显著损失精度的情况下大幅压缩模型参数[23]；随后又在 Deep Compression 中进一步结合量化和编码策略，展示了深度模型压缩的系统化路径[22]。

从粒度上看，剪枝方法大致可分为非结构化剪枝和结构化剪枝两类。非结构化剪枝主要针对单个权值或连接，典型代表包括基于幅值的权值剪枝、L0 正则化稀疏学习、训练前一次性剪枝 SNIP，以及训练过程中动态更新连接拓扑的 RigL 等[23][30-31]。这类方法通常可以获得较高的参数稀疏率，但产生的是不规则稀疏结构，对通用硬件并不友好，因此实际加速收益往往有限。

结构化剪枝则更关注卷积核、通道、层级或模块的整体删除，目标是在压缩模型的同时保留规则结构，以便获得更稳定的推理加速和部署收益。Molchanov 等利用泰勒展开评估卷积通道的重要性，为通道级剪枝提供了较强的可解释性[24]；He 等提出基于特征图重构的通道剪枝方法，在精度和加速之间取得了较好平衡[25]；Luo 等提出 ThiNet，通过分析下一层输入统计关系选择保留滤波器，增强了过滤器级剪枝的实用性[26]；Liu 等提出 Network Slimming，将通道选择与训练过程中的稀疏约束结合起来，使结构化剪枝更加自动化[27]；He 等提出 Soft Filter Pruning，通过“软删除”而非永久移除滤波器，提高了剪枝过程的柔性和后续恢复能力[28]。这些方法使结构化剪枝逐步成为轻量化模型研究中的主流方向。

随着边缘部署需求增强，研究者开始将剪枝目标从“参数最少”扩展到“实际资源最优”。例如，Energy-Aware Pruning 将能耗估计显式纳入卷积网络压缩过程，强调参数量减少并不必然等价于能耗最优[32]；NetAdapt 通过面向特定平台的迭代调整，使压缩结果更贴近真实部署场景[33]；AMC 和 MetaPruning 分别通过强化学习和元学习自动决定各层压缩比例，推动剪枝从经验式设计走向自动化决策[34-35]。这些工作表明，现代剪枝研究已经从单纯的模型压缩问题，发展为与硬件约束、系统目标和任务需求紧密耦合的综合优化问题。

总体而言，ANN 剪枝研究已经形成较为成熟的技术体系，但其大多数方法是基于连续值激活和静态前向传播建立的。SNN 具有脉冲稀疏性、时间动态性和神经元状态依赖性，这意味着 ANN 中常用的权值幅值、特征重构误差或静态激活统计，未必能够充分反映 SNN 结构单元的真实贡献。由此可见，将传统剪枝思路直接迁移到 SNN 并不能自然得到理想结果，必须结合脉冲活动特性重新设计重要性评估和压缩策略。

## 1.2.3 面向能耗优化的模型压缩研究现状

模型压缩与能耗优化的结合，近年来已成为轻量化智能研究的重要趋势。对于传统深度神经网络，已有大量工作从参数规模、浮点运算量、硬件延迟和能量开销等多个维度评估压缩效果，并逐步认识到“模型更小”并不必然意味着“运行更省电”[3][18][32-33]。特别是在边缘端部署场景中，访存代价、数据复用模式和硬件执行特性都会显著影响最终能耗，因此压缩方法需要更贴近实际资源约束。

在脉冲神经网络领域，能耗优化问题具有更强的模型特异性。SNN 的能耗不仅与参数量有关，还与脉冲发放率、时间步长度、突触事件数量以及神经形态硬件上的事件调度方式密切相关[1][18-19]。因此，单纯借鉴 ANN 中以参数量或 FLOPs 为核心的压缩评价方式，往往难以准确反映 SNN 的真实能耗收益。近年来，一些研究开始尝试从脉冲活动或时空稀疏性出发开展 SNN 压缩。Li 等提出基于脉冲活动统计的深层 SNN 剪枝方法，通过利用发放行为指导结构裁剪，证明了面向脉冲特性的结构化剪枝在深层 SNN 上具有可行性[36]。Han 等进一步提出受发育可塑性启发的自适应剪枝方法，以及结合剪枝与再生的稀疏结构演化方法，说明恢复机制对于维持高压缩率下的模型性能具有重要作用[37-38]。此外，也有研究在特定任务场景中探索时空联合剪枝，以同时兼顾空间特征提取与时间动态建模[39]。

现有研究虽然为 SNN 能耗优化提供了有益探索，但仍存在明显不足。第一，许多工作仍将能耗作为实验结果中的附加指标，而没有将其真正纳入通道重要性评估和剪枝决策核心。第二，剪枝后恢复和微调过程大多依赖经验规则，缺少能够表征“恢复价值”或“再生潜力”的专门指标。第三，训练、剪枝、恢复、能耗评估与结果展示常常彼此分离，尚未形成统一的研究闭环。总体而言，当前面向能耗优化的 SNN 压缩研究已经从“单纯结构删减”发展到“结构删减与恢复协同”的阶段，但在指标设计、恢复机制和平台化支撑方面仍有较大提升空间。

## 1.2.4 现有研究的不足分析

综合上述国内外研究现状可以看出，现有工作虽然为脉冲神经网络压缩和能耗优化奠定了基础，但仍存在以下不足。

其一，现有 SNN 剪枝方法对结构重要性的刻画仍不充分。许多方法直接借鉴 ANN 的幅值、梯度或激活统计指标，尚未充分考虑脉冲发放频率、时间分布、膜电位动态及层间时空协同对通道贡献的影响，因此评估结果可能与真实任务贡献存在偏差。

其二，现有研究对剪枝后性能退化的恢复机制关注仍然不足。虽然已有工作开始探索结构再生和动态演化，但对于哪些通道应优先恢复、恢复比例如何设定、恢复后如何与微调协同优化等问题，仍缺乏统一而可操作的理论依据。

其三，能耗约束在很多方法中尚未成为核心优化目标。一些研究虽然报告了参数压缩率或能耗改善比例，但相关指标更多停留在实验后评价层面，并未深入参与到剪枝决策和恢复策略中，导致“面向能耗优化”的方法设计深度仍显不足。

其四，面向方法验证和工程应用的一体化支撑仍较薄弱。现有研究多集中在算法本身，缺少覆盖训练、剪枝、恢复、能耗评估和可视化分析的统一平台，这在一定程度上影响了方法复现性、横向比较和工程落地能力。

# 1.3 研究问题与挑战

## 1.3.1 脉冲神经网络结构冗余问题

随着深层 SNN 在复杂任务中的应用不断增加，网络通常需要更深层次结构、更宽通道配置以及更长时间步展开来提升表达能力。然而，这种“以规模换性能”的方式也使模型中出现明显的参数冗余、通道冗余和时序冗余。对于 SNN 来说，冗余结构不仅意味着模型规模增大，还可能引入多余的脉冲传递和突触操作，直接影响综合能耗表现。因此，如何准确识别真正低贡献的结构单元，是面向能耗优化剪枝研究的首要问题。

## 1.3.2 剪枝过程中精度保持困难问题

与 ANN 相比，SNN 的时空耦合特性更强。某些通道虽然在静态统计意义下看似贡献较低，但在特定时间步、特定类别或特定特征传播路径中可能承担关键作用。一旦简单依据静态指标进行大幅删减，可能破坏脉冲发放节律、膜电位积累过程以及层间信息传递链路，进而导致识别精度明显下降，甚至训练和微调过程不稳定。因此，如何在压缩率、能耗收益与模型性能之间取得合理平衡，是 SNN 剪枝需要解决的核心挑战。

## 1.3.3 剪枝后网络恢复能力不足问题

现有很多剪枝方法侧重于“如何删除”，但对“删除后如何恢复”考虑不足。对于高压缩率场景，仅依赖常规微调往往难以完全弥补关键结构被误删带来的损失。尤其在 SNN 中，剪枝对脉冲动态过程的影响可能具有累积性和层间传导性，一旦某些关键通道被删除，后续层的特征表达和时序响应都可能受到连锁影响。因此，需要构建能够量化通道恢复价值的再生指标，并据此设计有针对性的恢复与渐进式微调机制，以增强剪枝后网络的恢复能力和鲁棒性。

## 1.3.4 能耗优化与模型性能协同困难问题

面向部署的模型优化从来不是单一目标问题。对于 SNN 而言，参数量减少、脉冲活动降低、能耗下降和精度保持之间并不总是同步变化，某一目标的改善可能会牺牲另一目标。例如，过度追求极高压缩率可能导致脉冲表达能力受损，而单纯追求精度保持又可能使能耗收益有限。因此，如何围绕能耗优化目标，构建贯穿“重要性评估 - 剪枝决策 - 通道恢复 - 渐进式微调 - 能耗评估”的统一方法框架，实现模型性能与能源效率的协同优化，是本文需要重点解决的关键问题。

基于以上分析，本文将围绕面向能耗优化的脉冲神经网络剪枝方法展开研究，重点聚焦于通道重要性评估指标设计、再生指标设计以及剪枝后重建与渐进式微调机制，并进一步结合平台化实现，形成从算法设计到系统验证的完整研究链条。

# 参考文献

[1] ROY K, JAISWAL A, PANDA P. Towards spike-based machine intelligence with neuromorphic computing[J]. Nature, 2019, 575(7784): 607-617. DOI:10.1038/s41586-019-1677-2.

[2] SZE V, CHEN Y H, YANG T J, et al. Efficient processing of deep neural networks: A tutorial and survey[J]. Proceedings of the IEEE, 2017, 105(12): 2295-2329. DOI:10.1109/JPROC.2017.2761740.

[3] CHENG Y, WANG D, ZHOU P, et al. Model compression and acceleration for deep neural networks: The principles, progress, and challenges[J]. IEEE Signal Processing Magazine, 2018, 35(1): 126-136. DOI:10.1109/MSP.2017.2765695.

[4] MAASS W. Networks of spiking neurons: The third generation of neural network models[J]. Neural Networks, 1997, 10(9): 1659-1671. DOI:10.1016/S0893-6080(97)00011-7.

[5] GERSTNER W, KISTLER W M, NAUD R, et al. Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition[M]. Cambridge: Cambridge University Press, 2014.

[6] NEFTCI E O, MOSTAFA H, ZENKE F. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks[J]. IEEE Signal Processing Magazine, 2019, 36(6): 51-63. DOI:10.1109/MSP.2019.2931595.

[7] GUO Y, HUANG X, MA Z. Direct learning-based deep spiking neural networks: A review[J]. Frontiers in Neuroscience, 2023, 17: 1209795. DOI:10.3389/fnins.2023.1209795.

[8] MEROLLA P A, ARTHUR J V, ALVAREZ-ICAZA R, et al. A million spiking-neuron integrated circuit with a scalable communication network and interface[J]. Science, 2014, 345(6197): 668-673. DOI:10.1126/science.1254642.

[9] DAVIES M, SRINIVASA N, LIN T H, et al. Loihi: A neuromorphic manycore processor with on-chip learning[J]. IEEE Micro, 2018, 38(1): 82-99. DOI:10.1109/MM.2018.112130359.

[10] FURBER S B, GALLUPPI F, TEMPLE S, et al. The SpiNNaker project[J]. Proceedings of the IEEE, 2014, 102(5): 652-665. DOI:10.1109/JPROC.2014.2304638.

[11] PEI J, DENG L, SONG S, et al. Towards artificial general intelligence with hybrid Tianjic chip architecture[J]. Nature, 2019, 572(7767): 106-111. DOI:10.1038/s41586-019-1424-8.

[12] WU Y, DENG L, LI G, et al. Direct training for spiking neural networks: Faster, larger, better[C]//Proceedings of the AAAI Conference on Artificial Intelligence. Palo Alto: AAAI Press, 2019, 33(1): 1311-1318. DOI:10.1609/aaai.v33i01.33011311.

[13] FANG W, YU Z, CHEN Y, et al. Incorporating learnable membrane time constant to enhance learning of spiking neural networks[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. Montreal: IEEE, 2021: 2661-2671.

[14] ZHENG H, WU Y, DENG L, et al. Going deeper with directly-trained larger spiking neural networks[C]//Proceedings of the AAAI Conference on Artificial Intelligence. Palo Alto: AAAI Press, 2021, 35(12): 11062-11070.

[15] FANG W, YU Z, CHEN Y, et al. Deep residual learning in spiking neural networks[C]//Advances in Neural Information Processing Systems. Red Hook: Curran Associates, 2021, 34: 21056-21069.

[16] HU Y, TANG H, PAN G. Spiking deep residual networks[J]. IEEE Transactions on Neural Networks and Learning Systems, 2024, 35(4): 5141-5155. DOI:10.1109/TNNLS.2021.3104247.

[17] KIM S, PARK S, NA B, et al. Spiking-YOLO: Spiking neural network for energy-efficient object detection[C]//Proceedings of the AAAI Conference on Artificial Intelligence. Palo Alto: AAAI Press, 2020, 34(7): 11270-11277. DOI:10.1609/aaai.v34i07.6787.

[18] HOROWITZ M. 1.1 Computing's energy problem (and what we can do about it)[C]//2014 IEEE International Solid-State Circuits Conference Digest of Technical Papers. Piscataway: IEEE, 2014: 10-14. DOI:10.1109/ISSCC.2014.6757323.

[19] RUECKAUER B, LOBO J, TAKKAR A, et al. Neuron-level sparsity and mixed precision for energy-efficient neural network inference on neuromorphic hardware[C]//Proceedings of Machine Learning and Systems. 2023, 5: 1-17.

[20] LECUN Y, DENKER J S, SOLLA S A. Optimal brain damage[C]//Advances in Neural Information Processing Systems 2. San Francisco: Morgan Kaufmann, 1990: 598-605.

[21] HASSIBI B, STORK D G. Second order derivatives for network pruning: Optimal Brain Surgeon[C]//Advances in Neural Information Processing Systems 5. San Francisco: Morgan Kaufmann, 1993: 164-171.

[22] HAN S, MAO H, DALLY W J. Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding[EB/OL]. [2026-03-13]. https://arxiv.org/abs/1510.00149.

[23] HAN S, POOL J, TRAN J, et al. Learning both weights and connections for efficient neural network[C]//Advances in Neural Information Processing Systems 28. Red Hook: Curran Associates, 2015: 1135-1143.

[24] MOLCHANOV P, TYREE S, KARRAS T, et al. Pruning convolutional neural networks for resource efficient inference[C]//International Conference on Learning Representations. Toulon: ICLR, 2017.

[25] HE Y, ZHANG X, SUN J. Channel pruning for accelerating very deep neural networks[C]//Proceedings of the IEEE International Conference on Computer Vision. Venice: IEEE, 2017: 1389-1397. DOI:10.1109/ICCV.2017.155.

[26] LUO J H, WU J, LIN W. ThiNet: A filter level pruning method for deep neural network compression[C]//Proceedings of the IEEE International Conference on Computer Vision. Venice: IEEE, 2017: 5058-5066. DOI:10.1109/ICCV.2017.541.

[27] LIU Z, LI J, SHEN Z, et al. Learning efficient convolutional networks through network slimming[C]//Proceedings of the IEEE International Conference on Computer Vision. Venice: IEEE, 2017: 2736-2744. DOI:10.1109/ICCV.2017.298.

[28] HE Y, KANG G, DONG X, et al. Soft filter pruning for accelerating deep convolutional neural networks[C]//Proceedings of the 27th International Joint Conference on Artificial Intelligence. Stockholm: IJCAI, 2018: 2234-2240. DOI:10.24963/ijcai.2018/309.

[29] GUO Y, YAO A, CHEN Y. Dynamic network surgery for efficient DNNs[C]//Advances in Neural Information Processing Systems 29. Red Hook: Curran Associates, 2016: 1379-1387.

[30] LOUIZOS C, WELLING M, KINGMA D P. Learning sparse neural networks through L0 regularization[C]//International Conference on Learning Representations. Vancouver: ICLR, 2018.

[31] EVCI U, IOANNOU Y, KESKAR N S, et al. Rigging the lottery: Making all tickets winners[C]//Proceedings of the 37th International Conference on Machine Learning. Vienna: PMLR, 2020, 119: 2943-2952.

[32] YANG T J, CHEN Y H, SZE V. Designing energy-efficient convolutional neural networks using energy-aware pruning[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. Honolulu: IEEE, 2017: 5687-5695. DOI:10.1109/CVPR.2017.604.

[33] YANG T J, HOWARD A, CHEN B, et al. NetAdapt: Platform-aware neural network adaptation for mobile applications[C]//Proceedings of the European Conference on Computer Vision. Munich: Springer, 2018: 285-300. DOI:10.1007/978-3-030-01249-6_18.

[34] HE Y, LIN J, LIU Z, et al. AMC: AutoML for model compression and acceleration on mobile devices[C]//Proceedings of the European Conference on Computer Vision. Munich: Springer, 2018: 784-800. DOI:10.1007/978-3-030-01234-2_48.

[35] LIU Z, MU H, ZHANG X, et al. MetaPruning: Meta learning for automatic neural network channel pruning[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. Seoul: IEEE, 2019: 3296-3305. DOI:10.1109/ICCV.2019.00339.

[36] LI Y, XU Q, SHEN J, et al. Towards efficient deep spiking neural networks construction with spiking activity based pruning[C]//Proceedings of the 41st International Conference on Machine Learning. Vienna: PMLR, 2024, 235: 29063-29073.

[37] HAN B, ZHAO F, ZENG Y, et al. Developmental Plasticity-Inspired Adaptive Pruning for Deep Spiking and Artificial Neural Networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025, 47(1): 240-251. DOI:10.1109/TPAMI.2024.3467268.

[38] HAN B, ZHAO F, PAN W, et al. Adaptive sparse structure development with pruning and regeneration for spiking neural networks[J]. Information Sciences, 2025, 689: 121481. DOI:10.1016/j.ins.2024.121481.

[39] LI C, WU H, HUANG Y, et al. Deep spatio-temporal pruning for efficient spiking neural network based object detection in remote sensing images[J]. Remote Sensing, 2024, 16(17): 3200. DOI:10.3390/rs16173200.
