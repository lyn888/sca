# 1.1 研究背景与研究意义

随着人工智能技术在边缘计算、智能感知、无人系统和物联网等场景中的持续渗透，模型部署环境对计算效率、存储成本和功耗水平提出了更为严格的要求。传统深度神经网络虽然在图像识别、语音理解和序列建模等任务中取得了显著成果，但其高密度、同步式的计算范式通常伴随着较高的算力消耗与能量开销，难以完全满足低功耗、低时延和持续在线的应用需求[1]。在此背景下，类脑智能和神经形态计算逐渐成为人工智能研究的重要方向，脉冲神经网络（Spiking Neural Networks, SNNs）因其事件驱动、时空联合处理和较强生物合理性等特点，受到越来越多研究者的关注[1-3]。

与传统人工神经网络使用连续值激活不同，脉冲神经网络通过离散脉冲传递信息，仅在神经元膜电位达到阈值时触发发放与计算。这种稀疏、异步的处理方式使其在理论上具有更高的能量利用效率，也更适合与神经形态硬件协同部署[1-2]。Roy 等指出，脉冲驱动的神经形态计算为实现低功耗机器智能提供了新的技术路径[1]；Neftci 等则系统总结了替代梯度学习对深层脉冲神经网络训练的推动作用，表明 SNN 已从早期的小规模生物启发模型逐步发展为能够承担复杂任务的深层学习模型[2]。随着直接训练方法、可学习神经元动力学和更深层网络结构的发展，SNN 的性能持续提升，其应用范围也从基础分类任务逐步扩展到检测、分割和时序决策等更复杂场景[3-5]。

然而，脉冲神经网络的低功耗潜力并不意味着其在实际部署中天然具备最优能效。一方面，为追求更高识别精度和更强表征能力，当前深层 SNN 往往引入更深的层次结构、更宽的通道配置以及更多时间步展开，这会带来参数冗余、通道冗余和时序冗余[3-5]。另一方面，脉冲神经网络的能耗不仅与参数规模相关，还与脉冲发放稀疏性、突触操作次数以及时空动态过程密切相关。如果网络中存在大量低贡献通道或无效连接，即使采用事件驱动机制，也可能造成不必要的计算与访存开销，从而削弱其在低功耗场景中的部署优势[1][3]。因此，仅依赖脉冲神经网络自身的事件驱动属性，仍难以充分释放其能耗优化潜力，必须进一步结合模型压缩与结构优化技术开展研究。

剪枝技术是神经网络压缩的重要分支，其核心思想是在尽量保持模型性能的前提下，移除对任务贡献较低的连接、卷积核、通道或层级结构，以降低参数规模、运算量和存储访问成本。自 Optimal Brain Damage、Optimal Brain Surgeon 等早期工作以来，神经网络剪枝已经形成较为清晰的发展脉络，并在深度神经网络压缩中展现出显著效果[6-11]。近年来，剪枝思想逐步从传统人工神经网络扩展到脉冲神经网络领域，研究者开始关注如何结合脉冲活动特性、时间动态规律和神经形态部署需求，设计更适合 SNN 的剪枝与恢复方法[25-28]。但与 ANN 相比，SNN 在结构重要性评估、剪枝后性能恢复和能耗建模等方面具有更高复杂性，现有方法仍存在评估依据不充分、恢复机制不完善以及能耗约束不突出等问题。

基于此，研究面向能耗优化的脉冲神经网络剪枝方法具有重要的理论意义与工程价值。从理论层面看，围绕脉冲神经网络的通道重要性评估、再生指标设计、剪枝恢复与渐进式微调等问题开展研究，有助于深化对 SNN 结构冗余、脉冲动态退化机理和能耗优化机制的认识，丰富类脑模型压缩方法体系。从应用层面看，若能够在尽量保持模型精度的前提下有效降低参数规模、脉冲活动和综合能耗，则有望进一步提升脉冲神经网络在边缘设备、神经形态芯片及低功耗智能系统中的部署可行性。因此，围绕面向能耗优化的脉冲神经网络剪枝方法展开系统研究，具有较强的学术意义和现实价值。

# 1.2 国内外研究现状

## 1.2.1 脉冲神经网络训练与建模研究现状

脉冲神经网络的早期研究主要聚焦于生物合理的神经元建模、脉冲编码机制以及局部学习规则，但由于脉冲发放函数不可导、时间维度依赖显著、优化过程不稳定，深层 SNN 的训练长期受到限制[2-3]。随着替代梯度方法的发展，研究者开始通过可导近似替代脉冲函数的真实梯度，使反向传播机制能够扩展至脉冲神经网络。Neftci 等对该方向进行了系统总结，指出替代梯度学习极大提升了 SNN 的可训练性，是推动深层脉冲网络快速发展的关键因素之一[2]。

在直接训练方法方面，Wu 等提出时空反向传播框架，通过在时间维度展开网络并结合替代梯度进行参数更新，使脉冲神经网络能够在更大规模任务中实现端到端训练[4]。Fang 等进一步提出可学习膜时间常数机制，使神经元动力学参数与连接权重协同优化，从而增强网络的时间建模能力并提升训练稳定性[5]。Guo 等在综述中指出，近年来深层 SNN 的研究重点主要集中于训练精度提升、时间步压缩、神经元动力学优化以及网络结构改进等方向[3]。这些工作共同推动了 SNN 从理论模型向实用模型的转变。

尽管如此，现有研究也表明，深层脉冲神经网络在提升任务性能的同时，往往引入更多网络层数、更宽通道配置和更长时间展开步数，这使得结构冗余与时序冗余问题逐步凸显[3-5]。换言之，SNN 的低功耗潜力并不会自动转化为实际部署中的最优能效表现。如何在保持性能的同时减少冗余结构和无效脉冲活动，已经成为脉冲神经网络研究的重要问题，也为后续开展模型压缩与剪枝优化研究提供了现实动因。

## 1.2.2 神经网络剪枝方法研究现状

神经网络剪枝的核心目标是在尽量维持模型性能的前提下，移除冗余参数或结构单元，以降低模型规模、计算复杂度、存储访问开销以及实际部署成本。围绕这一目标，国内外研究已经形成较为完整的方法体系。若从发展逻辑来看，相关研究大致经历了从二阶敏感度分析到深度网络迭代剪枝、从非结构化稀疏化到结构化压缩、再到资源约束驱动与自动化决策的发展过程[6-11]。若从方法类型来看，则可以从非结构化剪枝、结构化剪枝以及资源或能耗感知剪枝三个层面加以梳理[11]。

从发展脉络上看，神经网络剪枝的早期代表性工作主要是 LeCun 等提出的 Optimal Brain Damage 方法以及 Hassibi 等提出的 Optimal Brain Surgeon 方法[6-7]。这两类方法均建立在二阶导数分析基础上，通过估计参数删除对损失函数的影响来评估其重要性，为后续“重要性评估驱动剪枝”的研究路线奠定了理论基础。进入深度学习阶段后，随着模型规模急剧增长，剪枝研究重新受到重视。Han 等提出基于权值幅值的迭代式剪枝方法，通过“训练-剪枝-再训练”流程删除大量低重要性连接，有效降低了模型参数量[8]；随后又在 Deep Compression 中将剪枝、量化和编码联合起来，进一步压缩模型存储成本[9]。Guo 等提出动态网络外科方法，在剪除冗余连接的同时允许部分已删除连接在训练过程中恢复，从而减轻不可逆剪枝带来的性能损失[10]。这些工作表明，剪枝方法已经从静态参数删减逐步发展为动态结构优化过程。

在非结构化剪枝方面，研究主要围绕单个权值或连接的重要性建模展开。这类方法通常能够获得较高稀疏率，具有较强的压缩能力。除基于权值幅值的经典方法外，Louizos 等提出通过 L0 正则化直接学习稀疏网络结构，从优化目标层面推动参数级稀疏化[12]；Lee 等提出 SNIP 方法，在训练开始前基于连接敏感度进行一次性剪枝，避免了复杂的多轮迭代训练过程[13]；Frankle 和 Carbin 提出的 Lottery Ticket Hypothesis 则从可训练子网络角度揭示了稀疏模型的潜在表达能力[14]；Evci 等提出 RigL 方法，在训练过程中动态更新稀疏连接拓扑，使网络能够在固定稀疏预算下持续演化并维持较强性能[15]。总体来看，非结构化剪枝在理论压缩率上具有优势，也为稀疏训练研究提供了重要基础，但其生成的不规则稀疏结构往往不利于通用硬件高效执行，实际加速效果通常低于参数压缩幅度。

与之相比，结构化剪枝更关注卷积核、通道、层级或块结构的整体删除，目标是在压缩模型的同时保留规则结构，以便直接获得更稳定的硬件加速收益。Molchanov 等基于泰勒展开构建卷积通道的重要性评估准则，为卷积层结构化删减提供了可解释的依据[16]。He 等提出面向网络加速的通道剪枝方法，通过最小化特征重构误差选择保留通道，实现了较好的精度和速度平衡[17]。Luo 等提出 ThiNet，从下一层输入统计关系出发筛选保留滤波器，增强了过滤器级剪枝的实用性[18]。Liu 等提出 Network Slimming，通过在批归一化缩放因子上施加稀疏约束实现通道自动收缩，使结构化剪枝能够与训练过程更紧密结合[19]。He 等提出 Soft Filter Pruning，在训练过程中对滤波器执行软删除而非永久清零，使被剪枝结构仍保留一定恢复能力，从而提高模型压缩的柔性[20]。总体而言，结构化剪枝更加符合实际部署需求，也是当前模型轻量化与硬件协同优化中的主流方向之一。

随着移动端部署和边缘智能应用需求的增长，研究者进一步将剪枝目标从单纯减少参数量扩展到延迟、能耗和平台适配等更贴近实际系统约束的层面。Yang 等提出能量感知剪枝方法，将能耗估计引入卷积神经网络压缩过程，强调参数减少并不必然等价于能耗最优[21]。NetAdapt 针对特定硬件平台进行逐步网络调整，通过实测资源消耗反馈优化模型结构，提高了压缩结果与实际部署环境的一致性[22]。AMC 则引入强化学习策略自动决定各层压缩比例，使剪枝从人工经验设计转向自动决策[23]。MetaPruning 进一步利用元学习思想生成不同压缩率下的候选网络权重，降低了自动化通道剪枝的搜索成本[24]。这一阶段的研究表明，剪枝已不再只是一个静态压缩问题，而是与具体平台、资源约束和系统目标紧密耦合的综合优化问题。

总体来看，ANN 剪枝研究已经形成较为成熟的技术谱系。其方法既可以按照剪枝粒度划分为连接级、卷积核级、通道级和层级剪枝，也可以按照执行时机划分为训练后剪枝、训练时剪枝和动态剪枝，还可以按照优化目标划分为精度优先、压缩率优先、硬件延迟优先和能量效率优先等不同取向[11]。大量研究已经证明，剪枝是深度模型轻量化的重要技术路径。然而需要指出的是，这些方法大多建立在连续值激活、静态前向传播和传统算力模型基础之上，其重要性评估准则、误差重构机制和资源建模方式并不能无缝迁移到具有显著时间动态和脉冲稀疏特征的脉冲神经网络中。

近年来，随着深层 SNN 训练能力的提升，研究者开始尝试将剪枝思想拓展到脉冲神经网络压缩中。Li 等在 ICML 2024 提出基于脉冲活动的深层 SNN 结构化剪枝框架，通过脉冲活动统计指导卷积核裁剪，并结合再生机制缓解性能退化[25]。Han 等在 TPAMI 2025 中提出受发育可塑性启发的自适应剪枝方法，使深度脉冲神经网络能够在训练过程中动态完成结构删减与演化[26]。Han 等在 Information Sciences 2025 中进一步提出结合剪枝与再生的自适应稀疏结构发展方法，强调恢复机制在保持高压缩率网络性能方面的重要作用[27]。此外，Li 等面向遥感目标检测任务提出时空联合剪枝方法，说明在特定应用场景下，剪枝策略还需要同时考虑空间特征提取与时间动态建模[28]。这些研究表明，SNN 剪枝已经从直接借鉴 ANN 方法的早期阶段，逐步发展到结合脉冲活动特性、结构恢复机制和任务场景约束的专门化阶段。

尽管如此，与 ANN 剪枝相比，SNN 剪枝研究仍明显不足。首先，SNN 中结构单元的重要性不仅与权值幅值相关，还与脉冲发放频率、时间分布、膜电位动态和层间信息传播紧密耦合，现有方法尚缺乏统一而稳定的评估依据[2-5][25]。其次，虽然部分工作已经关注剪枝后的结构恢复，但对恢复潜力的量化刻画仍然不够充分，尚未形成清晰的恢复优先级判别机制[26-27]。再次，不少研究仍将能耗作为实验结果中的附带指标，而未将其真正融入剪枝决策与恢复优化过程，使得“面向能耗优化”的方法设计仍有较大改进空间。

## 1.2.3 面向能耗优化的脉冲神经网络压缩研究现状

脉冲神经网络之所以受到广泛关注，核心原因之一在于其事件驱动机制和神经形态硬件适配性所带来的低功耗潜力[1]。因此，SNN 压缩研究天然与能耗优化问题密切相关。现有工作通常从减少参数规模、降低突触操作次数、抑制无效脉冲活动和优化时间步展开等角度提升模型效率[1][3]。其中，一类研究通过改进编码方式、神经元动力学和训练策略减少无效时序计算；另一类研究则通过剪枝、稀疏训练和结构演化减少冗余计算单元[3][25-28]。

从已有结果看，基于脉冲活动的结构化剪枝能够在保持模型性能的同时降低计算负载，并为深层 SNN 在低功耗和高效率场景中的应用提供支持[25]。此外，结合结构恢复或再生机制的研究表明，仅依赖静态删除冗余结构往往不足以获得最佳压缩效果，若能在剪枝过程中引入恢复与再优化机制，则更有可能在较高压缩率下保持模型稳定性能[26-27]。这说明面向能耗优化的 SNN 压缩研究正在从“单纯删减结构”转向“剪枝-恢复-优化”的综合思路。

但总体而言，该方向仍存在若干明显不足。首先，已有研究通常将能耗作为结果指标进行展示，而较少将其直接纳入重要性评估或剪枝决策的核心目标[21][25-28]。其次，剪枝后的恢复与微调大多依赖经验设置，缺乏能够直接表征结构恢复价值的指标支持。再次，现有研究普遍更重视算法验证，缺少对训练、剪枝、恢复、能耗评估和结果展示进行统一支撑的平台化工具，这在一定程度上限制了方法复现、横向比较和工程落地。因此，围绕通道重要性评估、再生指标设计、剪枝恢复与渐进式微调以及平台化实现开展研究，仍具有较大的理论与应用价值。

# 1.3 研究问题与挑战

综合上述研究现状可以看出，脉冲神经网络在模型压缩与能耗优化方面虽然已经取得一定进展，但面向复杂任务和实际部署需求，仍存在若干亟待解决的关键问题。

第一，脉冲神经网络中通道冗余的刻画难度较大。与传统人工神经网络不同，SNN 中结构单元的重要性不仅取决于静态权值或激活幅值，还与脉冲发放频率、时间响应模式、膜电位累积过程以及层间动态传递关系紧密相关[2-5]。这使得 ANN 中常用的静态评估指标难以直接适用于 SNN 剪枝场景，容易出现误删关键结构或冗余结构保留不足的问题。因此，如何设计能够反映脉冲时空特性并兼顾能耗目标的通道重要性评估指标，是开展高效剪枝的首要挑战。

第二，剪枝后模型性能退化问题较为突出。脉冲神经网络内部存在显著的时空耦合关系，某些通道在静态统计意义上看似贡献有限，但在特定时间步或特定特征传播路径中可能具有重要作用。一旦剪枝策略过于激进，便可能破坏网络的特征表达能力和脉冲动态平衡，进而导致识别精度下降、训练稳定性变差甚至网络无法有效收敛[25-28]。因此，如何在压缩率、能耗收益与模型精度之间取得合理平衡，是面向 SNN 剪枝研究必须解决的核心问题。

第三，剪枝后恢复潜力缺乏有效量化依据。现有部分研究已经开始尝试通过结构再生、连接恢复或动态演化等方式缓解过剪枝带来的性能损失[10][26-27]，但对“哪些结构值得恢复、恢复优先级如何确定、恢复后如何实现稳定优化”等问题仍缺乏统一而清晰的判别标准。换言之，仅有剪枝机制仍不足以支撑高压缩条件下的稳定性能保持，还需要进一步设计能够表征结构恢复价值的再生指标，并据此构建更有针对性的剪枝恢复策略。

第四，能耗、精度与压缩率之间的协同优化仍然不足。当前很多研究虽然在结果中报告了参数量减少、计算量下降或能耗改善，但能耗往往并未真正融入方法设计过程，而更多停留在实验后评价层面[21][25-28]。此外，训练、剪枝、恢复、微调与能耗评估之间常常相互分离，缺少统一框架支撑，这使得研究结果在横向比较、可复现性和工程应用方面仍存在局限。因此，如何围绕能耗优化目标构建贯穿“剪枝-恢复-微调”的递进式方法框架，并进一步实现平台化集成，是提升研究完整性和应用价值的重要方向。

基于上述分析，本文将围绕通道重要性评估、再生指标设计、剪枝恢复与渐进式微调以及平台系统实现展开研究，力求在模型轻量化、能耗降低与性能保持之间取得更优折中，并为面向实际部署的脉冲神经网络优化提供系统化方法支撑。

# 参考文献

[1] ROY K, JAISWAL A, PANDA P. Towards spike-based machine intelligence with neuromorphic computing[J]. Nature, 2019, 575: 607-617. DOI: 10.1038/s41586-019-1677-2.

[2] NEFTCI E O, MOSTAFA H, ZENKE F. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks[J]. IEEE Signal Processing Magazine, 2019, 36(6): 51-63. DOI: 10.1109/MSP.2019.2931595.

[3] GUO Y, HUANG X, MA Z. Direct learning-based deep spiking neural networks: a review[J]. Frontiers in Neuroscience, 2023, 17: 1209795. DOI: 10.3389/fnins.2023.1209795.

[4] WU Y, DENG L, LI G, et al. Direct training for spiking neural networks: Faster, larger, better[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33(1): 1311-1318. DOI: 10.1609/AAAI.V33I01.33011311.

[5] FANG W, YU Z, CHEN Y, et al. Incorporating learnable membrane time constant to enhance learning of spiking neural networks[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021: 2661-2671.

[6] LECUN Y, DENKER J S, SOLLA S A. Optimal brain damage[C]//Advances in Neural Information Processing Systems 2. 1990.

[7] HASSIBI B, STORK D G. Second order derivatives for network pruning: Optimal Brain Surgeon[C]//Advances in Neural Information Processing Systems 5. 1993.

[8] HAN S, POOL J, TRAN J, et al. Learning both weights and connections for efficient neural network[C]//Advances in Neural Information Processing Systems 28. 2015.

[9] HAN S, MAO H, DALLY W J. Deep compression: Compressing deep neural networks with pruning, trained quantization and Huffman coding[EB/OL]. arXiv:1510.00149, 2015.

[10] GUO Y, YAO A, CHEN Y. Dynamic network surgery for efficient DNNs[C]//Advances in Neural Information Processing Systems 29. 2016.

[11] CHENG Y, WANG D, ZHOU P, et al. Model compression and acceleration for deep neural networks: The principles, progress, and challenges[J]. IEEE Signal Processing Magazine, 2018, 35(1): 126-136. DOI: 10.1109/MSP.2017.2765695.

[12] LOUIZOS C, WELLING M, KINGMA D P. Learning sparse neural networks through L_0 regularization[C]//International Conference on Learning Representations. 2018.

[13] LEE N, AHN T, TORR P H S. SNIP: Single-shot network pruning based on connection sensitivity[C]//International Conference on Learning Representations. 2019.

[14] FRANKLE J, CARBIN M. The lottery ticket hypothesis: Finding sparse, trainable neural networks[C]//International Conference on Learning Representations. 2019.

[15] EVCI U, IOANNOU Y, KESKAR N S, et al. Rigging the lottery: Making all tickets winners[C]//Proceedings of the 37th International Conference on Machine Learning. PMLR, 2020, 119: 2943-2952.

[16] MOLCHANOV P, TYREE S, KARRAS T, et al. Pruning convolutional neural networks for resource efficient inference[C]//International Conference on Learning Representations. 2017.

[17] HE Y, ZHANG X, SUN J. Channel pruning for accelerating very deep neural networks[C]//Proceedings of the IEEE International Conference on Computer Vision. 2017: 1389-1397.

[18] LUO J H, WU J, LIN W. ThiNet: A filter level pruning method for deep neural network compression[C]//Proceedings of the IEEE International Conference on Computer Vision. 2017: 5058-5066.

[19] LIU Z, LI J, SHEN Z, et al. Learning efficient convolutional networks through network slimming[C]//Proceedings of the IEEE International Conference on Computer Vision. 2017: 2736-2744.

[20] HE Y, KANG G, DONG X, et al. Soft filter pruning for accelerating deep convolutional neural networks[C]//Proceedings of the 27th International Joint Conference on Artificial Intelligence. 2018: 2234-2240.

[21] YANG T J, CHEN Y H, SZE V. Designing energy-efficient convolutional neural networks using energy-aware pruning[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2017: 5687-5695.

[22] YANG T J, HOWARD A, CHEN B, et al. NetAdapt: Platform-aware neural network adaptation for mobile applications[C]//Proceedings of the European Conference on Computer Vision. 2018: 285-300.

[23] HE Y, LIN J, LIU Z, et al. AMC: AutoML for model compression and acceleration on mobile devices[C]//Proceedings of the European Conference on Computer Vision. 2018: 784-800.

[24] LIU Z, MU H, ZHANG X, et al. MetaPruning: Meta learning for automatic neural network channel pruning[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019: 3296-3305.

[25] LI Y, XU Q, SHEN J, et al. Towards efficient deep spiking neural networks construction with spiking activity based pruning[C]//Proceedings of the 41st International Conference on Machine Learning. PMLR, 2024, 235: 29063-29073.

[26] HAN B, ZHAO F, ZENG Y, et al. Developmental Plasticity-Inspired Adaptive Pruning for Deep Spiking and Artificial Neural Networks[J]. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025, 47(1): 240-251. DOI: 10.1109/TPAMI.2024.3467268.

[27] HAN B, ZHAO F, PAN W, et al. Adaptive sparse structure development with pruning and regeneration for spiking neural networks[J]. Information Sciences, 2025, 689: 121481. DOI: 10.1016/j.ins.2024.121481.

[28] LI C, WU H, HUANG Y, et al. Deep spatio-temporal pruning for efficient spiking neural network based object detection in remote sensing images[J]. Remote Sensing, 2024, 16(17): 3200. DOI: 10.3390/rs16173200.
