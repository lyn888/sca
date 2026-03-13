脉冲神经网络（SNNs）被称为第三代人工神经网络[1]，以其在生物可解释性和低功耗方面的优势而闻名。受生物信息处理机制的启发，与传统的人工神经网络（ANNs）不同，SNNs的神经元通过发射二元尖峰与突触后神经元通信来产生稀疏和离散的事件[2]，利用脉冲序列进行特征编码、传输和处理。部署在神经形态芯片上[3]，SNN利用事件驱动和异步计算来显著降低计算复杂性并最大限度地减少不必要的开销[4]。与ANN相比，SNN在神经形态硬件上部署时具有事件驱动的特性，因此具有节能优势[5][6][7]，近年来在神经形态计算中越来越受欢迎[8]。

最近，人们对开发能够处理具有挑战性任务的深度SNN架构越来越感兴趣，例如对象识别[9][10]、检测[11]、文本分类[12]和机器人控制[13]。这些架构，如SpikingVGG[14][15]和SpikingResNet[16][17][18]，采用具有多层的大规模SNN来实现卓越的任务性能[19]。脉冲神经网络（Spiking Neural Networks, SNNs）凭借其对生物神经元机制的高度模拟，在低功耗和时序数据处理方面展现了巨大潜力。通过利用脉冲信号进行信息传递，SNN能够以更低的计算成本处理复杂的动态数据，与传统人工神经网络（ANN）相比，具有显著的能效优势和时间信息捕捉能力。这种特性使SNN在计算机视觉、自然语言处理、智能机器人以及嵌入式系统等领域成为了备受关注的研究热点。随着软硬件技术的快速迭代，SNN正在为智能计算提供更加高效和灵活的解决方案，推动人工智能技术迈向更广阔的应用场景。

现代深度神经网络，具有庞大的模型大小，需要大量的计算和存储资源。为了在资源受限的环境中部署现代模型并加快推理时间，研究人员越来越多地探索修剪技术作为神经网络压缩的一个流行研究方向。在过去的几年里，深度神经网络（DNN）在各个领域和应用中取得了显著进展，如计算机视觉、自然语言处理、音频信号处理和跨模态应用。

根据剪枝的目标不同，可以将剪枝方法分为结构化剪枝、非结构化剪枝。非结构化剪枝主要关注模型中的参数，通过识别和删除对模型性能影响较小的参数来减小模型的规模。这种方法通常基于参数的重要性指标进行选择,例如参数的绝对值大小或其对损失函数的贡献。剪枝后,被删除的参数将不再参与模型的计算,从而减小了计算资源和存储需求。Han等[20]为了解决传统网络的不足,提出了一种在尽量保持神经网络性能的情况下只修剪网络中不重要连接的方法,该方法首次给出了三阶段剪枝流程，即训练修剪微调。Mariet等[21]提出通过DPP(DeterminantalPointProcess)方法来模拟神经元可以更有原则、更灵活地进行神经元重要性的评估。该方法帮助网络结构有效地自动调整而不影响性能,不需要再微调模型，从而节省了时间成本。Srinivas等[22]定义了另一种神经元冗余,并且提出了一种不依赖训练数据直接剪枝的方法。然而非结构化剪枝往往会将滤波器中的元素置0，从而指定了一个固定的子空间来约束滤波器,特别是在训练前或者训练中进行剪枝往往会产生很大的误差。Wimmer等[23]提出了一种空间剪枝，空间剪枝使用在底层自适应的滤波器基础的线性组合在动态空间表示滤波器,并将未剪枝的参数和滤波器基础进行联合训练。结构化修剪可以删除整个过滤器、通道甚至层。相比于非结构化剪枝，结构化剪枝后的结构更加规整，因此在软硬件层面可以获得更有效的加速，但是大范围的剪枝结构会带来精度大幅度下降的问题。Zhuang等[24]介绍了network slimming方法,通过给网络模型中的每个通道加上一个新的参数并将其作为缩放因子来进行结构化剪枝。结构冗余减少（SRR）[25]通过寻找最冗余的层而不是所有层中排名最低的过滤器来利用结构冗余。在最冗余的层中，可以应用过滤器规范来修剪最不重要的过滤器，然后重新建立层的图，并重新评估层的冗余度。协作通道修剪（CCP）[26]仅使用预训练模型的一阶导数来近似Hessian矩阵。一阶信息可以从反向传播中检索，不需要额外的存储。Liu等人[27]提出动态稀疏图（DSG），在每次迭代时动态地使用构建的稀疏图激活少量关键神经元。为了防止BN层破坏稀疏性，Liu等人引入了双掩模选择，在BN层前后使用相同的选择掩模。

根据修剪的时间不同，可以分为训练前修剪、训练期间修剪、训练后修剪。训练前修剪的主要动机是消除预训练的成本。Liu等人[28]证明，网络大小和适当的逐层修剪比率是从头开始训练随机修剪网络以匹配密集模型性能的两个关键因素。Bai等人[29]提出了双彩票假说，其中子网和权重在初始化时都是随机选择的，并提出了随机稀疏网络变换，该变换修复了稀疏架构，但逐渐训练了剩余的权重。训练过程中的修剪通常将随机初始化的密集网络作为输入模型，并在训练过程中通过更新权重和权重掩码来联合训练和修剪神经网络。Li等人[30]提出了因子化卷积滤波器，该滤波器为每个滤波器引入了一个二进制标量，并提出了一种具有交替方向乘子法的反向传播算法，以在训练过程中联合训练权重和标量。Cho等人[31]在没有额外可训练参数的情况下为权重生成软修剪掩码，以实现CNN和Transformer的可微分修剪。训练后修剪是比较受欢迎的一种类型，因为人们普遍认为对密集网络进行预训练是获得高效子网络所必需的。Diffenderfer和Kailkhura[32]提出了一种更强的多奖LTH，该LTH声称中奖彩票对极端形式的量化具有鲁棒性。基于此，他们首次引入了多奖票（MPTs）算法，在二元神经网络上寻找MPTs。Shi等人[33]介绍了统一渐进修剪，它通过在搜索阶段的每次迭代中利用累积的可训练掩模梯度来修剪大型多模态模型。运行时动态剪枝作为一种新兴的优化策略，结合了动态调整和剪枝技术的优势，为深度学习模型的高效部署提供了新的解决方案。Elkerdawy等人[34]将动态模型修剪视为一个自监督的二元分类问题。Meng等人[35]提出了对比双门控，这是另一种使用对比学习的自监督动态修剪方法。Tuli和Jha[36]介绍了DynaTran，它根据输入矩阵的大小在运行时修剪激活，以提高transformer推理吞吐量。

[1] Maass, W. Networks of spiking neurons: the third generation of neural network models. Neural networks, 10(9):1659–1671, 1997.

[2] Olga Krestinskaya, Alex Pappachen James, and Leon Ong Chua. Neuromemristive circuits for edge computing: A review. IEEE Transactions on Neural Networks and Learning Systems, 31(1):4–23,2019.

[3] Ma, D., Jin, X., Sun, S., Li, Y., Wu, X., Hu, Y., Yang, F.,Tang, H., Zhu, X., Lin, P., et al. Darwin3: A large-scale neuromorphic chip with a novel isa and on-chip learning.arXiv preprint arXiv:2312.17582, 2023.

[4] Davies, M., Srinivasa, N., Lin, T.-H., Chinya, G., Cao, Y.,Choday, S. H., Dimou, G., Joshi, P., Imam, N., Jain, S.,et al. Loihi: A neuromorphic manycore processor with on-chip learning. Ieee Micro, 38(1):82–99, 2018.

[5] Steve B Furber, Francesco Galluppi, Steve Temple, and Luis A Plana. The spiNNaker project.Proceedings of the IEEE, 102(5):652–665, 2014.

[6] Paul A Merolla, John V Arthur, Rodrigo Alvarez-Icaza, Andrew S Cassidy, Jun Sawada, Filipp Akopyan, Bryan L Jackson, Nabil Imam, Chen Guo, Yutaka Nakamura, et al. A million spikingneuron integrated circuit with a scalable communication network and interface. Science, 345(6197):668–673, 2014.

[7] Jing Pei, Lei Deng, Sen Song, Mingguo Zhao, Youhui Zhang, Shuang Wu, Guanrui Wang, ZheZou, Zhenzhi Wu, Wei He, et al. Towards artificial general intelligence with hybrid tianjic chip architecture. Nature, 572(7767):106–111, 2019.

[8] Catherine D Schuman, Thomas E Potok, Robert M Patton, J Douglas Birdwell, Mark E Dean, Garrett S Rose, and James S Plank. A survey of neuromorphic computing and neural networks in hardware. arXiv preprint arXiv:1705.06963, 2017.

[9] Youngeun Kim and Priyadarshini Panda. Revisiting batch normalization for training low-latency deep spiking neural networks from scratch. Frontiers in Neuroscience, 15:773954, 2021.

[10] Yaoyu Zhu, Zhaofei Yu, Wei Fang, Xiaodong Xie, Tiejun Huang, and Timothee Masquelier. Train- ´ing spiking neural networks with event-driven backpropagation. In Advances in Neural Information Processing Systems, pp. 30528–30541, 2022.

[11] Seijoon Kim, Seongsik Park, Byunggook Na, and Sungroh Yoon. Spiking-YOLO: spiking neural network for energy-efficient object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 11270–11277, 2020.

[12] Changze Lv, Jianhan Xu, and Xiaoqing Zheng. Spiking convolutional neural networks for text classification. In Proceedings of the International Conference on Learning Representations, pp.1–17, 2023.

[13] Guangzhi Tang, Neelesh Kumar, Raymond Yoo, and Konstantinos Michmizos. Deep reinforcement learning with population-coded spiking neural network for continuous control. In Proceedings of the Conference on Robot Learning, pp. 2016–2029, 2021.

[14] Hanle Zheng, Yujie Wu, Lei Deng, Yifan Hu, and Guoqi Li. Going deeper with directly-trained larger spiking neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence,pp. 11062–11070, 2021.

[15] Chankyu Lee, Syed Shakib Sarwar, Priyadarshini Panda, Gopalakrishnan Srinivasan, and Kaushik Roy. Enabling spike-based backpropagation for training deep neural network architectures. Frontiers in Neuroscience, 14:119, 2020.

[16] Yangfan Hu, Huajin Tang, and Gang Pan. Spiking deep residual networks. IEEE Transactions on Neural Networks and Learning Systems, 34(8):5200–5205, 2021.

[17] Hanle Zheng, Yujie Wu, Lei Deng, Yifan Hu, and Guoqi Li. Going deeper with directly-trained larger spiking neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence,pp. 11062–11070, 2021.

[18] Wei Fang, Zhaofei Yu, Yanqi Chen, Tiejun Huang, Timothee Masquelier, and Yonghong Tian. Deep ´residual learning in spiking neural networks. In Advances in Neural Information Processing Systems, pp. 21056–21069, 2021a.

[19] Chaoteng Duan, Jianhao Ding, Shiyan Chen, Zhaofei Yu, and Tiejun Huang. Temporal effective batch normalization in spiking neural networks. In Advances in Neural Information Processing Systems, pp. 34377–34390, 2022.

[20] HAN S,POOL J,TRAN J,et al. Learning both weights andconnections for efficient neural networks[C]// Proceedings ofthe 28th International Conference on Neural Information Processing Systems-Volume l.2015:1135-1143.

[21]MARIET Z,SRA S, Diversity networks : Neural network compression using determinantal point processes[J]. arXiv:1511.05077,2015.

[22]SRINIVAS S,BABUA R V.Data-free parameter pruning for deep neural networks[J].arXiv:1507.06149,2015.

[23]WIMMER P,MEHNERT J,CONDURACHE A.Interspace pruning:Using adaptive filter representations to improve training of sparse cnns[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.2022:12527-12537.

[24] ZHUANG L,LIJ,SHEN Z,et al. Learning Efficient Convolutional Networks through Network Slimming[C]//2017 IEEEInternational Conference on Computer Vision ( ICCV). IEEE2017.

[25] Z. Wang, C. Li, and X. Wang, “Convolutional neural network pruning with structural redundancy reduction,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2021, pp. 14908–14917.

[26] H. Peng, J.Wu, S. Chen, and J. Huang, “Collaborative channel pruning for deep networks,” in Proc. Int. Conf. Mach. Learn., 2019, pp. 5113–5122.

[27] L. Liu et al., “Dynamic sparse graph for efficient deep learning,” in Proc.Int. Conf. Learn. Representations, 2019.

[28] S. Liu et al., “The unreasonable effectiveness of random pruning: Return of the most naive baseline for sparse training,” in Proc. Int. Conf. Learn.Representations, 2022.

[29] Y. Bai, H. Wang, Z. Tao, K. Li, and Y. Fu, “Dual lottery ticket hypothesis,”in Proc. Int. Conf. Learn. Representations, 2022.

[30] T. Li, B. Wu, Y. Yang, Y. Fan, Y. Zhang, and W. Liu, “Compressing convolutional neural networks via factorized convolutional filters,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2019, pp. 3977–3986.

[31] M. Cho, S. Adya, and D. Naik, “PDP: Parameter-free differentiable pruning is all you need,” in Proc. Int. Conf. Neural Inf. Process. Syst.,2023, Art. no. 1986.

[32] J. Diffenderfer and B. Kailkhura, “Multi-prize lottery ticket hypothesis: Finding a accurate binary neural networks by pruning a randomly weighted network,” in Proc. Int. Conf. Learn. Representations, 2021.

[33] D. Shi, C. Tao, Y. Jin, Z. Yang, C. Yuan, and J. Wang, “UPop: Unified and progressive pruning for compressing vision-language transformers,”in Proc. Int. Conf. Mach. Learn., 2023, pp. 31292–31311.

[34] S. Elkerdawy, M. Elhoushi, H. Zhang, and N. Ray, “Fire together wire together: A dynamic pruning approach with self-supervised mask prediction,” in Proc. IEEE Conf. Comput. Vis. Pattern Recognit., 2022,pp. 12454–12463.

[35] J.Meng, L. Yang, J. Shin, D. Fan, and J. Sun Seo, “Contrastive dual gating:Learning sparse features with contrastive learning,” in Proc. IEEE Conf.Comput. Vis. Pattern Recognit., 2022, pp. 12247–12255.

[36] S. Tuli and N. K. Jha, “AccelTran: A sparsity-aware accelerator for dynamic inference with transformers,” IEEE Trans. Comput. Aided Des.Integr. Circuits Syst., vol. 42, no. 11, pp. 4038–4051, Nov. 2023.