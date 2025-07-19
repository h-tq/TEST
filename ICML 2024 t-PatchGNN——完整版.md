[TOC]

# [论文分享]ICML 2024 📄 不规则多元时间序列预测：一种可转换的分块图神经网络方法

| 🏷️ 标题  | Irregular Multivariate Time Series Forecasting:  A Transformable Patching Graph Neural Networks Approach |
| :-----: | :----------------------------------------------------------- |
| 👨‍💻 作者 | *Weijia Zhang¹, Chenlong Yin¹, Hao Liu¹², Xiaofang Zhou², Hui Xiong¹²* |
| 🏫 机构  | ¹香港科技大学（广州）, ²香港科技大学                         |
| 📧 邮箱  | liuh@ust.hk, xionghui@ust.hk                                 |
| 📄 论文  | https://openreview.net/pdf?id=UZlMXUGI6e                     |
| 💻 代码  | https://github.com/usail-hkust/t-PatchGNN                    |

------

### 摘要（Abstract）

不规则多变量时间序列（Irregular Multivariate Time Series, IMTS）的预测在医疗、人体生物力学、气候科学和天文学等众多领域具有关键意义。尽管现有研究尝试通过常微分方程（Ordinary Differential Equations, ODE）应对时间序列中的不规则性，但在建模异步IMTS之间的相关性方面仍缺乏深入探索。为弥补这一空白，本文提出了一种可变换补丁图神经网络（Transformable Patching Graph Neural Networks, T-PATCHGNN）方法，该方法将每个单变量不规则时间序列转化为一系列可变换补丁（transformable patches），每个补丁包含数量不等的观测值，但在时间上具有统一的分辨率。该方法在有效捕捉局部语义信息和建模时间序列间相关性的同时，避免了将IMTS对齐处理所导致的序列长度膨胀问题。在对齐补丁结果的基础上，本文进一步引入时间自适应图神经网络（time-adaptive graph neural networks），通过一系列学习得到的随时间变化的自适应图，建模时间序列间的动态相关性。我们在所构建的综合性IMTS预测基准上展示了T-PATCHGNN的显著优势，该基准覆盖了医疗、生物力学与气候科学领域的四个真实世界科学数据集，并包含17个来自相关研究领域的强基线模型。

> 本文深刻识别了IMTS领域中异步性与高缺失率带来的联合挑战，不再试图将时间轴强行对齐，而是引入“可变换补丁”作为中介单位，将局部时间段的信息整合为对齐友好的表示，极大地缓解了传统对齐方法带来的信息冗余与结构爆炸问题；在此基础上，时间自适应图神经网络的引入体现出作者对时变性与跨变量依赖建模的精准把握，适应异步数据间复杂动态关系，是时间序列图建模从静态图走向动态图的关键一步，整体方案在架构设计上极具原创性，兼具方法创新性与工程实用性，预示着异步时间序列建模范式的转变。

------

### 1. 引言（Introduction）

![figure1](./figure1.png)

尽管多元时间序列（Multivariate Time Series, MTS）的预测问题已被广泛研究，但大多数研究集中在采样规则且数据完整的MTS上（Lim & Zohren, 2021）。相比之下，不规则多元时间序列（Irregular Multivariate Time Series, IMTS）预测所面临的挑战则鲜有关注。IMTS以其不规则的采样间隔和大量缺失数据为特征，广泛存在于诸多领域中，如医疗健康、生物力学、气候科学、天文学和金融（Rubanova et al., 2019; De Brouwer et al., 2019; Yao et al., 2018; Vio et al., 2013; Engle & Russell, 1998; Zhang et al., 2021a）。准确地预测IMTS不仅为做出明智决策提供基础，也支撑了前瞻性规划等重要活动。

与规则MTS相比，IMTS的建模与分析更具挑战性，主要由于其固有的不规则性和时间序列间的异步性（Horn et al., 2020）。如图1(a)所示，给定一组历史IMTS观测值和预测查询，IMTS预测旨在准确预测与这些查询对应的值。尽管已有少量前沿研究致力于IMTS预测（Rubanova et al., 2019; De Brouwer et al., 2019; Biloš et al., 2021; Schirmer et al., 2022），但这些工作主要基于神经常微分方程（neural Ordinary Differential Equations, ODEs）（Chen et al., 2018）来处理时间序列内的不规则性，未能显式建模多个时间序列间的重要相关性。此外，由于数值积分过程，ODE求解器计算代价高昂，导致训练与推理效率低下（Biloš et al., 2021; Shukla & Marlin, 2020）。

准确地进行IMTS预测是一项非平凡的任务，面临三大挑战：

1. **时间序列内部依赖建模中的不规则性挑战。** 相邻观测值之间的时间间隔变化打破了时间序列数据的一致流动，使得经典的时间序列预测模型（Lim & Zohren, 2021）难以准确捕捉潜在的时间动态与依赖关系（Rubanova et al., 2019; Che et al., 2018）。

2. **时间序列间相关性建模中的异步性挑战。** 虽然不同变量的时间序列间通常存在显著相关性，但由于不规则采样或缺失数据，IMTS中的观测值常常在时间上错位。这种异步性使得特定时间点上的直接比较和相关性建模更加复杂，进而可能遮蔽或扭曲序列间的实际关系，从而带来建模上的重大挑战（Zhang et al., 2021b）。

3. **变量数量增长引发的序列长度爆炸问题。** 如图1(b)所示，为了便于IMTS建模，当前研究通常将IMTS转换为时间对齐的格式，将每个单变量不规则时间序列扩展为统一长度，对应IMTS观测中的所有唯一时间戳数量（Che et al., 2018）。然而，这种经典的预对齐表示会导致序列长度随变量数增长而指数级膨胀，从而在处理大量变量时引发计算与内存开销上的严重可扩展性问题。

为应对上述挑战，本文提出了一种可变换补丁图神经网络（Transformable Patching Graph Neural Networks, **T-PATCHGNN**）方法用于IMTS预测。T-PATCHGNN最初将每个单变量不规则时间序列转化为一系列**可变换补丁（transformable patches）**，这些补丁在观测数量上可变，但保持统一的时间范围分辨率。该IMTS处理流程具有三大优势：

1. 每个单变量不规则时间序列独立进行补丁划分，避免了IMTS的经典预对齐表示，从根本上解决了在大规模变量下序列长度爆炸的问题；

2. 将每个观测点置于语义更丰富的补丁中，有助于更好地捕捉不规则时间序列的局部语义（Nie et al., 2022）；

3. 经变换后，IMTS自然在补丁层面实现时间分辨率一致，从而缓解异步性问题，并顺畅促进后续的时间序列间相关性建模。

在此基础上，我们引入了一个**可变换时间感知卷积网络（transformable time-aware convolution network）**，将每个补丁编码为潜在嵌入，该嵌入随后作为输入token送入Transformer中进行时间序列内部依赖建模。同时，我们提出了**时间自适应图神经网络（time-adaptive graph neural networks）**用于建模IMTS中时间序列之间的动态相关性。为显式表示IMTS间的动态相关性，我们基于可学习的变量内在嵌入（inherent variable embedding）和动态补丁嵌入（dynamic patch embedding）构建了一系列时间变化的自适应图，这些图与补丁保持相同的时间分辨率。随后，将图神经网络应用于这些图上，从而在补丁层面建模IMTS之间的动态相关性。最后，采用多层感知机（Multi-Layer Perceptron, **MLP**）输出层，根据得到的IMTS潜在表示，生成与预测查询相对应的预测结果。

我们的主要贡献总结如下：

- 我们提出了一种新的**可变换补丁方法**，将IMTS中的每个单变量不规则时间序列转化为一系列长度可变但时间对齐的补丁。该方法巧妙绕过IMTS的经典预对齐表示，使得序列长度不再随着变量数的增加而爆炸性增长，并在此基础上顺畅捕捉局部语义并促进序列间的相关性建模。

- 基于变换补丁的表示结果，我们提出**时间自适应图神经网络**，用于建模IMTS中动态的时间序列间相关性。

- 我们构建了一个用于IMTS预测评估的基准测试，涵盖医疗健康、生物力学和气候科学等领域的四个公共科学IMTS数据集，并选取了来自IMTS预测、插值、分类和MTS预测等多个研究方向的17个最先进的基线模型进行公平比较。大量实验证明了T-PATCHGNN在预测精度上的显著优势。

------

### 2. 相关工作（Related Works）

#### 2.1. 非规则多变量时间序列预测（Irregular Multivariate Time Series Forecasting）

现有对非规则多变量时间序列（IMTS）的研究主要集中在分类任务上（Che 等，2018；Shukla & Marlin，2021；Zhang 等，2021b；2023a；Horn 等，2020；Shukla & Marlin，2018；Li 等，2023；Baytas 等，2017）。仅有少量前沿研究（Rubanova 等，2019；De Brouwer 等，2019；Biloš 等，2021；Schirmer 等，2022）尝试解决IMTS的预测问题。这些方法主要依赖**神经常微分方程（neural ODEs）**（Chen 等，2018），并专注于处理时间序列中的连续动态和非规则性。

例如，**Latent-ODE**（Rubanova 等，2019）使循环神经网络（RNN）具有由神经ODE定义的连续时间隐藏状态动态；**GRU-ODE-Bayes**（De Brouwer 等，2019）结合神经ODE构建连续时间GRU，并引入贝叶斯更新网络以处理稀疏观测数据；**CRU**（Schirmer 等，2022）则通过线性随机微分方程和连续-离散卡尔曼滤波器对隐藏状态进行演化，以应对不规则时间间隔。然而，由于数值积分开销较大，ODE求解器计算效率较低。为此，**Neural Flows**（Biloš 等，2021）通过神经网络建模ODE解的轨迹，从而降低神经ODE的数值计算成本。

尽管上述研究在处理非规则性方面做出了显著努力，但**如何在异步IMTS中有效建模时间序列之间的相互关联（inter-time series correlations）仍然是一个尚未充分探索的问题**。

#### 2.2. 非规则多变量时间序列表示（Irregular Multivariate Time Series Representation）

为实现时间对齐表示并促进后续建模，现有大多数工作采用**预对齐表示方法（pre-alignment representation）**（Che 等，2018；Shukla & Marlin，2021；Zhang 等，2021b；2023a；Baytas 等，2017；Rubanova 等，2019；De Brouwer 等，2019；Biloš 等，2021；Schirmer 等，2022）。该方法将IMTS中所有单变量序列扩展为长度一致的序列，其长度等于IMTS中所有唯一时间戳的数量，并通过掩码项表示缺失值（Che 等，2018）。

然而，随着变量数量增加，此类表示可能会面临**序列长度爆炸（sequence length explosion）**的问题，详见第3.2节，这在计算与内存开销方面引发严重的可扩展性挑战。

除预对齐方法外，**Horn 等（2020）**提出一种更具可扩展性的表示方法：将IMTS中的观测表示为一组元组（时间、值、变量指示符），并对这些元组进行汇总以用于IMTS分类任务。然而，此表示方式不适用于**需要对每个变量进行细粒度分析的预测任务**。

#### 2.3. 图神经网络在多变量时间序列中的应用（Graph Neural Networks for Multivariate Time Series）

**图神经网络（GNNs）**因其在建模变量间复杂关联方面的强大能力，已被广泛应用于多变量时间序列（MTS）建模中（Li 等，2018；Yu 等，2018；Wu 等，2019；2020b；Huang 等，2023；Yi 等，2023；Cao 等，2020；Liu 等，2022）。**DCRNN**（Li 等，2018）和**STGCN**（Yu 等，2018）将GNN应用于预定义图结构，但在某些领域中预定义图结构难以获取。

因此，部分研究（Wu 等，2019；2020b；Huang 等，2023；Yi 等，2023；Cao 等，2020）提出从数据中学习图结构，从而实现变量拓扑关系的自动建模。但在IMTS中，由于观测在时间上的显著非对齐性，**跨时间序列的关联建模（inter-series correlation modeling）面临挑战**。

**Raindrop**（Zhang 等，2021b）通过在任意变量有观测出现时，将其观测信息传播至所有时间戳，来应对这一挑战。但这种方法依赖于IMTS的预对齐操作，仍可能遭遇序列长度爆炸问题。

另一类相关工作则针对**存在缺失值的规则时间序列（regular MTS with missing data）**应用GNN进行建模（Cini 等，2022；Marisca 等，2022；Chen 等，2024）。这些方法通常需要对缺失的时间序列进行对齐，如前述的预对齐表示，并侧重于处理数据缺失问题。

然而，**本文工作强调绕过传统的预对齐表示方法，直接应对IMTS建模中非规则性与异步性双重挑战**。

------

### 3. 预备知识（Preliminary）

#### 3.1. 问题定义（Problem Definition）

**定义 1（非规则多变量时间序列）**  
一个非规则多变量时间序列（IMTS）可表示为：
$$
\mathcal{O} = \left\{ o_{1:L_n}^n \right\}_{n=1}^{N} 
= \left\{ \left[ \left( t_i^n, x_i^n \right) \right]_{i=1}^{L_n} \right\}_{n=1}^{N}
$$
其中，$N$ 是变量的数量，第 $n$ 个变量包含 $L_n$ 个观测值，第 $i$ 个观测由时间戳 $t_i^n$ 和对应的观测值 $x_i^n$ 构成。

**定义 2（预测查询）**  
一个预测查询记为 $q_j^n$，表示在第 $n$ 个变量上的第 $j$ 个查询，用于预测其在未来时间点 $q_j^n$ 的对应值。

**问题 1（非规则多变量时间序列预测）**  
给定历史观测的 IMTS：

$$
\mathcal{O} = \left\{ \left[ (t_i^n, x_i^n) \right]_{i=1}^{L_n} \right\}_{n=1}^{N}
$$

以及一组预测查询：

$$
\mathcal{Q} = \left\{ \left[ q_j^n \right]_{j=1}^{Q_n} \right\}_{n=1}^{N}
$$

我们的目标是准确预测与这些查询对应的值：

$$
\hat{\mathcal{X}} = \left\{ \left[ \hat{x}_j^n \right]_{j=1}^{Q_n} \right\}_{n=1}^{N}
$$

即学习一个预测模型 $\mathcal{F}$，满足：

$$
\mathcal{F}(\mathcal{O}, \mathcal{Q}) \rightarrow \hat{\mathcal{X}} \tag{1}
$$

#### 3.2. IMTS 的经典预对齐表示（Canonical Pre-Alignment Representation for IMTS）

为便于对不规则多变量时间序列（Irregular Multivariate Time Series, IMTS）进行建模，当前研究普遍采用一种预对齐表示方法（Che 等人，2018），已成为事实上的标准（Che et al., 2018; Shukla & Marlin, 2021; Zhang et al., 2021b; 2023a; Rubanova et al., 2019; De Brouwer et al., 2019; Biloš et al., 2021; Schirmer et al., 2022）。在该方法中，一个 IMTS $O$ 被表示为三个矩阵 $(T, X, M)$：

- $T = [t_l]_{l=1}^L = \bigcup_{n=1}^N [t_i^n]_{i=1}^{L_n} \in \mathbb{R}^L$ 表示所有观测值的时间戳去重后的时间轴，即 $O$ 中所有时间戳的联合集合；
- $X = \left[ [\tilde{x}_l^n]_{n=1}^N \right]_{l=1}^L \in \mathbb{R}^{L \times N}$ 表示与时间戳对应的变量取值，其中若第 $n$ 个变量在时间 $t_l$ 有观测值，则 $\tilde{x}_l^n = x_i^n$；否则填充为 ‘NA’；
- $M = \left[ [m_l^n]_{n=1}^N \right]_{l=1}^L \in \mathbb{R}^{L \times N}$ 是掩码矩阵，其中若 $\tilde{x}_l^n$ 在 $t_l$ 被观测到，则 $m_l^n = 1$，否则为 0。

可以观察到，该预对齐表示方法生成的序列长度 $L$ 取决于 $O$ 中所有变量观测时间戳的去重结果。令：

- $L_{\text{avg}} = \frac{1}{N} \sum_{n=1}^N L_n$ 表示每个变量平均拥有的观测数量，
- $L_{\text{max}} = \max_{n=1}^N [L_n]$ 表示最大观测长度，

则预对齐表示后的序列长度 $L$ 理论上满足：

$$
L_{\text{max}} \leq \left| \bigcup_{n=1}^N [t_i^n]_{i=1}^{L_n} \right| \leq N \times L_{\text{avg}} \tag{2}
$$

这意味着，当变量数量 $N$ 较大时，$L$ 可能呈指数式增长，从而在处理大规模变量时带来显著的可扩展性问题。

------

### 4. 方法（Methodology）

![figure2](./figure2.png)

T-PATCHGNN 的整体框架如图 2 所示。在接下来的各小节中，我们将依次介绍不规则时间序列的切片操作、序列内部与序列间的建模方法，以及 IMTS 的预测流程。

#### 4.1. 不规则时间序列切片（Irregular Time Series Patching）

在本节中，由于所有单变量不规则时间序列都将应用统一的切片操作，我们以下仅以第 $n$ 个变量为例进行说明，并为简洁起见省略上标 $n$。

##### 4.1.1. 可变换切片（Transformable Patching）

时间序列切片在多变量时间序列（MTS）预测任务中已被证明是有效的（Nie et al., 2022），其优势在于能够捕捉局部语义信息、降低计算和内存开销，以及建模更长范围的历史观测。标准的时间序列切片方法通常将规则采样时间序列划分为一系列子序列级别的固定长度切片，每个切片由若干连续观测点组成。

然而，在 IMTS 场景中，由于不同观测点之间时间间隔的不一致，这种方式会导致每个切片横跨不同的时间跨度。例如，一个由连续五个观测点构成的切片，在密集采样的情形下可能仅覆盖几分钟，而在稀疏采样的情形下则可能跨越数天。切片在时间分辨率上的高度可变性甚至可能进一步加剧 IMTS 建模中的不规则性和异步性问题。

为了解决这一问题，我们提出将每个单变量不规则时间序列 $o_{1:L}$ 划分为一系列可变换切片 $[o_{l_p:r_p}]_{p=1}^P$，其中每个切片由可变长度的连续观测点组成，$P$ 表示切片的总数，且满足 $l_1 = 1,\ r_P = L$。每个切片覆盖一个固定的时间窗口大小 $s$（例如 2 小时），从而在时间维度上实现一致的时间分辨率。两个相邻切片之间的划分可以选择重叠或不重叠。

基于此策略，最终生成的 IMTS 切片在时间跨度上达成了统一的对齐效果。由于每个单变量不规则时间序列是独立进行切片处理的，因此该方法可跳过对整个 IMTS 进行标准预对齐处理，从而避免了由于变量数量增加所导致的序列长度爆炸问题。

##### 4.1.2. 切片编码（Patch Encoding）

在将每个单变量不规则时间序列转换为一系列可变换切片后，我们需要对每个切片进行编码，以捕捉时间序列中的局部语义信息。

**连续时间嵌入（Continuous Time Embedding）**  
为了建模 IMTS 中的时间信息，我们首先采用连续时间嵌入（Continuous Time Embedding，Shukla & Marlin, 2021）来对观测时间进行编码。具体定义如下：

$$
\phi(t)[d] =
\begin{cases}
\omega_0 \cdot t + \alpha_0, & \text{if } d = 0 \\\\
\sin(\omega_d \cdot t + \alpha_d), & \text{if } 0 < d < D_t
\end{cases} \tag{3}
$$

其中，$\omega_d$ 和 $\alpha_d$ 是可学习参数，$D_t$ 是时间嵌入的维度。线性项用于建模非周期性演化模式，而正弦项则用于捕捉时间序列中的周期性结构，其中 $\omega_d$ 和 $\alpha_d$ 分别表示频率与相位。

通过拼接连续时间嵌入向量与原始观测值，可得切片中的每个观测表示为：

$$
z_{l_p:r_p} = [z_i]_{i=l_p}^{r_p} = [\phi(t_i) \,\|\, x_i]_{i=l_p}^{r_p} \tag{4}
$$

**可变换时间感知卷积（Transformable Time-aware Convolution）**  
由于每个可变换切片本质上是一个子不规则时间序列，我们引入 Transformable Time-aware Convolution Network（TTCN，Zhang et al., 2023b）对其内部语义进行建模。TTCN 利用元卷积核（meta-filter）动态生成时间感知卷积核，支持根据输入序列长度自适应调整卷积核大小。具体定义如下：

$$
f_d = \left[ \frac{\exp(F_d(z_i))}{\sum_{j=1}^{L_p} \exp(F_d(z_j))} \right]_{i=1}^{L_p} \in \mathbb{R}^{L_p \times D_{\text{in}}} \tag{5}
$$

其中，$L_p$ 是切片 $z_{l_p:r_p}$ 的长度，$f_d$ 表示第 $d$ 个特征图（feature map）对应的卷积核，$D_{\text{in}}$ 是输入维度，$F_d$ 是可学习神经网络实例化的元卷积函数。TTCN 通过对卷积核在时间维度上归一化，确保对不同长度序列的卷积结果具有一致的尺度。

基于公式 (5) 所导出的 $D-1$ 个卷积核，我们通过以下时间卷积获得潜在的切片嵌入 $h_c^p \in \mathbb{R}^{D-1}$：

$$
h_p^c = \left[ \sum_{i=1}^{L_p} f_d[i]^\top z_{l_p:r_p}[i] \right]_{d=1}^{D-1} \tag{6}
$$

TTCN 的优势在于其灵活适配可变长度序列，能够根据时间间隔生成专属参数化卷积核，并支持对任意长度的子序列进行建模，无需引入额外可学习卷积核参数。

考虑到在高时间分辨率或稀疏序列场景中，某些切片可能无观测值，我们进一步引入切片掩码项增强切片表示：

$$
h_p = [h_p^c \,\|\, m_p] \tag{7}
$$

其中，当切片存在观测值时 $m^p = 1$，否则为 0。最终我们获得所有切片的嵌入表示为：

$$
h_{1:P} = [h_p]_{p=1}^{P} \in \mathbb{R}^{P \times D}
$$

#### 4.2. 时序内部与跨序列建模（Intra- and Inter-Time Series Modeling）

本节将详细阐述如何通过对不规则时间序列应用可变换的 patching 操作，实现对时间序列内与时间序列间的建模。

##### 4.2.1. 使用 Transformer 建模序列补丁（Transformer to Model Sequential Patches）

在编码完补丁后，它们可以作为输入 token 被送入 Transformer（Vaswani et al., 2017），以建模非规则时间序列中的依赖关系。我们为补丁加入位置编码 $\mathrm{PE}_{1:P} \in \mathbb{R}^{P \times D}$，以指示其时间顺序：

$$
x^{\mathrm{tf},n}_{1:P} = h^n_{1:P} + \mathrm{PE}_{1:P}
$$

接着，通过以下方式对其施加多头注意力机制（multi-head attention），将其分别转换为查询矩阵、键矩阵和值矩阵：

$$
q^n_h = x^{\mathrm{tf},n}_{1:P} W^Q_h,\quad
k^n_h = x^{\mathrm{tf},n}_{1:P} W^K_h,\quad
v^n_h = x^{\mathrm{tf},n}_{1:P} W^V_h
$$

其中 $W^Q_h, W^K_h, W^V_h \in \mathbb{R}^{D \times (D/H)}$ 是可学习参数，$H$ 表示头的数量。随后，通过缩放点积注意力计算得到序列内部建模的输出：

$$
h^{\mathrm{tf},n}_{1:P} = \mathop{\Vert}_{h=1}^{H} \mathrm{Softmax}\left( \frac{q^n_h {k^n_h}^\top}{\sqrt{D/H}} \right) v^n_h \in \mathbb{R}^{P \times D} \tag{8}
$$

##### 4.2.2 时变自适应图结构学习（Time-Varying Adaptive Graph Structure Learning）

不同变量的时间序列往往具有显著相关性，来自其他变量的洞见对当前变量的预测可提供重要信息。例如，心率与血压之间具有高度相关性，其中一者的变化可反映另一者的趋势，揭示个体的心血管状态（Obrist et al., 1978）。然而，在不规则多变量时间序列（IMTS）中，不同变量的观测常常存在对齐问题，这为跨序列相关性建模带来挑战。现有工作（Zhang et al., 2021b）通过在任意变量出现观测时对所有时间点进行异步信息传播来处理这一问题，但这需要对 IMTS 进行预对齐，且可能引发序列长度爆炸的问题。

幸运的是，应用可变换补丁机制后，不同变量的序列被对齐到统一的时间分辨率，每个变量拥有相同数量的补丁，从而自然解决了 IMTS 中的异步问题。在此基础上，我们提出了一种时变自适应图神经网络（time-adaptive graph neural networks），以建模 IMTS 中变量间的动态相关性。

为了揭示 IMTS 中隐含的动态关联性，我们设计了一系列时变自适应图，这些图与补丁的时间分辨率保持一致。具体地，受研究（Wu et al., 2019; 2020b）启发，我们为所有变量维护两个可学习的嵌入字典 $E^s_1, E^s_2 \in \mathbb{R}^{N \times D_g}$，用于捕捉变量的内在特性。这些变量嵌入在训练过程中可更新，但在推理阶段保持静态，不随时间变化。

然而，变量间的相关性会随时间动态变化（Zhang et al., 2021b）。因此，我们引入补丁嵌入 $H^{\mathrm{tf}}_p = [h^{\mathrm{tf},n}_p]_{n=1}^N \in \mathbb{R}^{N \times D}$，以补丁级时间分辨率反映时间序列的时变语义，并通过门控加法操作（gated adding operation）将其与静态变量嵌入融合：

$$
E_{p,k} = E^s_k + g_{p,k} \ast E^d_{p,k}, \quad
E^d_{p,k} = H^{\mathrm{tf}}_p W^d_k, \quad
g_{p,k} = \mathrm{ReLU}\left(\tanh\left([H^{\mathrm{tf}}_p \Vert E^s_k] W^g_k \right)\right),\quad k \in \{1, 2\}
\tag{9}
$$

其中 $W^d_k \in \mathbb{R}^{D \times D_g}$，$W^g_k \in \mathbb{R}^{(D + D_g) \times 1}$ 是可学习参数。最终，我们利用以下方式构建每个补丁时间片上的时变图结构，用于显式表征 IMTS 中的动态相关性：

$$
A_p = \mathrm{Softmax}\left(\mathrm{ReLU}\left(E^p_1 (E^p_2)^\top \right)\right) \tag{10}
$$

##### 4.2.3. 使用图神经网络建模跨序列相关性（GNNs to Model Inter-Time Series Correlation）

基于前文学习到的图结构，我们引入图神经网络（Graph Neural Networks, GNNs）（Kipf & Welling, 2016；Wu et al., 2020a；Zhou et al., 2020）以补丁级时间分辨率建模动态的跨时间序列相关性：

$$
H_p = \mathrm{ReLU} \left( \sum_{m=0}^{M} (A_p)^m H^{\mathrm{tf}}_p W^{\mathrm{gnn}}_m \right) \in \mathbb{R}^{N \times D} \tag{11}
$$

其中，$M$ 表示 GNN 的层数，$W^{\mathrm{gnn}}_m \in \mathbb{R}^{D \times D}$ 是第 $m$ 层的可学习参数。

在实际应用中，我们可以灵活地堆叠多个 $K$ 个时序内部建模（intra-time series modeling）与跨序列建模（inter-time series modeling）模块，从而高效应对多种非规则多变量时间序列（IMTS）建模任务。

#### 4.3. 非规则多变量时间序列预测（IMTS Forecasting）

随后，利用展平层（flatten layer）及线性头（linear head）获得每个变量的最终潜在表示：

$$
H = \mathrm{Flatten}([H_p]_{p=1}^P) W^f \in \mathbb{R}^{N \times D_o} \tag{12}
$$

其中，$W^f \in \mathbb{R}^{(P D) \times D_o}$ 为可学习参数。

给定第 $n$ 个变量的表示 $H^n \in H$ 以及其对应的一组预测查询 $\{[q^n_j]_{j=1}^{Q_n}\}_{n=1}^{N}$，通过一个多层感知机（MLP）投影层生成对应预测结果：

$$
\hat{x}^n_j = \mathrm{MLP}([H^n \Vert \phi(q^n_j)]) \tag{13}
$$

模型通过最小化预测值与真实值之间的均方误差（MSE）损失进行训练：

$$
\mathcal{L} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{Q_n} \sum_{j=1}^{Q_n} \left( \hat{x}^n_j - x^n_j \right)^2 \tag{14}
$$

#### 4.4. 可扩展性分析（Analysis on Scalability）

由于所提出的可变换补丁机制（transformable patching）对每个单变量非规则时间序列进行独立处理，从而实现 IMTS 的时间对齐，因此模型所需处理的平均序列长度等于所有变量观测数量的平均值，即：

$$
L_{\mathrm{avg}} = \frac{1}{N} \sum_{n=1}^{N} L_n
$$

基于公式 (2) 中的分析，使用 transformable patching 后所需处理的平均序列长度 $L_{\mathrm{tp}}$ 是由传统预对齐表示（canonical pre-alignment representation）所产生的结果序列长度 $L_{\mathrm{cpr}}$ 的下界：

$$
L_{\mathrm{tp}} = L_{\mathrm{avg}} \leq L_{\mathrm{max}} \leq L_{\mathrm{cpr}} \leq N \times L_{\mathrm{avg}} \tag{15}
$$

该机制有效防止了 $L_{\mathrm{tp}}$ 随变量数量 $N$ 的增加而爆炸性增长，从而提升了模型在变量维度扩展时的可扩展性。我们在第 5.4 节和附录 A.2 中提供了实证结果以进一步分析模型的可扩展性。

------

### 5. 实验（Experiments）

#### 5.1. 实验设置（Experimental Setup）

##### 5.1.1. 数据集（Datasets）

我们在四个涵盖不同学科领域的数据集上对模型的IMTS预测性能进行全面评估，这些数据集包括：**PhysioNet、MIMIC、Human Activity 和 USHCN**，分别涉及医疗健康、生物力学和气候科学等领域。所有数据集中样本均随机划分为训练集、验证集和测试集，比例分别为60%、20%和20%。有关这些数据集的详细信息，请参阅附录A.5节。

##### 5.1.2. 实现细节（Implementation Details）

所有实验在一台Linux服务器上进行，服务器配备20核Intel(R) Xeon(R) Platinum 8255C CPU @ 2.50GHz和NVIDIA Tesla V100 GPU。为确保公平对比，我们对所有比较模型统一设定隐藏维度：**PhysioNet 和 MIMIC 为64维**，**Human Activity 和 USHCN 为32维**。**批大小（batch size）**为USHCN设置为192，其他数据集为32。所有模型均使用 **Adam优化器**进行训练，并在验证集损失连续10轮不下降时启用**早停（early stopping）**策略。为减小随机性，我们对每组实验使用**五个不同随机种子**重复实验，并报告其**平均值与标准差**。

对于 T-PATCHGNN 的详细设置，补丁窗口大小 $s$ 分别设为：

- **8小时**（PhysioNet 和 MIMIC），  
- **300毫秒**（Human Activity），  
- **2个月**（USHCN）。

为减少生成的补丁数量，我们不进行窗口重叠（non-overlapping segmentation），并将滑动步长设为窗口大小。时间嵌入维度 $D_t$ 和变量嵌入维度 $D_g$ 均设为 **10**。Transformer中的**注意力头数** $H$、图神经网络层数 $M$、以及**模块块数** $K$ 均设为 **1**。我们采用**三层MLP**来实例化TTCN中的元滤波器（meta-filters）以及输出投影层。模型训练使用的学习率设为 **0.001**。

##### 5.1.3. （评估指标）Evaluation Metrics

当前IMTS预测研究主要采用 **均方误差（Mean Square Error, MSE）** 作为性能评估指标，但该指标对异常值较为敏感，且解释性较差（Chai & Draxler, 2014）。为更全面地评估模型性能，我们同时引入 **平均绝对误差（Mean Absolute Error, MAE）**，这是传统时间序列预测中广泛采用的评估指标（Lim & Zohren, 2021；Fan et al., 2023）。两项指标的数学定义如下：

- **均方误差（MSE）**：

$$
\text{MSE} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{Q_n} \sum_{j=1}^{Q_n} \left( \hat{x}_j^n - x_j^n \right)^2
$$

- **平均绝对误差（MAE）**：

$$
\text{MAE} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{Q_n} \sum_{j=1}^{Q_n} \left| \hat{x}_j^n - x_j^n \right|
$$

##### 5.1.4. 基准模型（Baselines）

为全面评估这一尚未充分探索的非规则多变量时间序列（IMTS）预测任务，我们引入了 17 个相关的基准模型进行公平比较，涵盖以下几大类的当前最先进方法（SOTA）：

1. **多变量时间序列预测（MTS Forecasting）**：  
   - DLinear（Zeng et al., 2023）  
   - TimesNet（Wu et al., 2022）  
   - PatchTST（Nie et al., 2022）  
   - Crossformer（Zhang & Yan, 2022）  
   - GraphWaveNet（Wu et al., 2019）  
   - MTGNN（Wu et al., 2020b）  
   - StemGNN（Cao et al., 2020）  
   - CrossGNN（Huang et al., 2023）  
   - FourierGNN（Yi et al., 2023）

2. **IMTS 分类任务（IMTS Classification）**：  
   - GRU-D（Che et al., 2018）  
   - SeFT（Horn et al., 2020）  
   - RainDrop（Zhang et al., 2021b）  
   - Warpformer（Zhang et al., 2023a）

3. **IMTS 插值任务（IMTS Interpolation）**：  
   - mTAND（Shukla & Marlin, 2021）

4. **IMTS 预测任务（IMTS Forecasting）**：  
   - Latent ODEs（Rubanova et al., 2019）  
   - CRU（Schirmer et al., 2022）  
   - Neural Flows（Biloš et al., 2021）

上述基准模型的详细介绍可见附录 A.6 节。

#### 5.2. 主要实验结果（Main Results）

![table1](./table1.png)

表 1 报告了各模型在四个数据集上的预测性能，评价指标为均方误差（MSE）与平均绝对误差（MAE）。可以观察到，**T-PATCHGNN** 在所有数据集上均获得了最优且一致的性能，甚至相较于第二优的基线模型，其性能提升超过 10%。此外，我们还观察到，现有的多变量时间序列预测模型（包括基于 patch 的模型与基于图神经网络的模型）在 IMTS 预测任务中并未表现出一致的竞争力。这表明，直接将这些技术应用于 IMTS 并不能有效应对其具有挑战性的**序列内建模**与**序列间建模**问题。

更进一步地，现有的 IMTS 预测模型整体性能也未达到令人满意的水平，可能原因在于它们未能充分建模变量间的动态相关性，从而未能有效提升预测表现。我们还对各模型在不同预测窗口长度（长窗口与短窗口）下的表现进行了评估，相关结果详见附录 A.1 节。

#### 5.3. 消融实验（Ablation Study）

![table2](./table2.png)

我们在四个数据集上评估了 T-PATCHGNN 及其若干变体的性能表现。（1）**Complete** 表示完整模型，未做任何消融；（2）**w/o Patch** 去除了 transformable patching，改为采用 canonical pre-alignment 表示方式；（3）**rp Patch** 使用标准时间序列切片方法（Nie et al., 2022）替代 transformable patching；（4）**w/o VE** 在构建自适应图结构时，移除了第（9）式中的变量嵌入（Variable Embedding）；（5）**w/o PE** 在构建图结构时移除了切片嵌入（Patch Embedding）；（6）**w/o Transformer** 去除了模型中的 Transformer 模块。

表 2 显示了各模型变体的消融实验结果。可以观察到，移除任何组件都会导致性能下降，进一步验证了各模块对整体模型性能的贡献。其中，**w/o Patch** 在所有数据集上都引起了显著的性能退化，这说明将非规则时间序列进行切片处理，确实有助于后续的时间序列内部和序列间建模。然而，**rp Patch** 在部分数据集（如 PhysioNet）中甚至表现劣于 **w/o Patch**，验证了我们关于“标准切片方式难以适配变动的时间分辨率，反而可能加剧非规则与异步性”的观点。比较 **w/o VE** 和 **w/o PE** 的结果可以发现，对于生理信号预测任务（如 PhysioNet 和 MIMIC），变量的固有特征（inherent characteristics）比其动态模式（dynamic patterns）更为关键，因为这些信号之间存在显著语义差异，若无法明确识别变量来源则难以有效刻画其间关系。而在人体动作和气候预测任务中，动态切片嵌入显得尤为重要，说明这些场景下变量之间的关系会随时间发生动态变化。例如冬季气温下降往往伴随着积雪增加，而这一相关性在夏季并不适用。

#### 5.4. 可扩展性与效率分析（Scalability and Efficiency Analysis）

![table3](./table3.png)

表 3 展示了在四个数据集上采用标准预对齐表示（canonical pre-alignment representation）后所面临的序列长度膨胀问题。显然，在平均情况下，序列长度相较原始观测点数量可扩展至 **20 倍以上**，尤其是在变量数量较多时更为严重。在极端情况下，序列长度甚至可能按照变量数量呈指数级增长（通过最大放大倍数可见），这对模型的可扩展性构成了严重挑战。然而，我们提出的可变换 patching 方法通过直接处理原始观测序列而无需预对齐，**有效规避了该问题**。

![figure3](./figure3.png)

为了进一步研究可变换 patching 在模型效率上的优势，我们在图 3 中展示了 MIMIC 数据集上每轮训练的平均耗时及每个 IMTS 实例的平均推理耗时。从中可以看出，**T-PATCHGNN 在训练与推理阶段的效率均优于所有使用标准预对齐表示的模型**。更进一步地，与当前主流的基于 ODE 的 IMTS 预测模型相比，T-PATCHGNN 在训练速度上快至少 **65 倍**，在推理速度上快至少 **15 倍**。关于在变量数量增加情况下模型可扩展性的更多实证测试，详见附录 A.2。

#### 5.5. Patch 大小的影响（Effect of Patch Size）

![figure4](./figure4.png)

图 4 展示了不同 patch 窗口大小对各个数据集性能的影响。我们观察到 patch 大小对模型性能的影响因数据领域而异。具体而言，在 PhysioNet 与 MIMIC 数据集上，当 patch 大小时长达到 **8 小时**时，模型性能达到峰值，此前 patch 越小性能差异不大。这一现象可归因于生理信号的稀疏特性——小于 4 小时的时间窗口往往无法囊括足够的观测值以有效捕捉局部模式。而当 patch 进一步增大，模型性能则开始下降。这是因为过大的 patch 会降低时间粒度，从而削弱了模型对时间序列内部与变量间动态关系的建模能力。

在 Human Activity 与 USHCN 数据集上，较小的 patch 大小更为合适。由于这两个领域的 IMTS 通常具有强动态变化特性，较小的 patch 能支持更细粒度的动态建模。另一方面，从任务角度来看，**最佳 patch 大小应结合预测窗口与观测窗口的长度综合选择**。例如，在长预测窗口或长观测期的情境中（如 PhysioNet 与 MIMIC），较大的 patch 有助于捕捉局部趋势语义及建模长期依赖；而在短时预测任务中（如 Human Activity 与 USHCN），较小 patch 更适合用于提升时间分辨率、刻画快速变化。

关于更多超参数的敏感性分析，详见附录 A.3。

------

### 6. 结论（Conclusion）

本文提出了一种用于不规则多变量时间序列（IMTS）预测任务的图神经网络方法——可变换 patch 图神经网络（T-PATCHGNN）。T-PATCHGNN 通过将每一条单变量不规则时间序列转化为一系列 observation 数目可变但时间尺度统一的可变换 patch，实现了异步 IMTS 之间的对齐。这一转换不仅有效提取了局部语义信息，也无缝支持了序列内部与序列之间的建模，避免了传统预对齐操作中由于变量数量增长而导致的序列长度爆炸问题。

在此基础上，本文进一步设计了时变自适应图神经网络（time-adaptive graph neural networks），用于基于一系列动态学习到的时变图，刻画序列间的动态相关关系。我们在构建的 IMTS 预测基准任务中验证了 T-PATCHGNN 的显著性能优势。

### 致谢（Acknowledgements）

本研究部分得到了以下项目的支持：中国国家重点研发计划（No.2023YFF0725001）、国家自然科学基金（No.92370204，No.62102110）、广州市-香港科技大学（广州）联合资助计划（No.2023A03J0008）、广州市教育局、广东省科技厅以及 CCF-百度开放基金。

### 影响声明（Impact Statement）

本文旨在推动时间序列分析方法的发展，并拓展其在多个科学领域中的应用。尽管本研究可能带来诸多社会影响，但我们认为无需在此特别强调某一具体后果。

### 参考文献（References）

（本节未列出具体文献，详见正文中引用。）

