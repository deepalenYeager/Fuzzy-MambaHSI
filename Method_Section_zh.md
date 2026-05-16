# 3. 方法 (Method)

本节系统阐述本文提出的 **Fuzzy-MambaHSI** ——一种面向高光谱图像 (HSI) 分类的
深度神经-模糊状态空间模型。我们首先在 §3.1 给出问题表述与整体架构;然后在
§3.2–§3.5 自底向上依次刻画四个核心组件:**模糊光谱分组与逐组嵌入**、
**模糊空间 Mamba 块 (FSpaMB)**、**模糊光谱 Mamba 块 (FSpeMB)** 以及
**TSK 空间-光谱融合模块 (TSSFM)**;最后在 §3.6 描述端到端训练策略,
在 §3.7 分析模型的计算复杂度与可解释性诊断协议。

---

## 3.1 概述 (Overview)

### 3.1.1 问题表述

给定一幅高光谱图像 $\mathbf{I}\in\mathbb{R}^{H\times W\times C}$,
其中 $H,W$ 为空间尺寸、$C$ 为原始光谱波段数。设其逐像素地面真值标注为
$\mathbf{Y}\in\{1,\dots,K,\,-1\}^{H\times W}$ ($-1$ 表示忽略类别)。
目标是学习一个映射 $f_\theta:\mathbb{R}^{H\times W\times C}\to
\mathbb{R}^{H\times W\times K}$,在仅给定少量带标注像素的弱监督下
输出全图的类别概率分布。

HSI 分类的两个根本难点驱动本文的设计:

* **光谱模糊性 (Spectral ambiguity).** 不同地物在反射光谱上往往存在显著
  重叠,且单一像元常常对应多种地物的混合(混合像元),使得"硬"类别划分
  在物理上不可恰当。换言之,每个像元相对类别的隶属本身就是**程度问题**
  ——这正是 Zadeh 意义下的**模糊性**。
* **线性代价下的长程空间依赖.** 像元级分类要求每个像元在表征上携带远距
  上下文。基于自注意力的 Transformer 复杂度为 $\mathcal{O}(L^2)$,
  CNN 受限于局部感受野,而选择性状态空间模型 (Selective SSM,
  Mamba) 在线性复杂度 $\mathcal{O}(L)$ 下实现长程建模——但其状态转移完全
  依赖于数据驱动的隐式投影,缺少对光谱不确定性的显式刻画。

我们的核心思想是将 **TSK (Takagi-Sugeno-Kang) 式模糊推理**作为可微的内嵌
模块直接注入 Mamba 主干,使主干在保持线性复杂度的同时获得对光谱模糊性
的归纳偏置。

### 3.1.2 整体架构

如图 1 所示,Fuzzy-MambaHSI 由四个阶段构成:

```
输入 HSI  I ∈ R^{H×W×C}
       │
       ▼
┌──────────────────────────────────────────┐
│  ① 模糊光谱分组 (Fuzzy Spectral Grouping)   │   ← 作用于原始物理波段轴 λ
│      G 个高斯软分组 μ(λ)                     │
│      S_g ∈ R^{H×W×C},  g = 1..G            │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  ② 逐组嵌入 (Per-Group Embedding)           │
│      1×1 Conv → GN → SiLU                  │
│      E ∈ R^{H×W×G×M},  D = G·M             │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│  ③ 编码器 (N_enc 个 Fuzzy Encoder Block)    │
│   ┌──────────────┐  ┌──────────────┐       │
│   │  FSpaMB      │  │  FSpeMB      │       │
│   │ (空间 Mamba+ │  │ (光谱 Mamba+ │       │
│   │  模糊 Δ-调制)│  │  TSK 解模糊) │       │
│   └─────┬────────┘  └─────┬────────┘       │
│         │  H_spa            │  H_spe        │
│         └────────┬──────────┘                │
│                  ▼                            │
│           ┌──────────────┐                   │
│           │   TSSFM      │                   │
│           │ (TSK 像素级  │                   │
│           │  空-谱融合)  │                   │
│           └──────┬───────┘                   │
└──────────────────┼──────────────────────────┘
                   ▼
┌──────────────────────┐
│ ④ 分割头 (Seg Head)   │   Conv1×1 → logits
│  l ∈ R^{H×W×K}       │
└──────────────────────┘
```
**图 1.** Fuzzy-MambaHSI 整体结构。粗体框出的三个模块即模糊推理与 Mamba
主干的耦合点。

模糊推理在主干中被部署于三个**功能正交**的位置——它们分别对应三种来源
不同的不确定性:

1. **特征组成层 (§3.2 + §3.4):** 模糊光谱分组决定**"哪些波段被组合在一起"**
   ——即特征的**组成**;
2. **状态动力学层 (§3.3):** 模糊 Δ-调制决定**"上下文如何沿空间序列流动"**
   ——即状态空间的**动力学**;
3. **特征融合层 (§3.5):** TSK 像素级融合决定**"空间与光谱表示如何被混合"**
   ——即表征的**融合**。

这种分层模糊设计构成一个端到端可微的三级模糊推理链。下文将逐一形式化每
一层。

---

## 3.2 模糊光谱分组与逐组嵌入

### 3.2.1 动机

直接以 1×1 卷积将 $C$ 个原始波段一次性混合为 $D$ 维通道会丢失"光谱局部
性"——同一吸收特征通常仅由相邻若干波段共同决定。我们因此在嵌入**之前**
引入一组定义在**原始波段索引轴 $\lambda\in\{1,\dots,C\}$ 上**、参数可学
的高斯隶属函数,以软方式将波段轴划分为 $G$ 个语义上有意义的光谱子带组。
这一改造对应代码中的 `model/fuzzy_modules.py:FuzzySpectralGrouping`。

### 3.2.2 公式化

为每一组 $g\in\{1,\dots,G\}$ 引入可学习的中心 $c_g\in[1,C]$ 与带宽
$\sigma_g\in\mathbb{R}_{>0}$:

$$
c_g \in [1,C], \quad \sigma_g \in \mathbb{R}_{>0}, \quad g=1,\dots,G. \tag{1}
$$

**初始化策略:** $c_g$ 在 $[1,C]$ 上均匀分布;$\sigma_g$ 取使相邻分组在
半高 (FWHM) 处覆盖率约为 $30\%$ 的值,即
$\sigma_g = 1.3\,C/[G\cdot 2\sqrt{2\ln 2}]$。该初始化保证训练开始时
所有波段都被覆盖,且相邻组存在适度交叠,符合真实光谱吸收特征的连续
性质。

对每个波段 $\lambda$ 计算其对各组的**软隶属度**(沿组维 softmax 归一化):

$$
\alpha_{g,\lambda}
= \frac{\exp\!\big(-\tfrac{(\lambda-c_g)^2}{2\sigma_g^2}\big)}
       {\sum_{j=1}^{G}\exp\!\big(-\tfrac{(\lambda-c_j)^2}{2\sigma_j^2}\big)}.
\tag{2}
$$

第 $g$ 组以此为加权得到一份**软波段加权 HSI 切片**:

$$
\mathbf{S}_g[i,j,\lambda] = \alpha_{g,\lambda}\,\mathbf{I}[i,j,\lambda],
\qquad \lambda=1,\dots,C. \tag{3}
$$

由于 $\alpha_{g,\lambda}$ 在 $c_g$ 处集中、向两侧衰减,$\mathbf{S}_g$
事实上保留了第 $g$ 组所"拥有"的那部分波段,而软边界允许相邻波段同时
参与若干组——这正是光谱吸收带连续延展的客观反映。

### 3.2.3 逐组嵌入

每个软切片 $\mathbf{S}_g$ 独立通过一个 $1\times 1$ 卷积、GroupNorm 与 SiLU
激活映射到 $M$ 维通道,其中 $M = D/G$:

$$
\mathbf{E}_g = \operatorname{SiLU}\!\Big(
   \operatorname{GN}\!\big(\operatorname{Conv}^{(g)}_{1\times1}(\mathbf{S}_g)\big)
\Big) \in\mathbb{R}^{H\times W\times M}. \tag{4}
$$

沿"组"轴堆叠得到带显式组结构的嵌入张量

$$
\mathbf{E}\in\mathbb{R}^{H\times W\times G\times M}
\;\cong\;\mathbb{R}^{H\times W\times D}\;(\text{沿通道拼接}). \tag{5}
$$

与传统 MambaHSI 在嵌入通道维上"硬切分"分组不同,$\mathbf{E}$ 的 $G\times M$
分块结构在**物理上是有意义的**:每一组对应一组以 $c_g$ 为中心的光谱子带。

### 3.2.4 组激发强度 $\bar{\phi}_g$

为支持后续 TSK 解模糊(§3.4),我们将每组的整体光谱覆盖权重定义为该组
隶属度在波段轴上的平均:

$$
\phi_g = \frac{1}{C}\sum_{\lambda=1}^{C}\alpha_{g,\lambda},
\qquad
\bar{\phi}_g = \frac{\phi_g}{\sum_{j=1}^{G}\phi_j}. \tag{6}
$$

$\bar{\phi}_g$ 可解释为第 $g$ 组对原始光谱的"占用比",并将以此权重参与
后续光谱通路的特征聚合。

---

## 3.3 模糊空间 Mamba 块 FSpaMB

### 3.3.1 角色

标准 Mamba 通过对输入做线性投影使状态空间的转移参数
$\mathbf{B},\mathbf{C},\boldsymbol{\Delta}$ 依赖于输入(即所谓"选择性")。
FSpaMB 在此基础上对**离散化时间步 $\boldsymbol{\Delta}$** 注入一组
**TSK 模糊规则**:不同规则在嵌入空间的不同区域被激发,从而以**多模态
(multi-modal)** 方式调制状态动力学的时间尺度,等价于在状态空间层面
引入一种"专家混合"机制。代码对应
`model/fuzzy_modules.py:FSpaMB`。

### 3.3.2 形式化:模糊 Δ-调制

定义 $R_{\text{spa}}$ 条规则,每条规则包含三类可学参数:

$$
\mathbf{m}_r^{\text{spa}}\in\mathbb{R}^{D'},\quad
\boldsymbol{\sigma}_r^{\text{spa}}\in\mathbb{R}_{>0}^{D'},\quad
\boldsymbol{\delta}_r\in\mathbb{R}^{D},\quad r=1,\dots,R_{\text{spa}}, \tag{7}
$$

其中 $D'\ll D$ 是用于规则激活计算的低秩投影维度。引入低秩投影是为了将
规则激活的代价由 $\mathcal{O}(D R)$ 压缩至 $\mathcal{O}(D' R)$,与 §3.7
的复杂度分析相一致。代码中 `proj`、`rule_centers`、`rule_log_sigmas`、
`rule_deltas` 分别对应 $\mathbf{W}_{\text{proj}}$、$\mathbf{m}_r^{\text{spa}}$、
$\log\boldsymbol{\sigma}_r^{\text{spa}}$、$\boldsymbol{\delta}_r$
(对方差取对数参数化以保证正值)。

对将 2D 特征展平后得到的第 $t$ 个空间 token
$\mathbf{x}_t\in\mathbb{R}^D$,首先做低秩投影:

$$
\tilde{\mathbf{x}}_t = \mathbf{W}_{\text{proj}}\,\mathbf{x}_t
\in\mathbb{R}^{D'}. \tag{8}
$$

对每条规则 $r$ 与每一维 $d\in\{1,\dots,D'\}$ 计算高斯隶属度,并取乘积
作为规则激发强度,再以 softmax 归一化得到归一规则激发:

$$
\mu_{r,d}^{\text{spa}}(\tilde x_{t,d})
= \exp\!\Big(-\tfrac{(\tilde x_{t,d}-m_{r,d}^{\text{spa}})^2}
                    {2(\sigma_{r,d}^{\text{spa}})^2}\Big),
\tag{9}
$$

$$
f_r^{\text{spa}}(\mathbf{x}_t) = \prod_{d=1}^{D'}\mu_{r,d}^{\text{spa}}(\tilde x_{t,d}),
\qquad
\bar f_r^{\text{spa}}(\mathbf{x}_t)
= \frac{f_r^{\text{spa}}(\mathbf{x}_t)}{\sum_{i=1}^{R_{\text{spa}}} f_i^{\text{spa}}(\mathbf{x}_t)}. \tag{10}
$$

实现上为数值稳定起见,所有计算均在对数域进行:
$\log f_r=\sum_d\log\mu_{r,d}$,再做 softmax。

随后以 TSK 一阶形式给出**模糊 Δ-偏移项**:

$$
\boldsymbol{\Delta}_t^{\text{fuzzy}}
= \sum_{r=1}^{R_{\text{spa}}}\bar f_r^{\text{spa}}(\mathbf{x}_t)\,\boldsymbol{\delta}_r
\in\mathbb{R}^{D}. \tag{11}
$$

最终的离散化时间步为标准 Mamba 投影与模糊偏移的可门控和:

$$
\boldsymbol{\Delta}_t = \operatorname{softplus}\!\Big(
   \mathbf{W}_{\Delta}\,\mathbf{x}_t + \lambda(t)\,\boldsymbol{\Delta}_t^{\text{fuzzy}}
\Big), \tag{12}
$$

其中 $\lambda(t)\in[0,1]$ 是预热系数(详见 §3.6),用于在训练早期抑制
未充分学习的模糊参数对主干的扰动。给定 $\boldsymbol{\Delta}_t$,通过零阶
保持 (ZOH) 离散化得到 $\bar{\mathbf{A}}_t,\bar{\mathbf{B}}_t$,SSM 递推
照搬标准 Mamba:

$$
\mathbf{h}_t = \bar{\mathbf{A}}_t\,\mathbf{h}_{t-1} + \bar{\mathbf{B}}_t\,\mathbf{x}_t,
\qquad
y_t = \mathbf{C}_t\,\mathbf{h}_t. \tag{13}
$$

### 3.3.3 块级前向传播

记编码器中第 $i$ 个块的输入为 $\mathbf{H}^i\in\mathbb{R}^{B\times D\times H\times W}$,
FSpaMB 的整体前向写作

$$
\begin{aligned}
\mathbf{HF}_{\text{spa}} &= \operatorname{Flatten}(\mathbf{H}^i)
   \in\mathbb{R}^{B\times L_1\times D},\quad L_1=HW, \\
\mathbf{HR}_{\text{spa}} &= \operatorname{SiLU}\!\Big(
   \operatorname{GN}\!\big(\operatorname{FuzzyMamba}_{\text{spa}}
        (\mathbf{HF}_{\text{spa}})\big)\Big), \\
\mathbf{H}_{\text{spa}}^o &= \operatorname{Reshape}(\mathbf{HR}_{\text{spa}}) + \mathbf{H}^i,
\end{aligned}\tag{14}
$$

其中 $\operatorname{FuzzyMamba}_{\text{spa}}$ 表示式 (8)–(13) 共同定义的
带模糊 Δ-调制的选择性 SSM。残差连接保证模糊扰动只是对主干的"加性增量",
不影响主干在 $\lambda(t)=0$ 时退化为标准 Mamba。

### 3.3.4 与软专家混合 (Soft MoE) 的关系

式 (11) 在数学上是 $R_{\text{spa}}$ 个"专家偏移向量"
$\{\boldsymbol{\delta}_r\}_{r=1}^{R_{\text{spa}}}$ 的软加权组合,
权重由高斯-乘积隶属度给出。与基于 softmax-MLP 的 MoE 路由相比,该路由器
有两条独有优势:(i) 中心 $\mathbf{m}_r$ 与带宽 $\boldsymbol{\sigma}_r$
均为可视化、可聚类的标量参数,显式刻画"专家激活区域";(ii) 高斯隶属
天然在数据稀疏处衰减为零,从而保证规则不会跨越数据分布外的区域过度激
活——这是 sigmoid-门控 MoE 不具备的性质。

---

## 3.4 模糊光谱 Mamba 块 FSpeMB

### 3.4.1 角色

FSpeMB 沿 $\mathbf{E}\in\mathbb{R}^{H\times W\times G\times M}$ 的**组轴 $G$**
做序列建模,用 Mamba 捕捉跨光谱组的依赖关系,并使用 §3.2.4 中得到的组
激发强度 $\bar{\phi}_g$ 做 TSK 解模糊,从而获得对像素级光谱表征的高效
聚合。代码对应 `model/fuzzy_modules.py:FSpeMB`。

### 3.4.2 形式化

设编码器输入仍保留组结构
$\mathbf{H}^i\in\mathbb{R}^{B\times H\times W\times G\times M}$。将批维与
空间维并为一个**伪批维** $N=BHW$,在组轴上应用 Mamba:

$$
\begin{aligned}
\mathbf{HF}_{\text{spe}} &= \operatorname{Flatten}(\mathbf{H}^i)
   \in\mathbb{R}^{N\times G\times M}, \\
\mathbf{HR}_{\text{spe}} &= \operatorname{SiLU}\!\Big(
   \operatorname{GN}\!\big(\operatorname{Mamba}(\mathbf{HF}_{\text{spe}})\big)\Big)
   \in\mathbb{R}^{N\times G\times M}.
\end{aligned}\tag{15}
$$

> **实现注记:** 在大尺寸 HSI(如 HanChuan、HongHu、Houston)中
> $N=BHW$ 可超过 $2\times 10^{5}$,而 Mamba 的 Triton CUDA grid 在 $y$
> 维上有 $65535$ 的硬上限。我们因此在 `FSpeMB.forward` 中以
> `_MAMBA_MAX_BATCH = 32768` 沿 $N$ 维度分块串行执行,保证大图也能在
> 单卡上完成训练且对结果无影响。

随后做**带 $\bar{\phi}_g$ 加权的 TSK 一阶解模糊**:每组以独立的线性映射
$(\mathbf{W}_g^{\text{up}}\in\mathbb{R}^{M\times D},\,\mathbf{b}_g\in\mathbb{R}^{D})$
将组内表征上投到主干维度 $D$,再以 $\bar{\phi}_g$ 加权求和:

$$
\mathbf{H}_{\text{spe}}^o[i,j]
= \mathbf{H}^i[i,j] +
  \sum_{g=1}^{G}\bar{\phi}_g\Big(
       \mathbf{W}_g^{\text{up}}\,\mathbf{HR}_{\text{spe}}[i,j,g,:] + \mathbf{b}_g
  \Big) \in\mathbb{R}^{D}. \tag{16}
$$

代码中 `group_up`、`group_bias` 即
$\{\mathbf{W}_g^{\text{up}},\mathbf{b}_g\}_{g=1}^{G}$,整体计算以
`torch.einsum('bgmhw,gmd->bgdhw', hr, group_up)` 一次完成。

### 3.4.3 物理一致性

由于 $\bar{\phi}_g$ 直接源自原始波段轴上的隶属度均值,式 (16) 实质上是
**以每组光谱占用比为权重**对各组上投影特征做凸组合。$c_g$ 的训练后值可
被解读为第 $g$ 组的**学习中心波长**,$\sigma_g$ 则为其**学习带宽**——
这赋予了 FSpeMB 在物理层面可解释的语义,而非仅是"隐式通道分组"。

---

## 3.5 TSK 空间-光谱融合模块 TSSFM

### 3.5.1 角色

经典 MambaHSI 通过两个全局标量权重将 $\mathbf{H}_{\text{spa}}^o$ 与
$\mathbf{H}_{\text{spe}}^o$ 相加。这种粗粒度融合不具备空间自适应性:
边界像素与同质区域内部像素应当采用不同的融合偏好。**TSSFM** 用基于像素
内容的、可微分的 **TSK 一阶模糊推理**取代它,代码对应
`model/fuzzy_modules.py:TSSFM`。

### 3.5.2 形式化

对每一像素 $(i,j)$,首先以 $1\times 1$ 卷积 $\mathbf{W}_z$ 将拼接的
空间-光谱特征压缩为低维**前件向量** $\mathbf{z}_{ij}\in\mathbb{R}^{D_z}$:

$$
\mathbf{z}_{ij} = \mathbf{W}_z\,
   [\mathbf{H}_{\text{spa}}^o[i,j]\,\|\,\mathbf{H}_{\text{spe}}^o[i,j]]
 + \mathbf{b}_z. \tag{17}
$$

定义 $R_f$ 条融合规则
$\{\mathbf{m}_r^{\text{fus}},\boldsymbol{\sigma}_r^{\text{fus}},
\mathbf{W}_r^{\text{fus}},\mathbf{b}_r^{\text{fus}}\}_{r=1}^{R_f}$,
其中前件部分以 Gaussian 乘积形式定义规则激发:

$$
f_r^{\text{fus}}(\mathbf{z}_{ij})
= \prod_{d=1}^{D_z}\exp\!\Big(-\tfrac{(z_{ij,d}-m_{r,d}^{\text{fus}})^2}
                                    {2(\sigma_{r,d}^{\text{fus}})^2}\Big),
\quad
\bar f_r^{\text{fus}}(\mathbf{z}_{ij})
= \frac{f_r^{\text{fus}}(\mathbf{z}_{ij})}
       {\sum_{i=1}^{R_f} f_i^{\text{fus}}(\mathbf{z}_{ij})}. \tag{18}
$$

后件部分采用一阶**向量值**线性映射:

$$
\hat{\mathbf{y}}_r(\mathbf{z}_{ij})
= \mathbf{W}_r^{\text{fus}}\,\mathbf{z}_{ij}+\mathbf{b}_r^{\text{fus}}
\in\mathbb{R}^{D}. \tag{19}
$$

最终融合特征以解模糊形式给出,并通过残差连接与原始空-谱特征求和:

$$
\mathbf{H}_{\text{fus}}[i,j]
= \mathbf{H}_{\text{spa}}^o[i,j] + \mathbf{H}_{\text{spe}}^o[i,j]
+ \lambda(t)\sum_{r=1}^{R_f}\bar f_r^{\text{fus}}(\mathbf{z}_{ij})\,
   \hat{\mathbf{y}}_r(\mathbf{z}_{ij}). \tag{20}
$$

预热系数 $\lambda(t)$ 与 §3.3 共享,使融合分支在训练早期由"空-谱求和"
平滑过渡到"模糊推理融合"。代码中 `antecedent`、`rule_centers`、
`rule_log_sigmas`、`consequent_w`、`consequent_b` 分别对应
$\mathbf{W}_z$、$\mathbf{m}_r^{\text{fus}}$、$\log\boldsymbol{\sigma}_r^{\text{fus}}$、
$\mathbf{W}_r^{\text{fus}}$、$\mathbf{b}_r^{\text{fus}}$。

### 3.5.3 表达能力

式 (19) 中每条规则的后件是 $\mathbf{z}_{ij}$ 的一阶线性函数,因此每条规则
"同时**混合并变换**"输入特征——这严格强于任何对
$\mathbf{H}_{\text{spa}}^o$ 与 $\mathbf{H}_{\text{spe}}^o$ 的标量加权和。
更重要的是,$\mathbf{z}_{ij}$ 是**逐像素**计算的,这使融合具备空间内容
自适应性:位于地物边界的像素会自发偏向空间特征,而同质植被内部像素则
偏向光谱特征。Wang (1992) 证明带高斯前件与一阶线性后件的 TSK 系统是
**通用近似器**,因此 TSSFM 在表达能力上严格优于全局标量融合。

---

## 3.6 训练策略

### 3.6.1 模糊参数的预热调度

随机初始化的高斯参数若与主干同步从零步训练,容易出现 (i) 规则中心
被推到数据分布之外、(ii) 隶属度饱和导致梯度消失,以及 (iii) 模糊偏移项
对早期不稳定主干的过强扰动。我们因此引入线性预热系数:

$$
\lambda(t) = \min\!\Big(1,\,\tfrac{t}{T_{\text{warm}}}\Big),
\qquad t=1,2,\dots, \tag{21}
$$

其中 $t$ 为训练 epoch。$\lambda(t)$ 同时门控 FSpaMB 的模糊 Δ-偏移项
(式 12) 与 TSSFM 的模糊融合项 (式 20)。在前 $T_{\text{warm}}$ 个 epoch
网络近似为标准 MambaHSI;之后模糊推理被平滑引入。注意 §3.2 中的模糊光
谱分组**从第一个 epoch 起就处于激活状态**,因为它构成嵌入本身,不能被
关闭。代码实现于 `train_MambaHSI.py` 中:

```python
lam = min(1.0, (epoch + 1) / max(1, args.T_warm))
net.set_warmup(lam)
```

并由 `MambaHSI.set_warmup → FuzzyEncoderBlock.set_warmup` 同步分发至
所有 FSpaMB 与 TSSFM。

### 3.6.2 分类损失

逐像素交叉熵损失,忽略未标注像元 ($\mathbf{Y}=-1$):

$$
\mathcal{L}_{\text{cls}}
= \operatorname{CrossEntropy}(\mathbf{l},\,\mathbf{Y}_{\text{tr}}). \tag{22}
$$

为对齐输出 logits 与标签的空间分辨率,我们以双线性插值 (`align_corners=True`)
将 $\mathbf{l}$ 上采样到标签尺寸,见 `utils/Loss.py:head_loss`。

### 3.6.3 规则多样性正则

无监督的"互推远"正则
$-\tfrac{1}{R(R-1)}\sum_{r\ne s}\|\mathbf{m}_r-\mathbf{m}_s\|_2$
会**无界**地奖励将规则中心推开,从而把中心驱出数据分布。我们采用一个
**铰链 (hinge) 形、按中心尺度归一化**的相对间隔惩罚:

$$
\mathcal{L}_{\text{div}}
= \frac{1}{R(R-1)}\sum_{r\ne s}
   \max\!\Big(0,\,\tau-\tfrac{\|\mathbf{m}_r-\mathbf{m}_s\|_2}
                                 {\|\mathbf{m}_r\|_2+\|\mathbf{m}_s\|_2+\epsilon}\Big)^2,
\tag{23}
$$

其中 $\tau$ (默认 $0.1$) 为相对间隔阈值。当任两条规则的归一化间隔已大于
$\tau$ 时正则项梯度为零,中心可在数据偏好处自由稳定;若两条规则相对
过于接近(冗余规则),则给予二次惩罚。该正则项对所有 FSpaMB 与 TSSFM
的规则中心求平均,实现于 `utils/Loss.py:diversity_loss`。

### 3.6.4 总损失

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \beta\,\mathcal{L}_{\text{div}},
\quad \beta\approx 0.01. \tag{24}
$$

### 3.6.5 优化与工程细节

* **优化器与学习率.** 采用 Adam 优化器,初始学习率 $3\times 10^{-4}$,并
  使用 `CosineAnnealingLR`(`T_max=max_epoch`,$\eta_{\min}=\eta_0\cdot 10^{-2}$)
  在 200 epoch 上做余弦退火,以稳定训练后期的指标振荡。
* **混合精度.** 通过 `torch.cuda.amp.autocast` 与 `GradScaler` 在显存
  受限的大图数据集上启用 FP16 训练 (`--use_amp`)。
* **分块前向 (tiled forward).** 对 HanChuan、HongHu、Houston 等大图,
  我们沿高度方向将整图切为 $n_{\text{splits}}$ 个带 5 像素重叠的水平条
  带,逐条进行前向并在拼接时丢弃重叠区,从而避免单次前向 OOM,且不
  破坏分类输出。
* **CUDA-内核兼容性.** Mamba2 的 `causal_conv1d` 要求若干通道维同时被
  $8$ 整除;`_choose_mamba2_kwargs` 自动选取最大可行的
  `headdim` 与 `d_state`,以保证快路径可用;若 `causal_conv1d_cuda` 不
  可加载,模型自动回退到纯 PyTorch 实现的等价 SSM,使训练在 CPU/无
  扩展环境也可运行(但仅 CUDA 路径具有完整 SSM 递推语义)。

---

## 3.7 复杂度与可解释性

### 3.7.1 计算复杂度

记 $L=HW$ 为空间 token 数。各模块的逐前向代价为:

| 模块 | 主要代价 | 复杂度 |
|---|---|---|
| 模糊光谱分组 | 计算 $\alpha_{g,\lambda}$ 与加权 | $\mathcal{O}(GC + LCG)$ |
| 逐组嵌入 | $G$ 个 $1\times 1$ Conv | $\mathcal{O}(LGCM)=\mathcal{O}(LCD)$ |
| FSpaMB(选择性 SSM + 模糊路由) | 标准 Mamba $\mathcal{O}(LD)$ + 模糊 $\mathcal{O}(L D'R_{\text{spa}})$ | $\mathcal{O}(L(D+D'R_{\text{spa}}))$ |
| FSpeMB(组轴 Mamba + TSK 解模糊) | $\mathcal{O}(LGM)+\mathcal{O}(LGMD)$ | $\mathcal{O}(LGD)$ |
| TSSFM(像素级 TSK) | $\mathcal{O}(L D_z R_f D)$ | $\mathcal{O}(L D_z R_f D)$ |

由于 $D',D_z,R_{\text{spa}},R_f,G$ 均为与 $L$ 无关的小常数(典型取值
$D'=D_z=32,\,R_{\text{spa}}=R_f=4,\,G=8$),整体逐前向代价为
$\mathcal{O}(L)$——**模糊推理并未破坏 Mamba 的线性 token 复杂度**。

### 3.7.2 参数量

设 $R = R_{\text{spa}} = R_f$。三个核心模块新增的参数主要由
$N_{\text{enc}}$ 个编码器块累积:

* **FSpaMB**(每块):$D D' + 2RD' + RD$
* **FSpeMB**(每块):$GMD + GD$
* **TSSFM**(每块):$2D\cdot D_z + 2RD_z + R D_z D + RD$
* **模糊光谱分组**(整网共享):$2G$

由于这些项的系数均不依赖 $H,W$,模型的参数量与图像尺寸无关。

### 3.7.3 可解释性诊断协议

我们将解释性视作一项**经验问题**,而非由设计直接断言。本文承诺以下三种
事后分析(详见实验部分):

1. **规则激活图.** 对测试图像逐 token 绘制 $\bar f_r^{\text{spa}}$ 与
   $\bar f_r^{\text{fus}}$ 的热力图,观察规则在空间上的分布是否与地物
   结构吻合;
2. **类条件激活均值.** 估计 $\mathbb{E}[\bar f_r\mid y=k]$,检验规则是
   否对类别有专一性偏好;
3. **规则使用熵.** 计算
   $\mathcal{H}(\bar f_r) = -\sum_r \bar f_r\log\bar f_r$ 的样本平均,
   若该值塌缩到 $\log R$ 远小的水平,则提示规则坍缩,需要回到式 (23)
   的多样性正则强度;
4. **光谱组中心稳定性.** 在训练曲线上记录 $\{c_g,\sigma_g\}$ 的轨迹,
   若稳定收敛则可与文献中已知吸收特征做对照,以验证 §3.4.3 中的物理
   可解释性。

在上述诊断完成之前,本文将 Fuzzy-MambaHSI 中的三处模糊推理仅作为**功
能性扩展**而非**显式可解释机制**呈现:它们经证实地拓宽了 SSM 的状态
动力学、光谱表征与空-谱融合的表达能力,而是否在语义上对应"可命名的
模糊规则"留待实验验证。

---

## 小结

本节给出了 Fuzzy-MambaHSI 的全部技术细节:从对原始光谱波段的可学习
软分组(§3.2)、到对 Mamba 选择性时间步的 TSK 模糊调制(§3.3)、
到组轴上的光谱 Mamba 与 $\bar\phi_g$ 加权解模糊(§3.4)、再到取代传统
标量融合的像素级 TSK 空间-光谱融合(§3.5),并辅以预热调度、多样性
正则和与硬件兼容的训练工程(§3.6),整体在线性复杂度内显式刻画了
HSI 数据的光谱不确定性(§3.7)。
