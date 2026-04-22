# Fuzzy-MambaHSI: A Deep Neuro-Fuzzy State Space Model for Hyperspectral Image Classification
## (Revised Design — v2)

> **Revision notes (v1 → v2).** This version incorporates four substantive corrections raised in technical review:
> (i) FSpeMB now applies fuzzy grouping on the **raw spectral dimension** rather than on the latent embedding index, restoring physical correspondence between Gaussian centers and wavelength-ordered bands;
> (ii) interpretability claims for FSpaMB are tempered and moved to a post-hoc validation commitment rather than being asserted a priori;
> (iii) the diversity regularizer is reformulated as a hinge-style separation penalty, which does not reward unbounded spread;
> (iv) the complexity analysis is expanded to account for rule-count and projection-dimension constants, with a corresponding reporting protocol (FLOPs, wall-clock, memory).
> A new §7 specifies the ablation protocol required to isolate the contribution of each module.

---

## 1. Motivation and Core Idea

Hyperspectral image (HSI) classification faces two fundamental challenges that existing methods address only partially:

**Challenge 1 — Spectral Ambiguity.** The spectral signatures of different land-cover materials frequently overlap, and mixed pixels (containing multiple materials within a single pixel footprint) introduce inherent uncertainty into the feature space. This uncertainty is *fuzzy* in nature: a pixel's membership in a given class is often a matter of degree, not a binary decision.

**Challenge 2 — Long-Range Spatial Dependence at Linear Cost.** Pixel-level classification demands that every pixel's representation encode contextual information from distant regions. Transformers achieve this but at quadratic cost; CNNs are limited to local receptive fields. Mamba (selective state space models) offers linear-complexity long-range modeling, but its state transitions are purely data-driven and lack an explicit mechanism for handling the fuzzy nature of spectral data.

**Our proposal — Fuzzy-MambaHSI** bridges these two challenges by embedding a TSK-style fuzzy inference system directly into the Mamba-based architecture. Rather than applying fuzzy logic as a post-hoc adapter (as in Fuzzy-ViT's cross-domain setting), we integrate it *within* the state space dynamics, the spectral grouping mechanism, and the spatial-spectral fusion stage, producing an architecture whose inductive biases are natively aligned with the spectral uncertainty inherent in HSI data.

### Distinction from Prior Work

| Aspect | MambaHSI | Fuzzy-ViT | **Fuzzy-MambaHSI (Ours)** |
|---|---|---|---|
| Backbone | Mamba (SSM) | ViT (Attention) | Mamba (SSM) |
| Fuzzy integration | None | Sigmoid membership + learnable rules as cross-domain adapter | Gaussian TSK membership embedded in Δ-modulation, raw-band spectral grouping, and pixel-wise fusion |
| Spectral handling | Hard group split on embedding channels + Mamba | N/A (RGB/medical) | Fuzzy membership grouping on **raw bands** + per-group embedding + Mamba |
| Spatial-spectral fusion | Two global scalar weights | N/A | Pixel-wise first-order TSK fuzzy inference |
| Complexity | O(L) | O(L²) for attention | O(L) preserved (see §6 for constants) |
| Primary setting | HSI classification | Cross-domain transfer (general → medical) | HSI classification with explicit uncertainty modeling |

---

## 2. Overall Architecture

```
Input HSI  I ∈ R^{H×W×C}
       │
       ▼
┌──────────────────────────────────────────┐
│     Fuzzy Spectral Grouping (on raw C)   │   ← operates on physical bands
│     G soft groups via Gaussian μ(λ)       │
│     S_g ∈ R^{H×W×|G_g|}, g = 1..G         │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│     Per-Group Embedding (1×1 Conv, GN,    │
│     SiLU) → stacked to E ∈ R^{H×W×D}      │
└──────────────────┬───────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────┐
│          Encoder  (×N_enc blocks)         │
│                                           │
│  ┌────────────────┐  ┌────────────────┐   │
│  │ Fuzzy Spatial  │  │ Fuzzy Spectral │   │
│  │ Mamba Block    │  │ Mamba Block    │   │
│  │ (FSpaMB)       │  │ (FSpeMB)       │   │
│  └───────┬────────┘  └───────┬────────┘   │
│          │                    │            │
│          └─────────┬──────────┘            │
│                    ▼                       │
│     ┌──────────────────────────┐           │
│     │  TSK Fuzzy Spatial-       │           │
│     │  Spectral Fusion (TSSFM)  │           │
│     └──────────┬───────────────┘           │
└────────────────┼──────────────────────────┘
                 ▼
┌──────────────────────┐
│  Segmentation Head    │  Conv1×1 → logits
│  l ∈ R^{H×W×K}       │
└──────────────────────┘
```

Three strategic points of fuzzy integration are preserved from v1, but the **first** point has moved *upstream of embedding*, not inside it:

1. **Fuzzy Spectral Grouping + FSpeMB** — soft band-level partitioning, followed by Mamba modeling of inter-group relations.
2. **FSpaMB** — fuzzy-modulated spatial state transitions (Δ-modulation).
3. **TSSFM** — pixel-wise, content-dependent TSK fusion replacing global scalar weights.

---

## 3. Module Design

### 3.1 Fuzzy Spatial Mamba Block (FSpaMB)

#### 3.1.1 Role

In standard Mamba, the selectivity mechanism makes **B**, **C**, and **Δ** input-dependent via linear projections. FSpaMB *enriches* this selectivity with a fuzzy-rule-based modulation of Δ: multiple rules compete to set the effective time-scale per token, yielding multi-modal state dynamics. We position this as a **functional** improvement — enabling mixture-of-experts-like behavior over the state dynamics — and reserve the interpretability question for empirical validation (§7).

#### 3.1.2 Fuzzy Δ-Modulation

Define a learnable fuzzy rule base over the embedding space:

$$
\mathbf{m}_r^{\text{spa}} \in \mathbb{R}^{D'}, \quad \boldsymbol{\sigma}_r^{\text{spa}} \in \mathbb{R}_{>0}^{D'}, \quad \boldsymbol{\delta}_r \in \mathbb{R}^{D}, \quad r = 1, \dots, R_{\text{spa}}
$$

where $D' \ll D$ is a low-rank projection dimension used only for rule activation (keeping rule evaluation cheap). Given a flattened spatial token $\mathbf{x}_t \in \mathbb{R}^D$:

$$
\tilde{\mathbf{x}}_t = \mathbf{W}_{\text{proj}}\mathbf{x}_t \in \mathbb{R}^{D'}
\tag{1}
$$

$$
\mu_{r,d}^{\text{spa}}(\tilde{x}_{t,d}) = \exp\!\left(-\frac{(\tilde{x}_{t,d} - m_{r,d}^{\text{spa}})^2}{2(\sigma_{r,d}^{\text{spa}})^2}\right)
\tag{2}
$$

$$
f_r^{\text{spa}}(\mathbf{x}_t) = \prod_{d=1}^{D'} \mu_{r,d}^{\text{spa}}(\tilde{x}_{t,d}), \quad
\bar{f}_r^{\text{spa}}(\mathbf{x}_t) = \frac{f_r^{\text{spa}}(\mathbf{x}_t)}{\sum_{i=1}^{R_{\text{spa}}} f_i^{\text{spa}}(\mathbf{x}_t)}
\tag{3}
$$

The fuzzy Δ-offset is then the TSK-weighted consequent:

$$
\boldsymbol{\Delta}_t^{\text{fuzzy}} = \sum_{r=1}^{R_{\text{spa}}} \bar{f}_r^{\text{spa}}(\mathbf{x}_t)\cdot \boldsymbol{\delta}_r
\tag{4}
$$

$$
\boldsymbol{\Delta}_t = \operatorname{softplus}\!\left(\mathbf{W}_{\Delta}\mathbf{x}_t + \lambda(t)\cdot\boldsymbol{\Delta}_t^{\text{fuzzy}}\right)
\tag{5}
$$

where $\lambda(t)\in[0,1]$ is the warm-up coefficient (§5.1) that gates the fuzzy contribution during early training. Discretized state matrices $\bar{\mathbf{A}}_t, \bar{\mathbf{B}}_t$ are computed from $\boldsymbol{\Delta}_t$ via zero-order hold and the recurrence proceeds as standard Mamba:

$$
\mathbf{h}_t = \bar{\mathbf{A}}_t \mathbf{h}_{t-1} + \bar{\mathbf{B}}_t \mathbf{x}_t, \qquad y_t = \mathbf{C}_t \mathbf{h}_t
\tag{6}
$$

#### 3.1.3 FSpaMB Forward Pass

$$
\begin{aligned}
\mathbf{HF}_{\text{spa}} &= \text{Flatten}(\mathbf{H}^i) \in \mathbb{R}^{B \times L_1 \times D} \\
\mathbf{HR}_{\text{spa}} &= \text{SiLU}\!\left(\text{GN}\!\left(\text{FuzzyMamba}_{\text{spa}}(\mathbf{HF}_{\text{spa}})\right)\right) \\
\mathbf{H}_{\text{spa}}^o &= \text{Reshape}(\mathbf{HR}_{\text{spa}}) + \mathbf{H}^i
\end{aligned}
\tag{7}
$$

#### 3.1.4 Honest Statement on Interpretability

The rules $\{r = 1, \dots, R_{\text{spa}}\}$ partition the embedding space into soft regions, each with its own state-dynamics preference $\boldsymbol{\delta}_r$. Whether these regions align with semantically meaningful "spatial context modes" (e.g., vegetation vs. urban textures) is an *empirical* question. We commit to the following post-hoc analyses in §7 before any interpretability claim is made:
(a) visualizing per-rule activation maps $\bar{f}_r^{\text{spa}}(\mathbf{x}_t)$ over test images;
(b) reporting per-class mean activation $\mathbb{E}[\bar{f}_r^{\text{spa}}\mid y=k]$ to check whether rules specialize to land-cover classes;
(c) measuring rule-usage entropy to detect rule collapse.
Until these analyses are reported, FSpaMB is presented purely as a multi-modal Δ-modulation mechanism.

---

### 3.2 Fuzzy Spectral Mamba Block (FSpeMB) — *Restructured*

#### 3.2.1 Why the Redesign

In v1, fuzzy spectral grouping was defined on the embedding dimension index $d \in \{1, \dots, D\}$. After 1×1 convolution, however, each channel $d$ is already a learned mixture of original bands, so Gaussian centers over $d$ do not capture "spectral locality" in any physical sense. In v2 we move fuzzy grouping **upstream of embedding**, directly onto the raw spectral axis $\lambda \in \{1, \dots, C\}$, where the index is wavelength-ordered and physically meaningful.

#### 3.2.2 Fuzzy Spectral Grouping on Raw Bands

Define $G$ learnable group centers and widths in band-index space:

$$
c_g \in [1, C], \quad \sigma_g \in \mathbb{R}_{>0}, \quad g = 1, \dots, G
\tag{8}
$$

(Initialization: $c_g$ evenly spaced over $[1, C]$; $\sigma_g$ initialized so neighboring group supports overlap by ≈30% at half-maximum.)

For each raw band $\lambda$, compute its soft membership to each group:

$$
\alpha_{g,\lambda} = \frac{\exp\!\left(-\frac{(\lambda - c_g)^2}{2\sigma_g^2}\right)}{\sum_{j=1}^{G}\exp\!\left(-\frac{(\lambda - c_j)^2}{2\sigma_j^2}\right)}
\tag{9}
$$

Each group then constructs a soft band-weighted HSI slice:

$$
\mathbf{S}_g[i,j,\lambda] = \alpha_{g,\lambda}\cdot \mathbf{I}[i,j,\lambda], \quad \lambda = 1,\dots,C
\tag{10}
$$

Because $\alpha_{g,\lambda}$ is concentrated near $c_g$, $\mathbf{S}_g$ effectively retains the subset of bands that group $g$ "owns," but with soft (differentiable) boundaries that let adjacent bands participate in multiple groups — reflecting the continuous nature of spectral absorption features.

#### 3.2.3 Per-Group Embedding

Each group-specific HSI slice is embedded independently:

$$
\mathbf{E}_g = \operatorname{SiLU}\!\left(\operatorname{GN}\!\left(\operatorname{Conv}_{1\times1}^{(g)}(\mathbf{S}_g)\right)\right) \in \mathbb{R}^{H\times W\times M}
\tag{11}
$$

where $M = D/G$. Stacking groups along a new "group" axis gives the embedding used by the encoder:

$$
\mathbf{E} \in \mathbb{R}^{H\times W\times G\times M}, \quad \text{equivalent to } \mathbb{R}^{H\times W\times D} \text{ after concatenation}
\tag{12}
$$

This is the embedding that enters FSpaMB. The crucial distinction from MambaHSI is that the block structure $G\times M$ is now **semantically grounded in spectral band groups**, not arbitrary channel partitioning.

#### 3.2.4 FSpeMB: Inter-Group Mamba with TSK Defuzzification

Within the encoder, FSpeMB operates on the group axis. Let $\mathbf{H}^i \in \mathbb{R}^{B\times H\times W\times G\times M}$ be the input with group structure preserved:

$$
\begin{aligned}
\mathbf{HF}_{\text{spe}} &= \text{Flatten}(\mathbf{H}^i) \in \mathbb{R}^{N\times G\times M}, \quad N = B\cdot H\cdot W\\
\mathbf{HR}_{\text{spe}} &= \text{SiLU}\!\left(\text{GN}\!\left(\text{Mamba}(\mathbf{HF}_{\text{spe}})\right)\right) \in \mathbb{R}^{N\times G\times M}
\end{aligned}
\tag{13}
$$

The Mamba here models inter-group sequence relations, as in MambaHSI. For defuzzification, we use the **group firing strength** derived from the mean band membership:

$$
\phi_g = \frac{1}{C}\sum_{\lambda=1}^{C}\alpha_{g,\lambda}, \quad \bar{\phi}_g = \frac{\phi_g}{\sum_j \phi_j}
\tag{14}
$$

The TSK output fuses groups via a first-order consequent:

$$
\mathbf{H}_{\text{spe}}^o[i,j] = \mathbf{H}^i[i,j] + \sum_{g=1}^{G}\bar{\phi}_g\left(\mathbf{W}_g^{\text{up}}\mathbf{HR}_{\text{spe}}[i,j,g,:] + \mathbf{b}_g\right) \in \mathbb{R}^D
\tag{15}
$$

#### 3.2.5 Summary of the Redesign

The restructured FSpeMB now has a **genuine** physical interpretation: $c_g$ is a learned central wavelength for group $g$, $\sigma_g$ is its bandwidth, and $\phi_g$ is the group's overall spectral coverage weight. This directly addresses the reviewer's concern that v1 was computing "latent-channel fuzzy grouping" disguised as spectral grouping.

---

### 3.3 TSK Fuzzy Spatial-Spectral Fusion Module (TSSFM)

This module is retained from v1 with only minor refinements. It remains the innovation most likely to deliver measurable gains, because it directly replaces the coarse two-scalar fusion of MambaHSI with pixel-wise content-dependent blending.

#### 3.3.1 Formulation

Define $R_f$ fusion rules. Compute a compact antecedent representation:

$$
\mathbf{z}_{ij} = \mathbf{W}_z [\mathbf{H}_{\text{spa}}^o[i,j] \,\|\, \mathbf{H}_{\text{spe}}^o[i,j]] + \mathbf{b}_z \in \mathbb{R}^{D_z}
\tag{16}
$$

Rule activations (Gaussian antecedent):

$$
f_r^{\text{fus}}(\mathbf{z}_{ij}) = \prod_{d=1}^{D_z} \exp\!\left(-\frac{(z_{ij,d} - m_{r,d}^{\text{fus}})^2}{2(\sigma_{r,d}^{\text{fus}})^2}\right), \quad
\bar{f}_r^{\text{fus}}(\mathbf{z}_{ij}) = \frac{f_r^{\text{fus}}(\mathbf{z}_{ij})}{\sum_i f_i^{\text{fus}}(\mathbf{z}_{ij})}
\tag{17}
$$

First-order TSK consequent (vector-valued, rather than scalar weighting):

$$
\hat{\mathbf{y}}_r(\mathbf{z}_{ij}) = \mathbf{W}_r^{\text{fus}}\mathbf{z}_{ij} + \mathbf{b}_r^{\text{fus}} \in \mathbb{R}^D
\tag{18}
$$

Defuzzified, residual-connected fused output:

$$
\mathbf{H}_{\text{fus}}[i,j] = \mathbf{H}^i[i,j] + \sum_{r=1}^{R_f}\bar{f}_r^{\text{fus}}(\mathbf{z}_{ij})\cdot\hat{\mathbf{y}}_r(\mathbf{z}_{ij})
\tag{19}
$$

The consequents are linear functions, so each rule simultaneously *blends* and *transforms* features — strictly more expressive than any scalar-weighted sum of $\mathbf{H}_{\text{spa}}^o$ and $\mathbf{H}_{\text{spe}}^o$. Crucially, $\mathbf{z}_{ij}$ is computed **per pixel**, so the fusion adapts locally: a boundary pixel can rely on spatial features while an interior vegetation pixel can rely on spectral features.

---

## 4. Theoretical Justification

### 4.1 Why Gaussian TSK Rather than Sigmoid Membership

Fuzzy-ViT uses $M = \sigma(XW_r^T)$, a sigmoid membership which acts as a soft cross-domain gate. Gaussian membership is better suited to HSI for three reasons:

1. **Physical locality.** With the v2 redesign, spectral group centers $c_g$ live in band-index space. Gaussian membership directly encodes the localized absorption structure of real spectra; a sigmoid projection would instead compute a global correlation that discards band ordering.
2. **TSK universal approximation.** A first-order TSK system with Gaussian antecedents and linear consequents is a universal approximator (Wang, 1992). The full TSSFM pipeline (antecedent → activation → linear consequent → weighted defuzzification) inherits this property, which the simpler sigmoid formulation does not.
3. **Post-hoc diagnosability.** Centers $m_r$ and widths $\sigma_r$ are inspectable scalars — we can plot them, cluster them, and check whether they stabilize during training. This enables the validation protocol in §7.

### 4.2 Expressiveness as a Hierarchical Fuzzy System

The three modules form a three-level fuzzy reasoning chain:
- **Level 1 (FSpeMB, upstream of embedding):** Fuzzy rules determine *what is grouped* — the composition of spectral features.
- **Level 2 (FSpaMB):** Fuzzy rules determine *how context flows* — the dynamics of the state space.
- **Level 3 (TSSFM):** Fuzzy rules determine *how sources are combined* — the fusion of spatial and spectral representations.

Each level addresses a distinct source of uncertainty (composition, dynamics, fusion), and their composition is end-to-end differentiable.

---

## 5. Training Strategy

### 5.1 Staged / Warm-Up Training

Randomly initialized Gaussian parameters can saturate or collapse if trained alongside the full network from step zero. We adopt a warm-up schedule:

$$
\lambda(t) = \min\!\left(1, \frac{t}{T_{\text{warm}}}\right), \quad t = 1, 2, \dots
\tag{20}
$$

$\lambda(t)$ multiplies the fuzzy contribution inside FSpaMB (Eq. 5) and — optionally — inside TSSFM. For the first $T_{\text{warm}}$ epochs the network behaves close to standard MambaHSI (modulo fuzzy spectral grouping, which is active from the start because it replaces the embedding itself).

### 5.2 Loss Function

Classification loss follows MambaHSI:

$$
\mathcal{L}_{\text{cls}} = \operatorname{CrossEntropy}(\mathbf{l}, \mathbf{Y}_{\text{tr}})
\tag{21}
$$

### 5.3 Rule Diversity Regularization (Revised)

The v1 regularizer $\mathcal{L}_{\text{div}} = -\frac{1}{R(R-1)}\sum_{r\neq s}\|\mathbf{m}_r - \mathbf{m}_s\|_2$ unconditionally rewards pushing centers apart, which can drive them outside the data distribution. We replace it with a hinge-style separation penalty that only activates when centers are too close:

$$
\mathcal{L}_{\text{div}} = \frac{1}{R(R-1)}\sum_{r\neq s} \max\!\left(0,\; \tau - \frac{\|\mathbf{m}_r - \mathbf{m}_s\|_2}{\|\mathbf{m}_r\|_2 + \|\mathbf{m}_s\|_2 + \epsilon}\right)^2
\tag{22}
$$

Here the separation is **normalized by the centers' own scales**, so the penalty does not grow unbounded as $\|\mathbf{m}\|_2$ increases. The threshold $\tau$ (e.g., 0.1) defines a minimum relative spacing below which duplicate rules are penalized. Above $\tau$, the gradient is zero — centers are free to settle wherever the data prefer.

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}} + \beta \cdot \mathcal{L}_{\text{div}}
\tag{23}
$$

with $\beta \approx 0.01$.

---

## 6. Complexity Analysis (Expanded)

### 6.1 Asymptotic Complexity

Let $L = H \cdot W$ be the number of pixels, $D$ the embedding dimension, $G$ the number of spectral groups, $R_{\text{spa}}$ the number of spatial rules, $R_f$ the number of fusion rules, $D'$ the spatial rule projection dimension, and $D_z$ the fusion compression dimension.

| Stage | Complexity |
|---|---|
| Fuzzy spectral grouping | $O(L \cdot C \cdot G)$ |
| Per-group embedding | $O(L \cdot C \cdot M) = O(L \cdot D)$ |
| FSpaMB (rule eval + Mamba) | $O(L \cdot R_{\text{spa}} \cdot (D' + D))$ |
| FSpeMB (Mamba over $G$ groups) | $O(L \cdot G \cdot M) = O(L \cdot D)$ |
| TSSFM | $O(L \cdot R_f \cdot (D_z + D))$ |
| **Total** | $O\!\left(L \cdot (C \cdot G + D \cdot (R_{\text{spa}} + R_f))\right)$ |

All terms are linear in $L$. Asymptotically, Fuzzy-MambaHSI matches MambaHSI.

### 6.2 Constants Matter

The reviewer correctly notes that reporting only FLOPs or "linear complexity" hides real-world cost. In practice:

- The extra constants $R_{\text{spa}} \cdot D'$, $R_f \cdot D_z$, and the per-group 1×1 convs add non-trivial kernel-launch overhead, especially for small $B$ and moderate $D$.
- Memory scales with the rule-activation tensors $\bar{f}_r^{\text{spa}}(\mathbf{x}_t) \in \mathbb{R}^{B\times L\times R_{\text{spa}}}$ during training.

We therefore commit to reporting, in the experimental section:
(a) parameter count;
(b) FLOPs;
(c) peak GPU memory on each benchmark;
(d) wall-clock training time per epoch and inference time per image.

---

## 7. Ablation and Validation Protocol

This section formalizes the reviewer's requirement that the TSK framing not be dismissible as "more parameters." We commit to the following studies as part of the paper, not as an afterthought.

### 7.1 Core Module Ablation

Hold the backbone depth and embedding dimension fixed. Run the following six configurations:

| # | FSpaMB | FSpeMB | TSSFM | Purpose |
|---|---|---|---|---|
| A0 | — | — | — | MambaHSI baseline |
| A1 | ✓ | — | — | Isolate FSpaMB contribution |
| A2 | — | ✓ | — | Isolate FSpeMB contribution |
| A3 | — | — | ✓ | Isolate TSSFM contribution |
| A4 | ✓ | ✓ | — | Spatial+spectral fuzzy, scalar fusion |
| A5 | ✓ | ✓ | ✓ | Full Fuzzy-MambaHSI |

### 7.2 Design-Choice Ablation

| Study | Comparison | What it tests |
|---|---|---|
| S1 | Hard band split vs. Fuzzy band grouping | Whether soft spectral grouping helps |
| S2 | Scalar fusion vs. TSK fusion | Whether pixel-wise fusion helps |
| S3 | Sigmoid (Fuzzy-ViT-style) vs. Gaussian TSK membership in TSSFM | Whether Gaussian+TSK structure matters, or a simpler gate would do |
| S4 | Parameter-matched baseline (add MLP blocks of equivalent parameter count) | Whether gains come from fuzzy structure or simply from capacity |
| S5 | Diversity regularizer: v1 (unconditional spread) vs. v2 (hinge) vs. off | Validates §5.3 change |
| S6 | Rule count sweep $R_{\text{spa}}, R_f \in \{2, 4, 8, 16\}$ | Expressiveness vs. cost trade-off |

### 7.3 Interpretability Validation (for FSpaMB)

As committed in §3.1.4:
(a) per-rule activation heatmaps on representative test images;
(b) per-class rule-activation distributions $\mathbb{E}[\bar{f}_r^{\text{spa}}\mid y=k]$;
(c) rule-usage entropy $H_r = -\sum_t \bar{f}_r^{\text{spa}}(\mathbf{x}_t) \log \bar{f}_r^{\text{spa}}(\mathbf{x}_t)$ to detect rule collapse.

We only claim interpretability if (b) shows class-specific specialization and (c) remains bounded away from degenerate values.

### 7.4 Spectral Grouping Sanity Check (for FSpeMB)

After training, plot the learned $(c_g, \sigma_g)$ pairs against the dataset's known absorption bands (water absorption, chlorophyll reflectance edge, SWIR features, etc.). Sensible placement supports the physical-correspondence claim; random placement would indicate the model ignores the band-ordering prior.

---

## 8. Recommended Hyperparameters

| Parameter | Symbol | Suggested Value | Rationale |
|---|---|---|---|
| Spatial fuzzy rules | $R_{\text{spa}}$ | 4–8 | Balance expressiveness vs. overhead |
| Spectral groups | $G$ | 8–16 | Consistent with MambaHSI; adjust to $C$ |
| Fusion rules | $R_f$ | 4–6 | Enough to capture major land-cover modes |
| Spatial projection dim | $D'$ | 16–32 | Low-rank for efficient rule activation |
| Fusion compression dim | $D_z$ | 32–64 | Compact antecedent representation |
| Embedding dim | $D$ | 128 | Following MambaHSI |
| Per-group embedding dim | $M$ | $D/G$ | Preserves parameter budget |
| Encoder blocks | $N_{\text{enc}}$ | 2–4 | Depth for complex scenes |
| Warm-up epochs | $T_{\text{warm}}$ | 10–20 | Gradual fuzzy introduction |
| Diversity threshold | $\tau$ | 0.1 | Minimum relative rule spacing |
| Diversity weight | $\beta$ | 0.01 | Mild regularization |

---

## 9. Summary of Innovations (Ordered by Expected Impact)

1. **TSSFM — TSK pixel-wise fusion.** The most direct improvement: replaces MambaHSI's two global scalar weights with a content-dependent, vector-valued TSK fuzzy inference, enabling per-pixel adaptive blending and local linear transformation.
2. **FSpaMB — fuzzy Δ-modulation of Mamba state dynamics.** Adds multi-modal state-transition behavior via a low-rank fuzzy rule base that competes to set the per-token time-scale. Preserves Mamba's original selectivity as a baseline; the fuzzy contribution is warm-up-gated. Interpretability is a *post-hoc validation target*, not an upfront claim.
3. **Fuzzy Spectral Grouping + FSpeMB (restructured).** Replaces hard band-index partitioning with learnable Gaussian-membership soft grouping **on the raw spectral axis**, respecting band ordering and spectral continuity. TSK defuzzification with group firing strength $\phi_g$ provides interpretable group aggregation.
4. **End-to-end differentiable fuzzy reasoning chain.** Composition (FSpeMB) → dynamics (FSpaMB) → fusion (TSSFM) forms a three-level hierarchy addressing the three distinct sources of uncertainty in HSI classification, with overall complexity remaining $O(L)$ (constants quantified in §6).

---

### Appendix A — Mapping Between v1 and v2

| Concern (v1) | Resolution (v2) |
|---|---|
| FSpeMB grouping defined on embedding index $d$, not physical bands | Grouping moved upstream of embedding; $c_g$ lives in band-index space $[1, C]$ (§3.2) |
| FSpaMB overclaims "vegetation vs. urban" interpretability | Claim tempered; validation protocol added in §3.1.4 and §7.3 |
| Diversity regularizer unbounded in center magnitude | Replaced with hinge-style, scale-normalized penalty (§5.3) |
| Complexity discussion ignores constants | Expanded §6 with constants, memory, and wall-clock reporting plan |
| No explicit ablation plan | New §7 with core-module, design-choice, and interpretability studies |
