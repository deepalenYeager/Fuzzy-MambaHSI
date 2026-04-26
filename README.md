# Fuzzy-MambaHSI

A deep neuro-fuzzy state space model for hyperspectral image (HSI) classification.
Fuzzy-MambaHSI embeds TSK-style fuzzy inference directly into a Mamba (selective SSM)
backbone, providing explicit spectral uncertainty modeling at linear complexity in
the number of pixels.

---

## Algorithm Overview

The model addresses two core challenges in HSI classification:

- **Spectral ambiguity** — overlapping signatures and mixed pixels introduce
  inherent fuzziness that a hard classifier cannot represent faithfully.
- **Long-range spatial context at linear cost** — transformers are quadratic;
  CNNs are local; Mamba offers linear-complexity sequence modeling.

Three fuzzy modules are integrated end-to-end:

1. **Fuzzy Spectral Grouping + FSpeMB** — learnable Gaussian membership functions
   partition the *raw band axis* into G soft spectral groups. Each group is
   independently embedded and processed by Mamba over the group sequence. A
   TSK defuzzification step, weighted by each group's spectral coverage φ̄_g,
   produces the spectral output.

2. **FSpaMB (Fuzzy Spatial Mamba Block)** — a fuzzy rule base operating in a
   low-rank projection of the embedding space adds a multi-modal residual to the
   Mamba spatial output, with its contribution gated by a warm-up coefficient λ(t).

3. **TSSFM (TSK Fuzzy Spatial-Spectral Fusion Module)** — replaces the two global
   scalar weights of plain MambaHSI with pixel-wise, content-dependent TSK fusion:
   a compact antecedent vector z_ij is computed per pixel, Gaussian rule activations
   select a weighted combination of first-order linear consequents, and the result is
   added residually to the combined spatial-spectral features.

A hinge-style diversity regulariser (Eq. 22 of the design doc) keeps rule centres
separated, and a linear warm-up schedule gradually introduces the fuzzy contributions
over the first T_warm epochs.

---

## Repository Structure

```
Fuzzy-MambaHSI/
├── model/
│   ├── MambaHSI.py          # Top-level model: FuzzySpectralGrouping + encoder + head
│   └── fuzzy_modules.py     # FuzzySpectralGrouping, FSpaMB, FSpeMB, TSSFM
├── utils/
│   ├── data_load_operate.py # Dataset loading and train/val/test sampling
│   ├── Loss.py              # head_loss, diversity_loss
│   ├── evaluation.py        # OA, AA, Kappa, mIoU evaluators
│   ├── HSICommonUtils.py    # Normalisation and image stretching helpers
│   ├── setup_logger.py      # Logging setup
│   └── visual_predict.py    # Prediction visualisation
├── tests/
│   └── test_data_load_operate.py
├── train_MambaHSI.py        # Main training script
└── Fuzzy-MambaHSI_Design_v2.md  # Full algorithm design document
```

---

## Dependencies

| Package | Notes |
|---|---|
| Python ≥ 3.9 | |
| PyTorch ≥ 2.0 | GPU strongly recommended |
| `mamba-ssm` | Mamba 2 CUDA kernels; install via pip (see below) |
| `causal-conv1d` | Required by mamba-ssm for full Mamba 2 support |
| `scipy` | `.mat` file loading |
| `numpy` | |
| `scikit-learn` | Preprocessing / PCA |
| `Pillow` | Visualisation |
| `calflops` | FLOPs / parameter counting |
| `torchvision` | Transforms |

### Install

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Mamba 2 (requires matching CUDA toolkit; adjust as needed)
pip install mamba-ssm causal-conv1d

# Remaining dependencies
pip install scipy numpy scikit-learn Pillow calflops
```

> **CPU / no CUDA extension fallback.** If `causal_conv1d_cuda` cannot be loaded,
> the model automatically substitutes a pure-PyTorch gated-convolution block in
> place of Mamba 2. Training will still run, but the SSM recurrence semantics and
> associated performance benefits only hold on the CUDA path.

---

## Dataset Preparation

### Download

All four benchmark datasets are bundled in a single archive:

**[Download datasets (Google Drive)](https://drive.google.com/file/d/1d-fzMXYhpwis9o_x8hPz4uHx0z5tg7LD/view?usp=sharing)**

### Layout

Extract the archive so the data directory matches the structure expected by
`utils/data_load_operate.py`:

```
data/
├── UP/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
├── HanChuan/
│   ├── WHU_Hi_HanChuan.mat
│   └── WHU_Hi_HanChuan_gt.mat
├── HongHu/
│   ├── WHU_Hi_HongHu.mat
│   └── WHU_Hi_HongHu_gt.mat
└── Houston/
    ├── Houston.mat
    └── Houston_GT.mat
```

### Dataset index mapping

| `--dataset_index` | Dataset | Notes |
|---|---|---|
| 0 | Pavia University (UP) | 103 bands, 9 classes |
| 1 | WHU-Hi HanChuan | 274 bands, 16 classes; large image — uses tiled forward |
| 2 | WHU-Hi HongHu | 270 bands, 22 classes; large image — uses tiled forward |
| 3 | Houston 2013 | 144 bands, 15 classes; large image — uses tiled forward |

---

## Training

### Basic usage

```bash
python train_MambaHSI.py \
    --dataset_index 0 \
    --data_set_path ./data \
    --work_dir ./experiments \
    --exp_name run_UP
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset_index` | `0` | Dataset to use (see table above) |
| `--data_set_path` | `./data` | Path to the data directory |
| `--work_dir` | `./` | Root directory for output logs and weights |
| `--exp_name` | `RUNS` | Sub-folder name for this experiment |
| `--lr` | `3e-4` | Adam learning rate |
| `--max_epoch` | `200` | Total training epochs |
| `--train_samples` | `30` | Labelled samples per class for training |
| `--val_samples` | `10` | Labelled samples per class for validation |
| `--T_warm` | `15` | Warm-up epochs before full fuzzy contribution |
| `--beta_div` | `0.01` | Weight β for the diversity regulariser |
| `--tau_div` | `0.1` | Hinge threshold τ for the diversity regulariser |
| `--n_splits` | `2` | Horizontal strips for tiled inference on large images (increase to reduce peak VRAM) |
| `--use_amp` | off | Enable FP16 automatic mixed precision |

### Large-dataset example (HongHu, reduced VRAM)

```bash
python train_MambaHSI.py \
    --dataset_index 2 \
    --data_set_path ./data \
    --work_dir ./experiments \
    --exp_name run_HongHu \
    --n_splits 4 \
    --use_amp
```

### Outputs

Results are written under `<work_dir>/<exp_name>/MambaHSI/<dataset>/`:

```
run0_seed0/
├── best_tr30_val10.pth      # Best model weights (by validation OA)
├── result_tr30_val10.txt    # Per-run metrics
└── vis/                     # Prediction maps saved every 50 epochs
mean_result.txt              # Mean ± std over all 10 seeds
train_tr30_val10.log         # Full training log
```

Reported metrics: Overall Accuracy (OA), Average Accuracy (AA), Cohen's Kappa,
mean IoU, per-class accuracy, and average training / inference time.

---

## Anomaly Detection

Gradient anomaly detection is off by default (it roughly doubles peak activation
memory). Enable it for debugging:

```bash
FUZZY_MAMBAHSI_DETECT_ANOMALY=1 python train_MambaHSI.py ...
```
