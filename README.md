# Vision Transformers for *Candida* Species Identification

> **Master's Dissertation** — *AI-Driven Diagnosis of Fungal Infections using a Biological Image Dataset*  
> Department of Informatics Engineering, University of Coimbra / CISUC  
> **Continuation of:** [`simpleViT_fungi`](https://github.com/Rodrigo2003-PT/simpleViT_fungi) — (IEEE WCCI 2026)

---

## Table of Contents

- [Overview](#overview)
- [Relationship to the Intermediate Repository](#relationship-to-the-intermediate-repository)
- [Results Summary](#results-summary)
  - [Ablation Study (Cross-Validation)](#ablation-study-cross-validation)
  - [Held-Out Test Set](#held-out-test-set)
  - [External Validation (Cross-Microscope)](#external-validation-cross-microscope)
  - [XAI Findings](#xai-findings)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Reproducibility: Step-by-Step](#reproducibility-step-by-step)
  - [Shared Prerequisites — Canonical Split & Preprocessing](#shared-prerequisites--canonical-split--preprocessing)
  - [CNN — DenseNet-121](#cnn--densenet-121)
  - [ViT — ConvViT (Convolutional Tokenization ablation)](#vit--convvit-convolutional-tokenization-ablation)
  - [ViT — LsaViT (Locality Self-Attention ablation)](#vit--lsavit-locality-self-attention-ablation)
  - [ViT — HybridViT (ConvTok + LSA)](#vit--hybridvit-convtok--lsa)
  - [HybridViT — External Validation](#hybridvit--external-validation)
  - [HybridViT — XAI Analysis](#hybridvit--xai-analysis)
  - [CNN — DenseNet-121 XAI (Grad-CAM)](#cnn--densenet-121-xai-grad-cam)
- [Architecture Reference](#architecture-reference)
  - [DenseNet-121 (CNN Baseline)](#densenet-121-cnn-baseline)
  - [ConvViT — Convolutional Tokenization](#convvit--convolutional-tokenization)
  - [LsaViT — Locality Self-Attention](#lsavit--locality-self-attention)
  - [HybridViT — Combined Architecture](#hybridvit--combined-architecture)
- [Configuration Reference](#configuration-reference)
- [Key Design Decisions](#key-design-decisions)
- [Hardware and Compute](#hardware-and-compute)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [References](#references)

---

## Overview

Invasive candidiasis demands rapid, species-level discrimination of *Candida albicans* from *Candida glabrata* — two morphologically ambiguous pathogens with divergent antifungal resistance profiles. The preceding repository ([`simpleViT_fungi`](https://github.com/Rodrigo2003-PT/simpleViT_fungi)) established a **SimpleViT baseline** and forensically identified two distinct error regimes: *morphological mimicry* in False Positives and *signal attenuation* in False Negatives, both attributable to the limitations of global self-attention on small, data-scarce cohorts.

This repository provides the complete, fully reproducible codebase for the **second phase of the dissertation**, comprising:

1. **CNN Baseline (DenseNet-121)** — Locality-aware convolutional benchmark with full Grad-CAM interpretability pipeline and perturbation-based faithfulness evaluation.
2. **Ablation Study — ConvViT** — Isolated assessment of Convolutional Tokenization (Hassani et al., 2021), replacing linear patch projection with an overlapping convolutional tokenizer to inject local inductive biases at the embedding stage.
3. **Ablation Study — LsaViT** — Isolated assessment of Locality Self-Attention (Lee et al., 2021), replacing standard MSA with diagonal-masked, temperature-scaled attention to suppress self-token dominance.
4. **Hybrid Architecture (HybridViT)** — Combined ConvTok + LSA model, engineered to simultaneously address feature fragmentation and attention score skew, achieving **balanced accuracy of 0.958** on the held-out test set.
5. **External Validation** — Frozen HybridViT evaluated on an independent cross-microscope cohort to characterize out-of-distribution robustness, including photometric and geometric sensitivity analyses.
6. **XAI Audit** — Multi-method interpretability framework (UMAP, Attention Rollout, Grad-CAM, deletion/insertion faithfulness curves) applied to both the HybridViT and DenseNet-121, with principled sample selection and perturbation-based faithfulness quantification.

All architectures share the **same canonical data split** and **identical preprocessing pipeline** as the intermediate repository, ensuring that all comparative results are methodologically consistent and directly comparable to the SimpleViT baseline.

---

## Relationship to the Intermediate Repository

This repository **directly continues** the experimental pipeline of [`simpleViT_fungi`](https://github.com/Rodrigo2003-PT/simpleViT_fungi). The following elements are **shared and preserved unchanged** across both repositories to guarantee comparability:

| Shared Element | Detail |
|---------------|--------|
| **Canonical split** | `split_indices.json` (seed = 42, 60 slides, 80/20 train/test). Each `preprocessed/` folder in this repository contains a copy of the same committed file. **Do not regenerate.** |
| **Preprocessing protocol** | Identical pipeline: 12-bit linear normalisation → Lanczos-4 resize to 384 × 384 → optional BaSiCPy flatfield correction (train-only). Produces identical `float32 .npy` arrays in [0, 1]. |
| **Dataset** | Same 3,731-patch, 60-slide brightfield microscopy cohort. Zenodo DOI: [10.5281/zenodo.20595139](https://doi.org/10.5281/zenodo.20595139) |
| **Training protocol** | Same 30 × 5-fold Repeated CV, lexicographic model selection, EMA-smoothed early stopping (α = 0.3, patience = 20, δ_min = 0.01), LOCF alignment, and hierarchical statistical aggregation. |
| **Evaluation metrics** | Same metric suite: Balanced Accuracy, MCC, Weighted F1, AUC — at both patch and slide level. Slide-level: arithmetic mean-pooling of patch probabilities. CIs: cluster-robust bootstrap (B = 1,000). |
| **Positive class convention** | Label 1 = *C. glabrata*; Label 0 = *C. albicans*. |

The SimpleViT results from the intermediate repository serve as the **primary transformer baseline** against which all ablation variants are compared. Readers should consult that repository for full details on the SimpleViT architecture, its XAI analysis, and the WCCI 2026 paper.

---

## Results Summary

### Ablation Study (Cross-Validation)

30 × 5-fold Repeated CV. Mean ± Std (between-iteration), 95% CI via t_{N−1} (N = 30). Slide-level Balanced Accuracy is the primary optimisation criterion.

| Architecture | Slide-Level BA | Patch-Level BA | Slide-Level MCC | Slide-Level AUC |
|---|---|---|---|---|
| SimpleViT (baseline†) | 0.9117 ± 0.0374 | 0.8287 ± 0.0330 | 0.8319 ± 0.0715 | 0.9618 ± 0.0340 |
| DenseNet-121 (CNN) | 1.0000 ± 0.0000 | 0.9965 ± 0.0021 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| ConvViT | 0.9706 ± 0.0137 | 0.9562 ± 0.0128 | 0.9421 ± 0.0269 | 0.9921 ± 0.0160 |
| LsaViT | 0.9048 ± 0.0442 | 0.8247 ± 0.0316 | 0.8169 ± 0.0876 | 0.9643 ± 0.032 |
| **HybridViT** | 0.9762 ± 0.0126 | 0.9546 ± 0.0132 | 0.9536 ± 0.0249 | 0.9933 ± 0.0102 |

*† From [`simpleViT_fungi`](https://github.com/Rodrigo2003-PT/simpleViT_fungi).*

### Held-Out Test Set

12 slides, 821 patches. 95% CIs via cluster-robust bootstrap (B = 1,000).

| Architecture | Slide-Level BA | Patch-Level BA | Patch-Level Errors | Slide-Level MCC |
|---|---|---|---|---|
| SimpleViT (baseline†) | 0.9286 [0.786, 1.000] | 0.8439 [0.745, 0.897] | 136 | 0.8452 [0.529, 1.000] |
| DenseNet-121 (CNN) | 1.0000 [1.000, 1.000] | 0.9964 [0.991, 1.000]  | 3 | 1.0000 [1.000, 1.000] |
| ConvViT | 0.9285 [0.785, 1.000] | 0.9568 [0.846, 0.998]  | 38 | 0.8451 [0.529, 1.000] |
| LsaViT | 0.9286 [0.786, 1.000] | 0.8445 [0.742, 0.904] | 135 | 0.8452 [0.529, 1.000] |
| **HybridViT** | 0.9285 [0.777, 1.000] | 0.9576 [0.852, 0.998]  | 37 | 0.8451 [0.522, 1.000] |

*† From [`simpleViT_fungi`](https://github.com/Rodrigo2003-PT/simpleViT_fungi). The reduction from 136 → 37 total patch-level errors represents the primary diagnostic improvement of the HybridViT.*

### External Validation (Cross-Microscope)

Frozen HybridViT evaluated on an independent cohort (24 slides, different microscope). Three inference pipelines reported.

| Pipeline | Scientific Role | Slide-Level BA | MCC | AUROC | Rec. *albicans* | Rec. *glabrata* |
|---|---|---|---|---|---|---|
| **Locked-pixel** | Primary (unadapted) | 0.5417 | 0.2085 | 0.7153 | 1.0000 | 0.0833 |
| Scale-matched | Post-hoc spatial sensitivity | 0.5000 | 0.0000 | 0.2778 | 1.0000 | 0.0000 |
| Photometric | Post-hoc radiometric sensitivity | 0.8333 | 0.6761 | 0.9306 | 0.7500 | 0.9167 |

The photometric harmonization result (BA = 0.8333) is a **post-hoc adaptation** using test-image statistics and does not represent unbiased deployment readiness.

### XAI Findings

| Model | Method | Key Finding |
|---|---|---|
| HybridViT | Attention Rollout + Deletion/Insertion | Improved latent class separation vs SimpleViT; focal, deletion-sensitive attribution for TP and FP predictions. Faithfulness is outcome-dependent. |
| DenseNet-121 | Grad-CAM + Faithfulness curves | Strong localized faithfulness for *C. glabrata* (AUIC Δ = +0.411). Diffuse, spatially redundant evidence for *C. albicans* consistent with peripheral/photometric shortcuts. Aggregate deletion test: Wilcoxon p ≈ 0.0002 (Bonferroni adj. p ≈ 0.0034). |

---

## Dataset

The dataset is **identical** to the one used in the intermediate repository. See [`simpleViT_fungi`](https://github.com/Rodrigo2003-PT/simpleViT_fungi) for full acquisition protocol details.

| Class | Slides | Train Patches | Test Patches | Total |
|---|---|---|---|---|
| *C. albicans* | 33 | 1,616 | 440 | 2,056 |
| *C. glabrata* | 27 | 1,294 | 381 | 1,675 |
| **Total** | **60** | **2,910** | **821** | **3,731** |

**Acquisition summary:**

| Parameter | Value |
|---|---|
| Microscope | Zeiss Axio Observer |
| Objective | Plan-Apochromat 63×/1.4 Oil DIC |
| Camera | Prime 95B monochrome (12-bit) |
| Pixel spacing | 0.175 µm/px |
| Illumination | Brightfield (TL LED, 10% intensity) |
| Mounting | 1.7% agarose pad |
| Culture | Synthetic Complete (pH 6.5), 35°C, OD₆₀₀ = 3.0 |
| FOVs/slide | ≈ 62 (random, fixed focus) |

**Data availability:** [https://doi.org/10.5281/zenodo.20595139](https://doi.org/10.5281/zenodo.20595139)

The external validation cohort (24 slides) was acquired on a different microscope platform with a pixel spacing of 0.043 µm/px, a 4× difference in physical scale relative to the training domain.

---

## Repository Structure

```
fungi-hybrid-vit/
│
├── README.md
├── LICENSE                                  # MIT
├── .gitignore
├── requirements.txt
│
├── CNN/                                     # ── CNN BASELINE ────────────────────────────────────
│   └── DenseNet121/
│       ├── preprocessed/
│       │   ├── split_indices.json           # Canonical split — identical to intermediate repo
│       │   └── fungi.zip                    # Preprocessed .npy dataset (384×384, float32)
│       │
│       ├── scripts/
│       │   ├── densenet_fungi.ipynb         # PRIMARY ARTIFACT — Colab training notebook
│       │   ├── config.py                    # DenseNet-121 hyperparameters
│       │   ├── densenet_model.py            # DenseNet-121 architecture definition
│       │   ├── training_logic.py            # ModelTrainer, CV loop, optimal epoch determination
│       │   ├── training_utils.py            # NPYImageFolder, slide-level CV folds, bootstrap CI
│       │   ├── checkpoints.py               # CheckpointManager: atomic saves, full resumability
│       │   └── plotting_utils.py            # Training curves, confusion matrix, ROC/AUC
│       │
│       ├── statistics/
│       │   └── cv_statistics/               # Hierarchical CV stats outputs (generated)
│       │
│       └── XAI/
│           ├── scripts/
│           │   ├── densenet_xai.ipynb       # Grad-CAM XAI entry point (Colab notebook)
│           │   ├── gradcam_engine.py        # Grad-CAM forward/backward pass implementation
│           │   ├── gradcam_analysis.py      # Main XAI orchestrator: sample selection + analysis
│           │   ├── gradcam_data_utils.py    # Data loading, patch/slide provenance mapping
│           │   ├── gradcam_faithfulness.py  # Deletion/insertion perturbation curves + AUC
│           │   ├── gradcam_visualization.py # Heatmap overlays, mean CAM plots, curve figures
│           │   └── test_gradcam/            # Sanity-check outputs (generated)
│           └── (outputs generated at runtime)
│
├── ViT/                                     # ── VISION TRANSFORMER ABLATIONS ───────────────────
│   │
│   ├── ConvViT/                             # Ablation 1: Convolutional Tokenization only
│   │   ├── preprocessed/
│   │   │   ├── split_indices.json           # Canonical split — identical copy
│   │   │   └── fungi.zip
│   │   ├── scripts/
│   │   │   ├── vit_fungi.ipynb              # PRIMARY ARTIFACT — Colab training notebook
│   │   │   ├── config.py                    # ConvViT hyperparameters
│   │   │   ├── convViT.py                   # Convolutional tokenizer + ViT encoder
│   │   │   ├── training_logic.py            # ModelTrainer (shared protocol)
│   │   │   ├── training_utils.py            # NPYImageFolder, slide-level CV folds, bootstrap CI
│   │   │   ├── checkpoints.py               # CheckpointManager
│   │   │   └── plotting_utils.py
│   │   └── statistics/
│   │       └── cv_statistics/               # CV stats outputs (generated)
│   │
│   ├── LsaViT/                              # Ablation 2: Locality Self-Attention only
│   │   ├── preprocessed/
│   │   │   ├── split_indices.json           # Canonical split — identical copy
│   │   │   └── fungi.zip
│   │   ├── scripts/
│   │   │   ├── vit_fungi.ipynb              # PRIMARY ARTIFACT — Colab training notebook
│   │   │   ├── config.py                    # LsaViT hyperparameters
│   │   │   ├── LSA_ViT.py                   # Standard tokenizer + LSA attention module
│   │   │   ├── lsa_logger.py                # LSA-specific attention diagnostics (τ, entropy)
│   │   │   ├── lsa_sanity_checks.py         # Diagonal mask and temperature scale validation
│   │   │   ├── training_logic.py
│   │   │   ├── training_utils.py
│   │   │   ├── checkpoints.py
│   │   │   └── plotting_utils.py
│   │   └── statistics/
│   │       └── cv_statistics/
│   │
│   └── HybridViT/                           # Full Hybrid: ConvTok + LSA ─────────────────────
│       ├── preprocessed/
│       │   ├── split_indices.json           # Canonical split — identical copy
│       │   └── fungi.zip
│       │
│       ├── scripts/
│       │   ├── vit_fungi.ipynb              # PRIMARY ARTIFACT — Colab training notebook
│       │   ├── config.py                    # HybridViT hyperparameters
│       │   ├── convViT.py                   # Convolutional tokenizer (shared with ConvViT)
│       │   ├── lsa_attention.py             # LSA module (diagonal mask + learnable temperature)
│       │   ├── lsa_convtok_vit.py           # Full Hybrid architecture definition
│       │   ├── lsa_logger.py                # LSA diagnostics
│       │   ├── training_logic.py
│       │   ├── training_utils.py
│       │   ├── checkpoints.py
│       │   └── plotting_utils.py
│       │
│       ├── statistics/
│       │   └── cv_statistics/               # CV stats outputs (generated)
│       │
│       ├── XAI/                             # HybridViT XAI (Attention Rollout + UMAP)
│       │   ├── Run.ipynb                    # XAI entry point (Colab notebook)
│       │   └── scripts/
│       │       ├── analysis_attention.py    # Main XAI orchestrator (CLI)
│       │       ├── extractor.py             # Embeddings + Attention Rollout for HybridViT
│       │       ├── embeddings.py            # UMAP projection, trustworthiness, permutation test
│       │       ├── stratified_sampling.py   # Principled sample selection (Kim et al., 2016)
│       │       ├── visualization.py         # Heatmaps, rollout overlays, slide-conditioned plots
│       │       ├── helper.py                # Slide ID parsing, provenance mapping
│       │       ├── config.py                # Self-contained XAI config (copy of scripts/config.py)
│       │       ├── lsa_attention.py         # Copy for self-contained XAI inference
│       │       ├── lsa_convtok_vit.py       # Copy for self-contained XAI inference
│       │       └── training_utils.py        # Copy for self-contained XAI inference
│       │
│       └── validation/                      # Cross-Microscope External Validation
│           ├── Run.ipynb                    # Validation entry point (Colab notebook)
│           │
│           ├── models/
│           │   ├── final_model_lsa_convtok.pth  # Final production model (trained on full S_train)
│           │
│           ├── data/
│           │   └── fungi.zip                # External cohort preprocessed images
│           │
│           ├── scripts/
│           │   ├── preprocess_external      # External TIFF → .npy (locked-pixel pipeline)
│           │   ├── external_manifest        # Slide manifest builder for the external cohort
│           │   ├── load_frozen_hybrid       # Checkpoint loading + frozen inference utilities
│           │   ├── evaluate_external_generalization  # Primary locked-pixel evaluation
│           │   ├── compare_external_pipelines        # Integrated comparison (Table in thesis)
│           │   ├── audit_external_tiffs     # Raw image audit: intensity statistics, domain shift
│           │   ├── run_external_validation_colab_local  # CLI orchestrator (Colab/local)
│           │   └── external_statistics      # Bootstrap CI computation for external metrics
│           │
│           ├── configs/
│           │   ├── external_validation_config.yaml   # Preprocessing and inference configuration
│           │   └── external_validation_metadata.json # External cohort provenance metadata
│           │
│           └── source/                      # Self-contained source copies for inference
│               ├── config.py
│               ├── convViT.py
│               ├── lsa_attention.py
│               ├── lsa_convtok_vit.py
│               ├── lsa_logger.py
│               ├── training_logic.py
│               ├── training_utils.py
│               └── checkpoints.py
```

> **Note on self-contained source copies.** The `HybridViT/XAI/scripts/` and `HybridViT/validation/source/` directories contain intentional copies of architecture and utility files from `HybridViT/scripts/`. This design allows both sub-pipelines to run independently on Google Colab without requiring `PYTHONPATH` manipulation. If you modify the canonical files in `HybridViT/scripts/`, update all copies accordingly to maintain consistency.

> **Note on `split_indices.json` copies.** Each `preprocessed/` directory contains an independent copy of the same canonical split file. They are byte-for-byte identical. The authoritative source is the file committed to the intermediate repository (`simpleViT_fungi`). Do **not** regenerate any of these files.

---

## Installation

### System Requirements

- Python ≥ 3.10
- **Training and XAI:** CUDA-capable GPU (NVIDIA A100 recommended via Google Colab Pro+)
- **Preprocessing and statistics:** CPU (any)

### Core Dependencies

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Training, inference, DenseNet-121 |
| `einops` | Tensor operations in ViT variants |
| `basicpy` | BaSiCPy illumination correction (preprocessing only) |
| `scikit-learn` | CV folds, metrics, bootstrap |
| `umap-learn` | UMAP projection (XAI only) |
| `opencv-python` | Lanczos resampling |
| `matplotlib`, `seaborn` | Visualisation |
| `scipy` | Wilcoxon test, KDE |
| `pyyaml` | Configuration loading |

**Note on BaSiCPy.** `basicpy` requires `jax` as a computational backend. If installation fails on your platform, consult the [BaSiCPy installation guide](https://github.com/peng-lab/BaSiCPy). The `baseline` preprocessing method (linear normalisation only) works without BaSiCPy and is sufficient to reproduce all reported results.

---

## Reproducibility: Step-by-Step

### Shared Prerequisites — Canonical Split & Preprocessing

**All four models share the same data split and preprocessing outputs.** These steps need to be run only once; the resulting `split_indices.json` and `fungi.zip` are then placed into each model's `preprocessed/` folder.

#### Step P1 — Canonical Data Split

This step is **already completed**. The `split_indices.json` committed to each `preprocessed/` directory in this repository is the **canonical split used in all reported results** — the same file produced in Step 1 of the intermediate repository. Do not regenerate it.

If you need to inspect the split algorithm, refer to `data/canonical_splits.py` in [`simpleViT_fungi`](https://github.com/Rodrigo2003-PT/simpleViT_fungi). The procedure performs a constrained randomised search (up to 2,000 attempts, seed = 42) to simultaneously satisfy slide-level class stratification and image-level global ratio matching within ±2% tolerance.

```
split_indices.json  ← committed; reproducibility anchor for all models in this repository
```

#### Step P2 — Preprocessing

Preprocessing is **identical** to the procedure documented in the intermediate repository. Raw 12-bit TIFF images are converted to normalised `float32 .npy` arrays at 384 × 384 pixels. Refer to [`simpleViT_fungi`](https://github.com/Rodrigo2003-PT/simpleViT_fungi) Step 2 for the full CLI reference, argument descriptions, and quality metric definitions.

Two methods are available:

| Method | Description | Recommended for |
|---|---|---|
| `baseline` | Linear normalisation only: `I_norm = I_raw / 4095` → [0, 1] | All paper results |
| `optimized` | Baseline + BaSiCPy flatfield/darkfield correction | Ablation only |

BaSiCPy is fitted **exclusively on training images** (identified via `split_indices.json`) to strictly prevent test-set signal from entering the illumination correction model.

After preprocessing, compress the output directory into `fungi.zip` and copy both the archive and `split_indices.json` into each model's `preprocessed/` folder:

```
CNN/DenseNet121/preprocessed/fungi.zip
CNN/DenseNet121/preprocessed/split_indices.json

ViT/ConvViT/preprocessed/fungi.zip
ViT/ConvViT/preprocessed/split_indices.json

ViT/LsaViT/preprocessed/fungi.zip
ViT/LsaViT/preprocessed/split_indices.json

ViT/HybridViT/preprocessed/fungi.zip
ViT/HybridViT/preprocessed/split_indices.json
```

**Output format.** Each `.npy` file is a `(384, 384)` `float32` array with values in [0, 1], compatible with the `NPYImageFolder` data loader in all `training_utils.py` files across this repository.

---

### CNN — DenseNet-121

#### Google Drive Setup

```
MyDrive/colab/densenet_fungi/
│
├── preprocessed/
│   ├── fungi.zip               ← output of Step P2
│   └── split_indices.json      ← canonical split
│
├── scripts/                    ← upload all .py files from CNN/DenseNet121/scripts/
│   ├── config.py
│   ├── densenet_model.py
│   ├── training_logic.py
│   ├── training_utils.py
│   ├── checkpoints.py
│   └── plotting_utils.py
│
├── output/densenet/            ← created automatically
└── checkpoints/densenet/       ← created automatically
```

Open `CNN/DenseNet121/scripts/densenet_fungi.ipynb` in Google Colab and execute all cells. The notebook implements the same four-phase resumable pipeline as the intermediate repository (30 × 5-fold CV → optimal epoch determination → final training on full cohort → held-out test evaluation).

**Key outputs** (in `MyDrive/colab/densenet_fungi/output/densenet/`):

| File | Description |
|---|---|
| `final_model_densenet.pth` | Trained model with config and normalisation stats embedded |
| `cv_results_densenet.csv` | One row per fold: all metrics at the best epoch |
| `checkpoints/fold_histories/` | Per-fold training histories (JSON) for LOCF curve generation |
| `test_results_densenet.json` | Full test evaluation with bootstrap CIs and per-class metrics |
| `config_densenet.json` | Exact hyperparameter config used |

#### CV Statistics

```bash
python CNN/DenseNet121/statistics/cv_statistics.py \
    --csv-path      /path/to/cv_results_densenet.csv \
    --history-dir   /path/to/fold_histories \
    --output-dir    ./CNN/DenseNet121/statistics/cv_statistics \
    --n-iterations  30 \
    --n-folds       5
```

The hierarchical aggregation protocol is identical to that documented in the intermediate repository: fold scores within each iteration are averaged first (collapsing within-iteration correlation), then statistics are computed across the 30 iteration means using t_{N−1} CIs.

---

### ViT — ConvViT (Convolutional Tokenization ablation)

This ablation **replaces the linear patch projection** of SimpleViT with a convolutional tokenizer (Hassani et al., 2021) while retaining standard MSA. This isolates the contribution of local inductive biases at the embedding stage.

#### Google Drive Setup

```
MyDrive/colab/convvit_fungi/
│
├── preprocessed/
│   ├── fungi.zip
│   └── split_indices.json
│
├── scripts/                    ← upload all .py files from ViT/ConvViT/scripts/
│   ├── config.py
│   ├── convViT.py
│   ├── training_logic.py
│   ├── training_utils.py
│   ├── checkpoints.py
│   └── plotting_utils.py
│
├── output/convvit/
└── checkpoints/convvit/
```

Open `ViT/ConvViT/scripts/vit_fungi.ipynb` in Colab. Training protocol is identical to all other models.

**Key outputs** (in `MyDrive/colab/convvit_fungi/output/convvit/`):

| File | Description |
|---|---|
| `final_model_convvit.pth` | Trained model checkpoint |
| `cv_results_convvit.csv` | Per-fold metrics |
| `test_results_convvit.json` | Held-out test evaluation |

---

### ViT — LsaViT (Locality Self-Attention ablation)

This ablation **retains the standard linear patch tokenization** of SimpleViT but replaces the standard MSA with Locality Self-Attention (Lee et al., 2021). This isolates the contribution of diagonal masking and learnable temperature scaling on attention score distributions.

#### LSA-Specific Diagnostics

`lsa_logger.py` logs the learned temperature parameter τ and attention entropy across training epochs, providing evidence that the LSA mechanism converges toward sharper attention distributions. `lsa_sanity_checks.py` validates at initialisation that (1) the diagonal of the pre-softmax attention matrix is correctly masked to −∞, and (2) the temperature parameter τ is initialised to `ln(d_k^{−0.5})` to match the standard scale.

#### Google Drive Setup

```
MyDrive/colab/lsavit_fungi/
│
├── preprocessed/
│   ├── fungi.zip
│   └── split_indices.json
│
├── scripts/                    ← upload all .py files from ViT/LsaViT/scripts/
│   ├── config.py
│   ├── LSA_ViT.py
│   ├── lsa_logger.py
│   ├── lsa_sanity_checks.py
│   ├── training_logic.py
│   ├── training_utils.py
│   ├── checkpoints.py
│   └── plotting_utils.py
│
├── output/lsavit/
└── checkpoints/lsavit/
```

Open `ViT/LsaViT/scripts/vit_fungi.ipynb` in Colab.

**Key outputs** (in `MyDrive/colab/lsavit_fungi/output/lsavit/`):

| File | Description |
|---|---|
| `final_model_lsavit.pth` | Trained model checkpoint |
| `cv_results_lsavit.csv` | Per-fold metrics |
| `test_results_lsavit.json` | Held-out test evaluation |

---

### ViT — HybridViT (ConvTok + LSA)

The full Hybrid architecture combines both mechanisms: the **Convolutional Tokenizer** from ConvViT and **Locality Self-Attention** from LsaViT. This is the primary proposed architecture of the dissertation.

#### Google Drive Setup

```
MyDrive/colab/hybridvit_fungi/
│
├── preprocessed/
│   ├── fungi.zip
│   └── split_indices.json
│
├── scripts/                    ← upload all .py files from ViT/HybridViT/scripts/
│   ├── config.py
│   ├── convViT.py
│   ├── lsa_attention.py
│   ├── lsa_convtok_vit.py
│   ├── lsa_logger.py
│   ├── training_logic.py
│   ├── training_utils.py
│   ├── checkpoints.py
│   └── plotting_utils.py
│
├── output/hybridvit/
└── checkpoints/hybridvit/
```

Open `ViT/HybridViT/scripts/vit_fungi.ipynb` in Colab. The final trained production model (`final_model_lsa_convtok.pth`) is committed to `ViT/HybridViT/validation/models/` for direct use in the external validation and XAI pipelines.

**Key outputs** (in `MyDrive/colab/hybridvit_fungi/output/hybridvit/`):

| File | Description |
|---|---|
| `final_model_lsa_convtok.pth` | Production model — used in external validation and XAI |
| `final_model_checkpoint.pth` | Best CV checkpoint (reference) |
| `cv_results_hybridvit.csv` | Per-fold metrics |
| `test_results_hybridvit.json` | Held-out test evaluation |

---

### HybridViT — External Validation

Evaluates the **frozen** `final_model_lsa_convtok.pth` on an independent cross-microscope cohort. The model weights are not updated at any point during this evaluation.

The external cohort was acquired on a different microscope platform (pixel spacing: 0.043 µm/px vs. 0.175 µm/px for the training domain), representing a genuine acquisition domain shift. Three inference pipelines are implemented:

| Script | Pipeline | Scientific Role |
|---|---|---|
| `evaluate_external_generalization` | Locked-pixel | Primary evaluation — no domain adaptation |
| `compare_external_pipelines` (scale branch) | Scale-matched | Post-hoc: downsamples external images to restore training-equivalent physical scale |
| `compare_external_pipelines` (photometric branch) | Photometric | Post-hoc: affine shift to match training μ/σ (μ_train ≈ 0.2369, σ_train ≈ 0.0074) |

> **Critical methodological note.** The photometric harmonization pipeline uses the statistical properties of the test images themselves to normalize inputs. It therefore functions as an **unsupervised test-domain adaptation** and the resulting metrics do not represent unbiased deployment readiness. The **locked-pixel pipeline is the scientifically valid primary result.**

#### Preprocessing the External Cohort

```bash
python ViT/HybridViT/validation/scripts/preprocess_external \
    --tiff-dir   /path/to/external_tiff_cohort \
    --output     ViT/HybridViT/validation/data/fungi.zip \
    --image-size 384 \
    --bit-depth  12
```

**No BaSiCPy correction is applied** to the external cohort. The locked-pixel pipeline applies only linear normalisation (`I_norm = I_raw / 4095`) to preserve the raw acquisition domain as closely as possible. Applying training-domain correction parameters to the external cohort would constitute a form of domain adaptation and would contaminate the primary evaluation.

#### Running the Primary Evaluation

Open `ViT/HybridViT/validation/Run.ipynb` in Colab, or use the CLI orchestrator:

```bash
python ViT/HybridViT/validation/scripts/run_external_validation_colab_local \
    --config     ViT/HybridViT/validation/configs/external_validation_config.yaml \
    --model      ViT/HybridViT/validation/models/final_model_lsa_convtok.pth \
    --data       ViT/HybridViT/validation/data/fungi.zip \
    --output-dir ./outputs/external_validation
```

The configuration file `external_validation_config.yaml` specifies inference parameters. The cohort provenance (slide identifiers, acquisition metadata) is documented in `external_validation_metadata.json`.

**Raw image audit.** Before running inference, run `audit_external_tiffs` to compute per-image intensity statistics (mean, std, percentiles) for the external cohort. This step quantifies the photometric offset between the external domain and the training domain, providing the empirical basis for the photometric sensitivity analysis.

**Key outputs:**

| File | Description |
|---|---|
| `external_results_locked.json` | Primary evaluation: BA, MCC, AUROC, per-class recall + bootstrap CIs |
| `external_results_scale.json` | Scale-matched sensitivity |
| `external_results_photometric.json` | Photometric sensitivity |
| `integrated_comparison_table.csv` | Table as reported in Chapter 5 |
| `audit_intensity_stats.json` | Domain shift quantification |

---

### HybridViT — XAI Analysis

The XAI pipeline for the HybridViT deploys UMAP representation-space analysis, Attention Rollout (adapted for the HybridViT's pooling architecture), and deletion/insertion perturbation-based faithfulness curves.

#### Google Drive Setup

```
MyDrive/colab/hybridvit_xai/
│
├── model/
│   └── final_model_lsa_convtok.pth   ← production model from HybridViT training
│
├── preprocessed/
│   ├── fungi.zip
│   └── split_indices.json
│
└── xai_scripts/                      ← upload all .py files from ViT/HybridViT/XAI/scripts/
```

Open `ViT/HybridViT/XAI/Run.ipynb` in Colab, or run via CLI:

```bash
python ViT/HybridViT/XAI/scripts/analysis_attention.py \
    --preprocessed-root   /path/to/preprocessed \
    --output-root         ./outputs/hybridvit_xai \
    --split-file          split_indices.json \
    --model-path          /path/to/final_model_lsa_convtok.pth \
    --split               test \
    --extract-attention \
    --run-embedding-analysis \
    --compute-attention-rollout \
    --rollout-discard-ratio 0.9
```

#### Attention Rollout Adaptation for HybridViT

The HybridViT uses **Global Average Pooling** (no CLS token), identical to SimpleViT. The Attention Rollout implementation in `extractor.py` applies the same GAP adaptation described in the intermediate repository: rather than extracting the CLS-token row from the final rollout matrix, patch relevance is computed by summing propagated flow across all query positions (`rollout.sum(axis=1)`), producing a global relevance vector that is reshaped to the spatial patch grid for visualisation.

The attention hook implementation in `extractor.py` must be verified against the internal API of `lsa_convtok_vit.py` (specifically the LSA attention module's attribute names for Q, K, V projections and the temperature parameter τ) before extracting rollout maps.

#### Principled Sample Selection

`stratified_sampling.py` implements the sample selection protocol of Kim et al. (2016): for each prediction outcome group (TP, TN, FP, FN), a representative patch is selected as the one closest to the group's median Gini focality coefficient. This ensures that the displayed attention heatmaps are statistically representative of their group rather than cherry-picked.

**Key outputs** (in `outputs/hybridvit_xai/`):

| File | Thesis equivalent |
|---|---|
| `umap_comprehensive_test.png` | Latent space topology (Chapter 6) |
| `rollout_stats_test.json` | Attention focality statistics |
| `rep_TP_XXXX_rollout_test.png` | Representative TP attribution map |
| `rep_FP_XXXX_rollout_test.png` | Representative FP attribution map |
| `faithfulness_curves_hybrid.png` | Deletion/insertion perturbation curves |

---

### CNN — DenseNet-121 XAI (Grad-CAM)

The Grad-CAM pipeline for DenseNet-121 targets the **final convolutional layer** to produce class-discriminative localization maps, paired with deletion/insertion perturbation curves to quantify faithfulness.

#### Running the Analysis

Open `CNN/DenseNet121/XAI/scripts/densenet_xai.ipynb` in Colab, or use the CLI:

```bash
python CNN/DenseNet121/XAI/scripts/gradcam_analysis.py \
    --model-path        /path/to/final_model_densenet.pth \
    --preprocessed-root /path/to/preprocessed \
    --split-file        split_indices.json \
    --split             test \
    --output-dir        ./outputs/densenet_xai
```

#### Faithfulness Protocol

`gradcam_faithfulness.py` implements the deletion and insertion perturbation protocol of Samek et al. (2017). For each test patch, pixels are masked in descending order of Grad-CAM saliency (deletion) or revealed in descending order (insertion), and model confidence is recorded at each masking step. The mean-value baseline (patch mean intensity) is used as the mask fill value. The area under the deletion curve (AUDC) and insertion curve (AUIC) are computed per patch and aggregated at the slide level via Wilcoxon signed-rank test (Bonferroni-corrected for multiple comparisons).

**Mean CAM visualization.** `gradcam_visualization.py` computes class-stratified mean Grad-CAM maps across all correctly classified test patches, providing a spatial summary of the network's aggregate localization behavior per outcome group.

**Key outputs** (in `outputs/densenet_xai/`):

| File | Thesis equivalent |
|---|---|
| `mean_cam_glabrata.png` | Mean CAM for *C. glabrata* (Chapter 6) |
| `mean_cam_albicans.png` | Mean CAM for *C. albicans* (Chapter 6) |
| `faithfulness_curves.png` | Deletion/insertion curves by class |
| `faithfulness_auc_summary.json` | AUDC, AUIC, Δ per class (Table in Chapter 6) |
| `wilcoxon_results.json` | Slide-level statistical test results |

---

## Architecture Reference

### DenseNet-121 (CNN Baseline)

DenseNet-121 (Huang et al., 2017) connects every layer to all subsequent layers via channel-wise concatenation (`x_l = H_l([x_0, ..., x_{l-1}])`), maximizing gradient flow and feature reuse. It serves as the locality-aware CNN benchmark, establishing the upper bound of what local inductive biases alone can achieve on this cohort. The architecture is loaded from `torchvision.models.densenet121` with the first convolution modified to accept single-channel (grayscale) input and the classifier head replaced with a two-class linear layer.

### ConvViT — Convolutional Tokenization

Replaces the linear patch projection of SimpleViT with a convolutional tokenizer (Hassani et al., 2021). The tokenizer processes the input through `L` sequential blocks, each executing `MaxPool(ReLU(Conv2D(·)))`, progressively mapping the channel depth to the embedding dimension `D` while shrinking the spatial grid. The resulting token sequence `T ∈ ℝ^{N×D}` is fed into the standard transformer encoder unchanged. Importantly, the effective sequence length `N = H' × W'` is determined by the tokenizer's stride and padding configuration, directly controlling the quadratic self-attention cost.

Sinusoidal positional embeddings from SimpleViT are **retained**. Standard MSA (no diagonal mask, fixed `1/√d_k` scaling) is used, isolating the contribution of the convolutional tokenizer.

### LsaViT — Locality Self-Attention

Retains the standard linear patch projection and sinusoidal positional embeddings of SimpleViT, but replaces MSA with Locality Self-Attention (Lee et al., 2021). LSA modifies scaled dot-product attention through two interventions:

1. **Diagonal masking** — A static mask `M_diag^{(i,j)} = −∞ if i = j, 0 otherwise` is added to the pre-softmax logits, driving self-relation values to zero after activation and forcing each token to redistribute attention toward non-self tokens.

2. **Learnable temperature scaling** — The fixed `1/√d_k` scale is replaced by `exp(τ)`, where τ is a learnable scalar initialized to `ln(d_k^{−0.5})`. This allows the optimization algorithm to dynamically sharpen or flatten the attention distribution during training.

```
Attention(Q, K, V) = Softmax(exp(τ)(QK^T) + M_diag) V
```

`lsa_sanity_checks.py` verifies at initialization that the diagonal of the pre-softmax matrix is correctly masked and that τ is initialized to the correct value.

### HybridViT — Combined Architecture

Combines both mechanisms: the **Convolutional Tokenizer** from ConvViT processes the input into overlapping local embeddings, which are then passed to the transformer encoder equipped with **Locality Self-Attention** in every attention block. The architecture is defined in `lsa_convtok_vit.py`, with the LSA module separated into `lsa_attention.py` for modularity and reuse across the XAI and validation pipelines.

---

## Configuration Reference

All hyperparameters are centralised in each model's `config.py`. The training protocol parameters (CV iterations, folds, early stopping, learning rate schedule, augmentation) are **identical across all four models** to ensure fair comparison. Architecture-specific parameters differ only in the tokenization and attention modules.

### Shared Training Protocol

| Parameter | Value | Description |
|---|---|---|
| `random_seed` | 42 | Global base seed (SeedSequence root) |
| `image_size` | 384 | Input resolution (px) |
| `n_iterations` | 30 | Independent CV iterations |
| `n_folds` | 5 | Folds per iteration |
| `num_epochs` | 100 | Maximum training epochs per fold |
| `patience` | 20 | Early stopping patience (epochs) |
| `ema_alpha` | 0.3 | EMA smoothing factor for early stopping signal |
| `min_delta` | 0.01 | Minimum EMA improvement to reset patience |
| `batch_size` | 64 | Training batch size |
| `learning_rate` | 3×10⁻⁴ | AdamW base learning rate |
| `weight_decay` | 0.05 | AdamW weight decay |
| `warmup_epochs` | 5 | Linear LR warmup duration |
| `gradient_clip_norm` | 1.0 | Gradient clipping max norm |
| `use_amp` | True | Automatic Mixed Precision (CUDA only) |

### Shared Augmentation Protocol

| Parameter | Value |
|---|---|
| `random_resized_crop_scale` | (0.9, 1.0) |
| `random_resized_crop_ratio` | (0.95, 1.05) |
| `horizontal_flip_p` | 0.5 |
| `vertical_flip_p` | 0.5 |
| `rotation_degrees` | 30 |
| `use_mixup` | True |
| `mixup_alpha` | 0.2 |
| `mixup_prob` | 0.5 |

### ViT Architecture Parameters (All ViT Variants)

| Parameter | Value | Description |
|---|---|---|
| `dim` | 256 | Transformer embedding dimension D |
| `depth` | 8 | Number of encoder layers |
| `heads` | 6 | Number of attention heads |
| `mlp_dim` | 512 | Feed-forward network dimension |
| `channels` | 1 | Input channels (grayscale) |

| Model | `patch_size` | Tokenizer | Attention |
|---|---|---|---|
| SimpleViT (baseline) | 16 → 576 tokens | Linear projection | Standard MSA |
| ConvViT | Determined by conv stride | Convolutional (L blocks) | Standard MSA |
| LsaViT | 16 → 576 tokens | Linear projection | LSA (diagonal mask + learnable τ) |
| HybridViT | Determined by conv stride | Convolutional (L blocks) | LSA (diagonal mask + learnable τ) |

---

## Key Design Decisions

### 1 — Identical Canonical Split Across All Models

Every model in this repository uses the same `split_indices.json` produced in the intermediate repository. This is a **hard requirement** for the ablation study: any performance difference between architectures must be attributable to architectural choices, not to data partition variance. The split file is committed to each `preprocessed/` directory and must not be regenerated.

### 2 — Identical Preprocessing Across All Models

All four models consume the same `float32 .npy` arrays produced by the shared preprocessing pipeline. Per-fold normalisation statistics (mean and std for Z-score standardisation) are computed strictly from training-fold image indices within each fold, preventing any signal from the validation or test partitions from entering the standardisation parameters.

### 3 — Slide-Level Data Leakage Prevention

All patches from a single biological slide are assigned exclusively to either the training or test partition. This constraint is enforced at every layer of the pipeline:
- `split_indices.json` — search operates at the slide level; images follow atomically
- `create_slide_level_folds()` — StratifiedKFold applied at slide level, mapped to image indices; integrity validated per fold
- Per-fold normalisation — statistics computed only from training-fold image indices
- BaSiCPy (if used) — fitted only on training TIFFs identified by `split_indices.json`

### 4 — Frozen Evaluation for External Validation

The HybridViT model is evaluated on the external cohort with **strictly frozen weights**. No fine-tuning, adaptation, or parameter update of any kind occurs during external validation. The locked-pixel pipeline applies only the preprocessing steps established during training (linear normalisation, Lanczos resize) without any domain-specific calibration.

### 5 — Two-Level Hierarchical Statistical Aggregation

Following Bengio & Grandvalet (2003): fold scores within each iteration are averaged first (collapsing within-iteration correlation), then statistics are computed across the 30 iteration means using t_{N−1} CIs. Treating all 150 folds as independent would overestimate precision by √5 ≈ 2.24 and is scientifically incorrect.

### 6 — Lexicographic Model Selection

The best checkpoint per CV fold is selected by lexicographic ordering (`LexBestModelTracker`):
1. **Primary**: maximise slide-level Balanced Accuracy
2. **Tiebreaker**: minimise validation loss (ε = 10⁻⁶)

This criterion is deliberately decoupled from the early stopping signal (which operates on the EMA-smoothed BA) to prevent the stopping mechanism from biasing model selection.

### 7 — Perturbation Baseline for Faithfulness Curves

The deletion and insertion faithfulness curves use the **per-patch mean intensity** as the baseline fill value. This choice is consistent across both the HybridViT and DenseNet-121 XAI pipelines. All faithfulness conclusions must be interpreted relative to this specific baseline, as the choice of fill value directly influences the insertion metric behavior (particularly for *C. albicans*, where early insertion saturation was observed under this baseline).

### 8 — Primary External Validation Metric

The **locked-pixel pipeline** (no domain adaptation) is the scientifically valid primary result for external validation. The scale-matched and photometric pipelines are post-hoc sensitivity analyses that provide descriptive context for the performance drop but do not represent unadapted deployment readiness. The photometric harmonization result in particular uses test-image statistics for normalization and should never be cited as a primary performance figure.

---

## Hardware and Compute

| Phase | Hardware | Approximate Duration |
|---|---|---|
| Step P2 — Preprocessing | CPU multi-core | 15–45 minutes (per model, reusable) |
| DenseNet-121 CV (30 × 5 folds) | NVIDIA A100 80GB (Colab Pro+) | 8–12 hours (resumable) |
| ConvViT CV | NVIDIA A100 80GB | 10–15 hours (resumable) |
| LsaViT CV | NVIDIA A100 80GB | 10–15 hours (resumable) |
| HybridViT CV | NVIDIA A100 80GB | 10–15 hours (resumable) |
| Final training (any model) | NVIDIA A100 80GB | 1–2 hours |
| External validation (locked-pixel) | CPU or GPU | < 30 minutes |
| HybridViT XAI (UMAP + Rollout) | CPU or GPU | 20–60 minutes |
| DenseNet-121 XAI (Grad-CAM + Faithfulness) | GPU recommended | 1–3 hours |

Automatic Mixed Precision (`use_amp = True`) is enabled by default for CUDA, reducing VRAM usage by approximately 40% with no impact on metric reproducibility. All training notebooks implement full checkpoint resumability: if a Colab session disconnects, re-running the notebook resumes from the last saved checkpoint with no loss of training progress.

---

## Citation

If you use this code, the architecture implementations, the XAI pipeline, or the external validation methodology in your research, please cite the associated dissertation and the WCCI 2026 paper:

```bibtex
@inproceedings{sa2026explainable,
  title     = {Explainable Vision Transformers for {\em Candida} Species
               Identification from Brightfield Microscopy},
  author    = {Sá, Rodrigo and Torres, L. H. M. and Pimentel, C. and Ribeiro, B.},
  booktitle = {Proceedings of the IEEE World Congress on Computational
               Intelligence (WCCI)},
  year      = {2026},
  note      = {To appear}
}
```

For the dataset:

```bibtex
@dataset{sa2026dataset,
  title     = {Brightfield Microscopy Image Dataset for {Candida} Species (v1.0.0)},
  author    = {Sá, Rodrigo and Torres, L. H. M. and Pimentel, C. and Ribeiro, B.},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.20595139},
  url       = {https://doi.org/10.5281/zenodo.20595139}
}
```

*BibTeX entries will be updated with full bibliographic details upon publication.*

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

This work was conducted at [CISUC](https://www.cisuc.uc.pt/) — Centre for Informatics and Systems of the University of Coimbra — in collaboration with the **Yeast Molecular Biology Laboratory** at ITQB NOVA, Universidade NOVA de Lisboa. All wet-lab procedures (strain cultivation, sample preparation, image acquisition) were performed by Dr. Catarina Pimentel, Dr. Catarina Amaral and Carolina Mariano at the ITQB NOVA facilities in Oeiras, Portugal.

The preceding SimpleViT baseline and WCCI 2026 paper are available at: [https://github.com/Rodrigo2003-PT/simpleViT_fungi](https://github.com/Rodrigo2003-PT/simpleViT_fungi)

Illumination correction uses [`BaSiCPy`](https://github.com/peng-lab/BaSiCPy), a Python reimplementation of the BaSiC algorithm (Peng et al., 2017). DenseNet-121 is sourced from `torchvision.models`.

Generative AI tools and automated grammar checkers were used solely for linguistic refinement and editorial assistance. All scientific content, experimental design, data analysis, and conclusions remain the full responsibility of the authors.

---

## References

- Abnar, S. & Zuidema, W. (2020). Quantifying attention flow in transformers. *ACL 2020*, 4190–4197.
- Bengio, Y. & Grandvalet, Y. (2003). No unbiased estimator of the variance of k-fold cross-validation. *JMLR, 5*, 1089–1105.
- Brodersen, K. H. et al. (2010). The balanced accuracy and its posterior distribution. *ICPR 2010*.
- Bussola, N. et al. (2020). AI slipping on tiles: data leakage in digital pathology. *arXiv:1909.06539*.
- Chicco, D. & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics, 21*(6).
- Dosovitskiy, A. et al. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *ICLR 2021*.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874.
- Hassani, A. et al. (2021). Escaping the big data paradigm with compact transformers. *arXiv:2104.05704*.
- He, K. et al. (2016). Deep residual learning for image recognition. *CVPR 2016*, 770–778.
- Huang, G. et al. (2017). Densely connected convolutional networks. *CVPR 2017*, 4700–4708.
- Kim, B. et al. (2016). Examples are not enough, learn to criticize! Criticism for interpretability. *NeurIPS 2016*.
- Lee, S. H. et al. (2021). Vision Transformer for small-size datasets. *arXiv:2112.13492*.
- McInnes, L., Healy, J. & Melville, J. (2020). UMAP: Uniform manifold approximation and projection for dimension reduction. *arXiv:1802.03426*.
- Peng, T. et al. (2017). A BaSiC tool for background and shading correction of optical microscopy images. *Nature Communications, 8*, 14836.
- Samek, W. et al. (2017). Evaluating the visualization of what a deep neural network has learned. *IEEE TNNLS, 28*(11), 2660–2673.
- Selvaraju, R. R. et al. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV 2017*, 618–626.
- Zhang, H. et al. (2018). MixUp: Beyond empirical risk minimization. *ICLR 2018*.
