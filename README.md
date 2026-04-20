# 🧠 Alzheimer's Disease Diagnosis with 3D Attention U-Net

> A deep learning pipeline for automated Alzheimer's Disease diagnosis from 3D structural MRI brain scans, using a **3D Attention U-Net** architecture with soft attention gates for improved classification accuracy across disease stages.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Datasets](#-datasets)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Results](#-results)
- [Report](#-report)
- [Requirements](#-requirements)
- [References](#-references)

---

## 📌 Overview

Alzheimer's Disease (AD) is a progressive neurodegenerative disorder and the leading cause of dementia worldwide. Early and accurate diagnosis is critical for timely clinical intervention.

This project implements a 3D deep learning pipeline for classifying MRI brain scans into three diagnostic categories:

| Label | Meaning |
|---|---|
| `CN` | Cognitively Normal |
| `MCI` | Mild Cognitive Impairment |
| `AD` | Alzheimer's Disease |

**Key highlights:**
- Three progressively richer 3D U-Net variants: `base`, `advance`, and `more_advance` (with attention)
- Soft attention gates integrated into skip connections to focus on clinically relevant brain regions (e.g. hippocampus, entorhinal cortex)
- Full MRI preprocessing pipeline: resampling to 2mm isotropic voxels, skull stripping via FSL-BET
- Training via PyTorch Lightning with gradient accumulation, early stopping, and WandB logging
- YAML-based configuration for reproducible experiments

---

## 🏗️ Architecture

Three model variants are available, selected via `configs/config.yaml`:

| Model Type | Description |
|---|---|
| `base` | Standard 3D U-Net encoder–decoder |
| `advance` | Enhanced U-Net with deeper convolutional blocks |
| `more_advance` | Attention U-Net with soft attention gates on skip connections |

The `more_advance` model (default) is based on **Attention U-Net** (Oktay et al., MIDL 2018). Attention gates learn to suppress irrelevant feature activations and highlight diagnostically salient regions, without requiring external localisation models.

```
Input 3D MRI  [1 × 80 × 100 × 80]
      │
  ┌───▼──────────────────────────────────┐
  │  Encoder (Contracting Path)          │
  │  Conv3D → BN → ReLU → DownSample     │
  │  (×4 scales: 4 → 16 → 32 → 64 ch)    │
  └───┬────────┬────────┬────────────────┘
      │        │        │
  [skip 1] [skip 2] [skip 3]  ← Attention Gates filter these
      │        │        │
  ┌───▼────────▼────────▼────────────────┐
  │           Bottleneck                 │
  │  Conv3D × 2 (64 channels)            │
  └───┬────────────────────────────────--┘
      │
  ┌───▼──────────────────────────────────┐
  │  Decoder (Expanding Path)            │
  │  UpSample + Attended Skip + Conv3D   │
  │  (×3 scales: 32 → 16 → 8 ch)         │
  └───┬──────────────────────────────────┘
      │
  Disease Classifier Head  →  [CN | MCI | AD]
```

---

## 📁 Repository Structure

```
Alzheimer-Diagnosis/
│
├── src/
│   ├── model/
│   │   ├── base_3d.py            # Base 3D U-Net
│   │   ├── advance_3d.py         # Advanced 3D U-Net
│   │   └── advance_att_3d.py     # Attention 3D U-Net (default)
│   ├── data/
│   │   ├── mri_processing.py     # MRI preprocessing (resampling, skull stripping)
│   │   └── utils_data.py         # Dataset classes and DataModule
│   └── utils/
│       ├── utils.py              # Patient folder navigation and inspection
│       ├── utils_grad.py         # Gradient utilities
│       └── plots.py              # Confusion matrix and training plots
│
├── scripts/
│   ├── generate_data.py          # Build and save dataset from ADNI raw files
│   └── train.py                  # Training entry point
│
├── configs/
│   └── config.yaml               # All experiment hyperparameters and paths
│
├── datasets/                     # Raw MRI data (not tracked)
├── data_split/                   # Saved train/val/test splits
├── ckpt/                         # Model checkpoints
├── confusion_matrices/           # Saved confusion matrix plots
│
├── test_notebook.ipynb           # Exploration and inference notebook
├── environment.yml               # Conda environment specification
├── pyproject.toml                # Package build config
└── report.pdf                    # Full academic report
```

---

## 📦 Datasets

This project uses data from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**. The following ADNI sub-collections are supported (configurable in `config.yaml`):

- `ADNI1-Annual_2_Yr_3T`
- `ADNI1-Baseline_3T`
- `ADNI1-Complete_1Yr_1.5T`
- `ADNI1-Complete_2Yr_1.5T`
- `ADNI1-Complete_3Yr_1.5T`
- `ADNI1-Screening_1.5T`

> ⚠️ ADNI data is **not included** in this repository. You must apply for access at [adni.loni.usc.edu](https://adni.loni.usc.edu/) and place the downloaded data under `datasets/` following the structure expected by `config.yaml`.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12
- Conda (recommended)
- FSL installed and available on `$PATH` (for skull stripping via `nipype`)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/antoogagliardi/Alzheimer-Diagnosis.git
cd Alzheimer-Diagnosis
```

2. **Create and activate the Conda environment:**

```bash
conda env create -f environment.yml
conda activate alzheimer
```

3. **Install the project as a package** (enables `src` imports):

```bash
pip install -e .
```

---

## 💻 Usage

### 1. Prepare the Dataset

After placing your ADNI data under `datasets/`, update the paths in `configs/config.yaml` and run:

```bash
python scripts/generate_data.py
```

This script reads the ADNI CSV metadata, matches patients to their MRI scan files, builds a `MRIDataset`, and saves it as a `.db` file for fast reloading.

### 2. Train the Model

```bash
python scripts/train.py
```

Training is managed by **PyTorch Lightning** and logged via **Weights & Biases**. Checkpoints are saved to `ckpt/<model_type>/<wandb_run_id>/`.

### 3. Explore and Evaluate

Open the notebook for interactive exploration, inference, and visualisation:

```bash
jupyter notebook test_notebook.ipynb
```

---

## ⚙️ Configuration

All experiment settings are controlled via `configs/config.yaml`:

To resume a previous WandB run, set `resume: True` and provide the `wandb_runID` and `last_epoch`.

---

## 📊 Results

Training metrics (loss, F1-score), confusion matrices, and validation curves are tracked via WandB and saved locally under `confusion_matrices/`. Refer to `test_notebook.ipynb` for a detailed breakdown of per-class performance across the three diagnostic categories.

---

## 📄 Report

A full academic-style report covering background, dataset, model architecture, training setup, experiments, and results is included:

📎 [`report.pdf`](./report.pdf)

---

## 🛠️ Requirements

| Library | Purpose |
|---|---|
| `torch` | Core deep learning framework |
| `lightning` | Training loop and callbacks |
| `torchmetrics` | F1 score, confusion matrix |
| `SimpleITK` | MRI image I/O and resampling |
| `nipype` + FSL | Skull stripping (FSL-BET) |
| `wandb` | Experiment tracking |
| `numpy`, `pandas` | Data handling |
| `matplotlib` | Visualisation |
| `tqdm` | Progress bars |

Full environment specification: [`environment.yml`](./environment.yml)

---

## 📚 References

- Oktay, O. et al. (2018). [*Attention U-Net: Learning Where to Look for the Pancreas*](https://arxiv.org/abs/1804.03999). MIDL 2018.
- Ronneberger, O. et al. (2015). [*U-Net: Convolutional Networks for Biomedical Image Segmentation*](https://arxiv.org/abs/1505.04597). MICCAI 2015.
- ADNI — Alzheimer's Disease Neuroimaging Initiative: [adni.loni.usc.edu](https://adni.loni.usc.edu/)

---

## 👤 Author

**Antonio Gagliardi**  
Email: [gaglia.anto95@gmail.com](mailto:gaglia.anto95@gmail.com)