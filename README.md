# 🧠 Alzheimer's Disease Diagnosis with Attention U-Net

> A deep learning approach to Alzheimer's Disease diagnosis using an Attention U-Net architecture for MRI brain image segmentation and classification.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Report](#report)
- [Requirements](#requirements)
- [License](#license)

---

## 📌 Overview

This project explores the use of **Attention U-Net**, a variant of the classic U-Net architecture enhanced with attention gates, for the automated diagnosis of **Alzheimer's Disease** from MRI brain scans.

Alzheimer's Disease is a progressive neurodegenerative disorder and one of the leading causes of dementia worldwide. Early and accurate diagnosis is critical for timely intervention. This project leverages deep learning techniques to assist in the identification and classification of disease stages from medical imaging data.

Key highlights:
- Attention-based skip connections to focus on relevant brain regions
- MRI brain scan segmentation and classification pipeline
- End-to-end Jupyter Notebook workflow for reproducibility
- Detailed exploratory data analysis

---

## 🏗️ Architecture

The core model is an **Attention U-Net**, which extends the standard U-Net with **soft attention gates** integrated into the skip connections. This mechanism suppresses irrelevant activations and highlights salient features in the input MRI scans, improving segmentation precision in heterogeneous brain structures.

```
Input MRI
   │
Encoder (Contracting Path)
   │  ↘ attention gates
Bottleneck
   │  ↗ skip connections
Decoder (Expanding Path)
   │
Output Segmentation / Classification
```

---

## 📁 Repository Structure

```
Alzheimer-Diagnosis/
│
├── functions/              # Utility and helper functions (model, data loading, metrics, etc.)
│
├── inspect_data.ipynb      # Exploratory Data Analysis: dataset inspection and visualization
├── notebook.ipynb          # Main training and evaluation pipeline
│
└── report.pdf              # Full project report with methodology, experiments, and results
```

---

## 🚀 Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Installation

1. Clone the repository:

```bash
git clone https://github.com/antoogagliardi/Alzheimer-Diagnosis.git
cd Alzheimer-Diagnosis
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** If no `requirements.txt` is present, the core libraries used are listed in the [Requirements](#requirements) section below.

---

## 💻 Usage

### 1. Explore the Data

Open and run the data inspection notebook to visualize and understand the dataset:

```bash
jupyter notebook inspect_data.ipynb
```

### 2. Train & Evaluate the Model

Run the main notebook for the full pipeline — preprocessing, model training, and evaluation:

```bash
jupyter notebook notebook.ipynb
```

---

## 📊 Results

The model was trained and evaluated on MRI brain scan data across multiple Alzheimer's stages. Refer to `notebook.ipynb` for detailed metrics including accuracy, loss curves, and segmentation outputs. For a comprehensive analysis, see [`report.pdf`](./report.pdf).

---

## 📄 Report

A full academic-style report detailing the methodology, dataset, model architecture, training procedure, and experimental results is available:

📎 [`report.pdf`](./report.pdf)

---

## 🛠️ Requirements

The main libraries used in this project include:

| Library | Purpose |
|---|---|
| `tensorflow` / `keras` | Deep learning framework |
| `numpy` | Numerical computation |
| `matplotlib` | Data visualization |
| `scikit-learn` | Evaluation metrics |
| `nibabel` / `SimpleITK` | MRI image I/O |
| `opencv-python` | Image preprocessing |

---

## 📬 Contact

**Antonio Gagliardi**
- GitHub: [@antoogagliardi](https://github.com/antoogagliardi)

---

## 📝 License

This project is intended for academic and research purposes. Please refer to the repository for licensing details.
