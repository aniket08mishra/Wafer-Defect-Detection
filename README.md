# Automated Wafer Defect Detection and Pattern Recognition

[![Python 3.9+](https://shields.io)](https://python.org)
[![PyTorch](https://shields.io)](https://pytorch.org)
[![TensorFlow](https://shields.io)](https://tensorflow.org)
[![License: GPL-3.0](https://shields.io)](https://gnu.org)

Developed at the **Q-MADE Lab, Department of Materials Science and Engineering, Indian Institute of Technology Kanpur (IITK)**. This repository hosts an end-to-end Deep Learning framework designed to automate industrial wafer-map defect inspection, substituting conventional and high-latency manual visual inspections with representation learning.

<p align="center">
  <img src="https://shields.io" />
  <img src="https://shields.io" />
</p>

---

## 📌 Project Overview

In semiconductor fabrication, spatial failure distributions on silicon substrates (wafer maps) encode highly specific information regarding root-cause process anomalies (e.g., plasma-etch imbalances, chemical vapor deposition variance, or mechanical scratches). 

This project establishes a robust computer vision pipeline utilizing **Double-Layer Convolutional Neural Networks (CNNs)** and **Autoencoders** to automatically segment, extract spatial signatures, and classify topologies from real-world fabrication data.

### Supported Topology Anomalies
The architecture is optimized to categorize the 8 industrial defect footprints found within the **WM811K dataset**:
*   **Center**: Failures clustered tightly at the center of the wafer.
*   **Donut**: Ring-shaped defect distributions with a clear core.
*   **Edge-Loc / Edge-Ring**: Symmetrical or localized marginal anomalies.
*   **Loc (Localized)**: High-density focal point defects.
*   **Scratch**: Linear directional signatures typical of tool/handling damage.
*   **Random**: Stochastic noise distributions across the die map.
*   **Near-full**: Severe macro-scale processing failures.

---

## 🔬 Framework & Architecture

The pipeline consists of three core engineering milestones:

1. **Data Preprocessing & Structural Normalization**
   * Dimensions within the WM811K dataset are highly variable (ranging across 632 unique shape permutations). Raw matrix data is dimensionally standardized to unified shapes (224 × 224 × 3) using bilinear area interpolation to retain pixel density configurations.
2. **Robustness Engineering via Data Augmentation**
   * Labeled wafer spaces are highly imbalanced, with native pattern defects comprising only 3.1% of the dataset (25,519 instances). To counteract severe overfitting, runtime augmentations are applied via the `imgaug` library, including bidirectional flipping, localized translations, spatial shearing, and uniform rotation steps across (-180°, 180°).
3. **Deep Convolutional Network Stack**
   * Implements a convolutional structure with Batch Normalization (`BatchNorm`) layers immediately preceding ReLU activation functions to stabilize inner covariate shifts.
   * Utilizes Spatial 2D Dropout layers to drop entire feature maps rather than individual pixels, preserving spatial structure while regularizing the internal layers.
   * **Bayesian Hyperparameter Optimization** is integrated to dynamically isolate performance boundaries across variable internal learning rates and kernel weights.

---

## 📊 Evaluation Metrics & Results

The network delivers high-performance evaluation benchmarks across highly imbalanced data matrices:

*   **Overall Classification Accuracy**: `98.0%`
*   **Overall Precision**: `93.0%`

### Confusion Matrix Insights
The model exhibits distinct structural precision when isolating macro-scale anomalies:
*   **Near-full** anomalies are classified with a high precision rate.
*   **Edge-Ring** signatures achieve exceptional topological partitioning due to symmetric spatial filters.

---

## 🛠️ Installation & Dependency Setup

Ensure you have a Python environment manager configured (such as Miniconda or Anaconda).

1. Clone this repository locally:
   ```bash
   git clone https://github.com
   cd Wafer-Defect-Detection
   ```

2. Create a clean virtual environment and install core engineering libraries:
   ```bash
   conda create -n wafer-env python=3.9 -y
   conda activate wafer-env
   pip install torch torchvision numpy pandas scikit-learn opencv-python imgaug matplotlib seaborn tensorflow
   ```

---

## 📂 Repository Structure
---

## 🎯 Research Applications

This pipeline is built to integrate with broader digital twin architectures in semiconductor manufacturing. By mapping specific spatial defect footprints to process steps, this model can be chained to:
*   **Plasma-Etch Surrogates**: Correlating `Edge-Ring` or `Edge-Loc` faults with process chemistry parameters.
*   **Bayesian Recipe Optimization**: Automating the feedback loop to dynamically adjust tool settings when structural degradation patterns are flagged.

---

## ✉️ Contact & Collaboration

**Aniket Mishra**  
*M.Tech Candidate, Department of Materials Science and Engineering*  
**Indian Institute of Technology Kanpur (IITK)**  
Research Focus: Semiconductor AI, Scientific ML, Process Digital Twins  

*   **Email**: [aniket08mishra@gmail.com](mailto:aniket08mishra@gmail.com)
*   **LinkedIn**: [://linkedin.com](https://www.://linkedin.com/)  
*   **ORCID**: [0009-0001-4550-7337](https://orcid.org)
