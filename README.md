# Metal Surface Defect Segmentation & Classification

This repository contains Jupyter Notebooks focused on identifying and localizing defects on steel surfaces using the [Severstal Steel Defect Detection Dataset](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data).

The primary focus of this repository is **pixel-level semantic segmentation** to precisely locate defects and their classes, with a supplementary **multi-label classification** approach for lightweight experimentation.

---

## 🌟 Main Model: Semantic Segmentation (`steel_defect_segmentation.ipynb`)

The main model extends beyond image-level classification to perform **pixel-level semantic segmentation**, predicting exactly which pixels belong to each of the 4 defect classes.

### Key Features
- **Architecture**: U-Net with a SegFormer **MiT-B2** Vision Transformer backbone via `segmentation_models_pytorch`.
- **Data Pipeline**: RLE-decoded masks with synchronized spatial augmentations (`albumentations`).
- **Loss Function**: Combined Dice Loss + Focal Loss for handling extreme class imbalance.
- **Visualization**: Colour-coded overlay of predicted defect masks (Red / Green / Blue / Yellow per class).

---

## 🔬 Lightweight Experiments: Classification (`steel_defect_classification.ipynb`)

For faster iteration and lightweight modeling, we also include a multi-label classification notebook. It identifies whether an image contains any of the 4 types of defects (without localizing them).

The notebook explores three experimental settings:
1. **Baseline (Setting 1)**: A custom Convolutional Neural Network (CNN) trained from scratch.
2. **Transfer Learning (Setting 2)**: A frozen ResNet-50 backbone with a new classification head, followed by partial fine-tuning.
3. **Advanced Partial Fine-Tuning (Setting 3)**: An EfficientNet-V2-S backbone paired with a highly customized and heavily regularized classification head.

---

## 🚀 Step-by-Step Setup Guide

Follow these instructions to set up the environment and run the notebooks locally.

### 1. Clone the Repository
```bash
git clone https://github.com/BlancJH/metal-defect-classification.git
cd metal-defect-classification
```

### 2. Set Up a Virtual Environment (Recommended)
It is highly recommended to use a Python virtual environment to manage dependencies.
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

### 3. Install Dependencies
Install the required packages. Ensure you have the correct version of PyTorch for your system (CUDA recommended for GPU acceleration).

```bash
# Core Machine Learning & Data Processing
pip install torch torchvision
pip install pandas numpy scikit-learn matplotlib Pillow opencv-python

# Utility for Dataset Downloads
pip install gdown

# Segmentation & Augmentation Specific
pip install segmentation-models-pytorch albumentations timm

# Jupyter to run the notebooks
pip install jupyter
```

*(Note: For PyTorch with CUDA, please refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) for your specific system.)*

### 4. Data Preparation
Both notebooks are configured to handle data downloading and preparation dynamically. You **do not** need to manually download the dataset beforehand.
- The notebooks will automatically download training images (via Google Drive `gdown`) and dataset splits.
- The data will be split into 80/10/10 ratios for Train/Validation/Test respectively and saved locally to ensure consistency across runs.

### 5. Run the Notebooks
Launch Jupyter Notebook:
```bash
jupyter notebook
```
Open `steel_defect_segmentation.ipynb` (Main Model) or `steel_defect_classification.ipynb` (Lightweight Experiment) and execute the cells sequentially.

---

## 📊 Evaluation Metrics

Depending on the notebook, the models are evaluated using appropriate metrics on a held-out test subset (10%):

- **Segmentation**: Pixel-level accuracy, Intersection over Union (IoU), and Dice Coefficient.
- **Classification**: F1-Score (Macro), Precision (Macro), and Recall (Macro).

Training and validation pipelines are fully visible, tracking losses and performance metrics across epochs, culminating in comprehensive metric visualizations.
