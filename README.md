# Metal Surface Defect Detection

This repository contains a Jupyter Notebook (`steel_defect_classification.ipynb`) that implements multi-label classification of metal surface defects using the [Severstal Steel Defect Detection Dataset](https://www.kaggle.com/competitions/severstal-steel-defect-detection/data).

## Project Overview

The task is to perform multi-label classification to identify 4 different types of defects that can occur on steel surfaces. 
The notebook explores three different experimental settings:

1. **Baseline (Setting 1)**: A custom Convolutional Neural Network (CNN) trained from scratch.
2. **Transfer Learning (Setting 2)**: A frozen ResNet-50 backbone with a new classification head, followed by partial fine-tuning of the final convolutional block.
3. **Advanced Partial Fine-Tuning (Setting 3)**: An EfficientNet-V2-S backbone paired with a highly customized and heavily regularized classification head based on findings from previous experiments.

## Data Preparation

The notebook handles data downloading and preparation dynamically:
- Automatically downloads training images (via Google Drive `gdown`) and dataset splits (from GitHub).
- Splits the dataset into 80/10/10 ratios for Train/Validation/Test respectively. The splits are saved locally and reused across runs to ensure consistency.
- Defines a custom PyTorch `Dataset` (`SteelDefectDataset`) that manages images with and without defects, utilizing multi-hot encoding for the 4 class labels.

## Execution

You can run the notebook locally or on cloud platforms like Google Colab.

### Requirements

To run the notebook, install the necessary dependencies:
- PyTorch (`torch`, `torchvision`)
- Data manipulation and modeling: `pandas`, `numpy`, `scikit-learn`
- Visualization: `matplotlib`
- Utilities: `gdown` (for Drive downloads), `Pillow` (for images)

## Evaluation Metrics

For each experimental setting, the notebook evaluates classification performance using:
- **F1-Score (Macro)**
- **Precision (Macro)**
- **Recall (Macro)**

A held-out test subset (10%) is used for final metrics evaluation to ensure unbiased testing on unseen data. The notebook also demonstrates the training and validation pipelines by tracking losses and F1-scores across epochs, followed by metric visualizations like confusion matrices.
