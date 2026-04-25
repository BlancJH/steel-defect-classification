# Severstal Steel Defect Classification

This repository contains an evolutionary pipeline for detecting structural defects in sheets of steel. The original dataset provides RLE (Run-Length Encoded) pixel masks for semantic segmentation, which we aggregated into a highly imbalanced **Multi-Label Image Classification** task (predicting whether one or more of 4 defect classes exist anywhere in the image).

To methodically push the bounds of performance, the `baseline.ipynb` notebook iterates through exactly 3 Experimental Settings.

---

## The Three Experimental Settings

### Experimental Setting 1: Baseline (Custom CNN)
**Implementation:** A lightweight Convolutional Neural Network built practically from scratch.
* **Architecture:** Simple cascaded Conv2D, MaxPool2D, and ReLU blocks followed by a fully connected classification head.
* **Goal:** Serve as a functional pipeline test to prove our DataLoaders, Multi-Hot label conversions, and Metric mappings (F1, Precision, Recall) were mathematically sound before introducing complexity. 

### Experimental Setting 2: Transfer Learning
This setting explores the standard procedure of importing large, pre-trained ImageNet models (`ResNet-50`), structured in two overlapping phases.

#### Phase A: Frozen Backbone (Head-Only)
* **Architecture:** The entire ResNet-50 core was strictly frozen (`requires_grad = False`). We stripped the final classification layer and replaced it with a randomly initialized multi-label linear head.
* **Goal:** Prove that transferring generic feature extraction from millions of ImageNet images instantly outperforms small CNNs trained from scratch on limited data.

#### Phase B: Partial Fine-Tuning
* **Architecture:** The exact same ResNet-50 backbone from Phase A, but the final, deepest convolutional block (`layer4`) was deliberately **unfrozen** alongside the classification head to allow gradients to adjust the internal feature maps.

#### Analytical Comparison: Why did Partial Fine-Tuning beat the Frozen Backbone?
The **Frozen Backbone** had a strict theoretical limit because its internal filters were optimized to identify shapes found in ImageNet (e.g., dog ears, car wheels, human faces). Steel defects present themselves as noisy, highly repetitive, greyscale textures (micro-fractures and subtle rust patches). 

By explicitly **unfreezing `layer4`** in Partial Fine-Tuning, we permitted backpropagation to mathematically rewire the model's deepest, most abstract feature maps. Instead of trying to classify steel scratches using "dog ear" filters, the network learned to mold those deepest layers into highly specialized defect detectors. This dramatically spiked the model's **Recall**, allowing it to spot subtle flaws the frozen feature-maps were originally blind to.

### Experimental Setting 3: Advanced Fine-Tuning (The Grandmaster Approach)
**Implementation:** A massive overhaul of both the data ingestion and the model architecture to combat the physical limitations of the ResNet-50 built in Setting 2.
* **Architecture:** Dumped ResNet-50 for **EfficientNet-V2-S**. We unfroze blocks `6` and `7` for deep partial fine-tuning.
* **Optimization:** Used `AdamW` with **Discriminative Learning Rates** (a highly restricted 1e-5 rate mapping applied to the backbone to prevent breaking pretrained weights, and a faster 1e-3 rate assigned to the Head).
* **Head Upgrade:** Replaced standard Global Average Pooling with **Concat Pooling** (Concatenating AdaptiveAveragePool and AdaptiveMaxPool).
* **Data Geometry:** Replaced severe `256x256` squish resizing with native **Rectangular Inputs** (`128x800`).

#### Analytical Comparison: Why did Setting 3 shatter the Partial Fine-Tuning ceiling of Setting 2?
The transition from Setting 2's Partial Fine-Tuning to Setting 3 yielded a monumental leap in Macro F1 scores (climbing up into the mid/upper `0.8x` limits). This occurred due to three major paradigm shifts:

1. **Geometry Preservation (The biggest factor):** The native steel images are incredibly wide (`1600x256`). Forcing them into `256x256` squares in Setting 2 mathematically warped and blurred the micro-fractures into invisible noise. By preserving the wide aspect ratio via `128x800` rectangular tensors, the model finally had the pixel clarity required to perceive tiny Class 2 and Class 3 defects without spatial distortion.
2. **Concat Pooling:** Standard ResNet layers use Average Pooling by default, which washes out localized textures. By injecting **Max Pooling** alongside it in Setting 3, the classification head was fed both the "overall steel context" (Average) AND the "sharpest, most extreme outlier pixels" (Max), making it brutally effective at spotting tiny, high-contrast scratches.
3. **Test-Time Augmentation (TTA):** By evaluating flipped variations of the Unseen Test Data and statistically averaging their probability outputs, we insulated the model against false positives triggered by unequal lighting and anomalous camera rotations.

---
*Results validated using perfectly deterministic seeds to ensure exact reproducibility across sequential runs.*