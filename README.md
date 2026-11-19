# LeukoNet: Deep Learning-Based Leukemia Detection from Peripheral Blood Smear Images

Undergraduate Thesis — Department of Computer Science and Engineering, Brac University (September 2026)

## Authors
- **Ateef Saklain** (22201302)
- **Khosnur Alam Shuchi** (22299167)
- **Mahabuba Afrose** (22299302)
- **Nusrat Zaman Nusa** (22301586)

**Supervisor:** Niloy Farhan, Lecturer, Department of CSE, Brac University

---

## Overview
LeukoNet is a deep learning system for binary leukemia screening (**Healthy** vs. **Leukaemia**) from peripheral blood smear images. Using a newly collected, clinically annotated hospital dataset split strictly by microscopic field to prevent data leakage, we compare four fine-tuned pretrained architectures (**DenseNet121**, **EfficientNetB0**, **ResNet50**, and **ViT-B/16**), a compact from-scratch CNN (**LeukoNet-S**), and a probability-averaging ensemble, alongside **Grad-CAM** explainability.

## Dataset
- **Source:** Ibn Sina Hospital & Diagnostic Center, Jashore, Khulna, Bangladesh (anonymized, clinically verified by a pathologist; private dataset—access by request).
- **Composition:** 1,102 unique microscopic fields (649 Healthy, 453 Leukaemia), each cropped at three magnification levels (**Z1** field view, **Z2** regional view, **Z3** cellular view) = **3,306 images**.
- **Leakage-Free Split (70:15:15 grouped by field):**
  - **Train:** 771 fields (2,313 images)
  - **Validation:** 165 fields (495 images)
  - **Test:** 166 fields (498 images)

## Trained Models
All trained model weights are hosted on Kaggle:
[Leukemia Thesis - Trained Models (Kaggle)](https://www.kaggle.com/datasets/ateefsaklain17/leukemia-thesis-trained-models)

| Model | Weight File | Framework |
|---|---|---|
| DenseNet121 | `densenet121_final.keras` | TensorFlow / Keras |
| EfficientNetB0 | `efficientnetb0_finetuned.keras` | TensorFlow / Keras |
| ResNet50 | `resnet50_final.keras` | TensorFlow / Keras |
| LeukoNet-S (Scratch) | `scratch_cnn_final.keras` | TensorFlow / Keras |
| ViT-B/16 | `vitb16_finetuned.pt` | PyTorch / torchvision |

## Key Results (Held-Out Test Set: 498 Images)

| Model | Accuracy | AUC | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|---:|
| DenseNet121 | 96.18% | 0.9916 | 0.9539 | 0.9583 | 0.9561 |
| EfficientNetB0 | 94.38% | 0.9774 | 0.9352 | 0.9352 | 0.9352 |
| ResNet50 | 95.58% | 0.9928 | 0.9663 | 0.9306 | 0.9481 |
| ViT-B/16 | 96.39% | 0.9947 | 0.9670 | 0.9491 | 0.9579 |
| LeukoNet-S (From Scratch) | 91.37% | 0.9797 | 0.9391 | 0.8565 | 0.8959 |
| **4-Model Ensemble** | **97.19%** | **0.9917** | **0.9676** | **0.9676** | **0.9676** |

*Note: Results are from a single hospital and require external validation before clinical use. LeukoNet is intended for screening support, not autonomous diagnosis.*

## Repository Structure
```text
leukonet/
├── data/          # Dataset metadata / CSV files
├── notebooks/     # Training, cross-validation, ensemble, and Grad-CAM notebooks
├── src/           # Modular Python scripts
├── results/       # Evaluation plots, confusion matrices, and Grad-CAM outputs
└── README.md
```