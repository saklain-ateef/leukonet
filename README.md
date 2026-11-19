LeukoNet: Deep Learning-Based Leukemia Detection from Peripheral Blood Smear Images
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow%2FKeras-2.x-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![Kaggle Weights](https://img.shields.io/badge/Kaggle-Trained%20Models-20BEFF.svg)
Undergraduate Thesis — Department of Computer Science and Engineering, BRAC University (September 2026)
> **Clinical Disclaimer:** LeukoNet is an academic research prototype designed for **screening support**, not autonomous clinical diagnosis. Because the dataset originates from a single hospital, external multi-center validation is required prior to any clinical deployment.
---
Overview
LeukoNet is a deep learning framework for binary classification (Healthy vs. Leukaemia) of peripheral blood smear images collected under routine clinical conditions in Bangladesh.
Key Contributions
Local Clinical Dataset: Collected 1,102 anonymized, pathologist-authenticated microscopic fields (3,306 multi-magnification images) from Ibn Sina Hospital & Diagnostic Center, Jashore, Khulna, Bangladesh.
Leakage-Free Evaluation Protocol: Enforced a strict field-grouped 70:15:15 split so that all three magnification views (`Z1`, `Z2`, `Z3`) of a patient's field remain in the same subset.
Multi-Architecture Benchmark: Evaluated four ImageNet-pretrained models (DenseNet121, EfficientNetB0, ResNet50, and ViT-B/16) using a unified two-phase transfer learning protocol, confirmed with 3-fold grouped cross-validation and statistical significance testing.
Transfer Learning Ablation (LeukoNet-S): Designed and trained a compact 4-block CNN from scratch (423,361 parameters) on the exact same split to isolate the contribution of pretraining (+3% to +5% accuracy and substantially higher recall).
Ensemble & Explainability: Combined the four fine-tuned models via probability averaging to achieve 97.19% test accuracy and 96.76% recall, accompanied by Grad-CAM heatmaps and field-level error analysis.
---
Dataset
LeukoNet Custom Hospital Dataset
All slides were stained with Wright-Giemsa stain under routine diagnostic workflows and captured at $100\times$ oil-immersion magnification. Every field belongs to a distinct, fully anonymized patient and is expanded into three magnification levels:
Level	View	Content	Visual Cues Captured
Z1	Field view	Full uncropped microscopic field	Overall cell density, crowding, background uniformity
Z2	Regional view	Cell-rich regional crop	Relative cell size, clustering, cell-to-cell arrangement
Z3	Cellular view	Tight single-leukocyte crop	Nuclear shape, chromatin texture, staining, N:C ratio
Field-Grouped Split (70:15:15)
Subset	Unique Fields	Total Images	Healthy	Leukaemia
Train	771	2,313	1,380	933
Validation	165	495	285	210
Test (Held-out)	166	498	282	216
Total	1,102	3,306	1,947	1,359
> **Data Access:** Due to patient privacy and hospital data governance, the raw clinical images are not publicly hosted in this repository. Access may be requested from the authors subject to approval from Ibn Sina Hospital & Diagnostic Center.
Public Benchmark Datasets (Pilot Experiments)
ALL-IDB2 / ALL-IDB Subtypes (150 images across FAB L1, L2, L3; pilot classification with `YOLOv8n-cls` and `YOLO11n-cls`).
C-NMC 2019 (10,661 single-cell images; pilot baseline with `ResNet18`).
---
Trained Model Weights
Due to GitHub file size limits, all trained model checkpoints (682.89 MB total) are hosted on Kaggle:
Kaggle Dataset: Leukemia Thesis - Trained Models
Model	Framework	Weight File	Notebook
DenseNet121	TensorFlow / Keras	`densenet121_final.keras`	`notebooks/`
EfficientNetB0	TensorFlow / Keras	`efficientnetb0_finetuned.keras`	`notebooks/`
ResNet50	TensorFlow / Keras	`resnet50_final.keras`	`notebooks/`
ViT-B/16	PyTorch / torchvision	`vitb16_finetuned.pt`	`notebooks/`
LeukoNet-S (Scratch)	TensorFlow / Keras	`scratch_cnn_final.keras`	`notebooks/`
---
Results
Held-Out Test Set Performance (498 Images; Positive Class = Leukaemia)
Model	Accuracy	AUC	Precision	Recall	F1-Score
DenseNet121	96.18%	0.9916	0.9539	0.9583	0.9561
EfficientNetB0	94.38%	0.9774	0.9352	0.9352	0.9352
ResNet50	95.58%	0.9928	0.9663	0.9306	0.9481
ViT-B/16	96.39%	0.9947	0.9670	0.9491	0.9579
LeukoNet-S (From Scratch)	91.37%	0.9797	0.9391	0.8565	0.8959
4-Model Ensemble (Prob. Avg.)	97.19%	0.9917	0.9676	0.9676	0.9676
3-Fold Grouped Cross-Validation (Test Accuracy)
Model	Fold 1	Fold 2	Fold 3	Mean $\pm$ SD
ViT-B/16	0.9659	0.9618	0.9659	$0.9645 \pm 0.0019$
ResNet50	0.9578	0.9598	0.9558	$0.9578 \pm 0.0016$
DenseNet121	0.9498	0.9598	0.9498	$0.9531 \pm 0.0047$
EfficientNetB0	0.9337	0.9398	0.9418	$0.9384 \pm 0.0034$
An omnibus Friedman test confirmed a statistically significant overall difference across the four architectures ($p = 0.0322$). Pairwise tests showed DenseNet121, ResNet50, and ViT-B/16 perform comparably well and outperform EfficientNetB0.
Across all 498 test images, 412 (82.7%) were classified correctly by all four fine-tuned models and only 6 (1.2%) (belonging to 4 unique fields) were missed by all four.
---
Training Protocol
Input Preprocessing: Resized to $224 \times 224$ RGB, pixel values scaled to $[0, 1]$. No grayscale conversion (Wright-Giemsa stain color is preserved).
Training Augmentation (Train split only): Random horizontal/vertical flips, brightness and contrast jitter.
Class Imbalance: Handled with class-weighted binary cross-entropy loss (`0.838` Healthy / `1.240` Leukaemia).
Two-Phase Fine-Tuning (Pretrained Models):
Phase 1 (Warm-up): Frozen backbone, custom classification head trained up to 10 epochs (Adam, $\text{lr} = 10^{-3}$, early stopping patience = 3).
Phase 2 (Fine-tuning): Top 30% of backbone layers unfrozen, trained up to 15 epochs ($\text{lr} = 10^{-5}$, early stopping patience = 4).
LeukoNet-S (From Scratch): Single phase up to 40 epochs (Adam, $\text{lr} = 10^{-3}$, `ReduceLROnPlateau`, early stopping patience = 8).
---
Repository Structure
```text
leukonet/
├── data/          # Split metadata CSVs (no raw clinical patient images)
├── notebooks/     # Training, cross-validation, ensemble, and Grad-CAM notebooks
├── src/           # Modular preprocessing, model, and evaluation scripts
├── results/       # ROC curves, confusion matrices, and Grad-CAM visualizations
├── .gitignore
└── README.md
```
---
Setup & Usage
Clone the repository:
```bash
   git clone https://github.com/saklain-ateef/leukonet.git
   cd leukonet
   ```
Install dependencies:
```bash
   pip install tensorflow torch torchvision numpy pandas scikit-learn matplotlib seaborn opencv-python
   ```
Download trained weights:
Download the `.keras` and `.pt` checkpoints from the Kaggle Trained Models repository and place them in a local `weights/` directory (ignored by Git).
Run notebooks:
Open any notebook in `notebooks/` via Jupyter Lab, VS Code, Google Colab, or Kaggle to reproduce training, ensemble evaluation, or Grad-CAM visualizations.
---
Authors & Acknowledgements
Authors (BRAC University):
Ateef Saklain (22201302)
Khosnur Alam Shuchi (22299167)
Mahabuba Afrose (22299302)
Nusrat Zaman Nusa (22301586)
Supervisor:
Niloy Farhan, Lecturer, Department of Computer Science and Engineering, BRAC University
Clinical Collaboration:
Dr. Md. Azam Saklain (BMDC - A 25196), Ibn Sina Hospital & Diagnostic Center, Jashore, Khulna, Bangladesh — for clinical authentication, annotation, and anonymization verification of the dataset.
---
Citation
If you reference this work, please cite:
```bibtex
@thesis{saklain2026leukonet,
  author  = {Saklain, Ateef and Shuchi, Khosnur Alam and Afrose, Mahabuba and Nusa, Nusrat Zaman},
  title   = {LeukoNet: Deep Learning-Based Leukemia Detection from Peripheral Blood Smear Images},
  school  = {BRAC University, Department of Computer Science and Engineering},
  year    = {2026},
  month   = {September},
  type    = {B.Sc. Thesis}
}
```