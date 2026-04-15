# OHL-MRC: Ontology-Guided Hierarchical Loss with Multi-Resolution Calibration

> **Code and data for the paper:**
> *"Ontology-guided hierarchical loss with multi-resolution calibration for fine-grained skin lesion classification on MILK10k"*
> Md Mynoddin, Dulal Chakraborty, Troyee Dev
> *Journal of Investigative Dermatology* (under review)

---

## Overview

Standard skin lesion classifiers are trained with flat cross-entropy that treats all misclassifications as equally costly. Clinically, predicting *Melanoma Invasive* as *Melanoma in situ* (same ontological branch) is far less dangerous than predicting it as *Seborrhoeic Keratosis* (cross-boundary). This repository provides the code, trained models, and evaluation framework to close that training–evaluation gap.

**OHL-MRC** combines three components:

| Component | What it does |
|-----------|-------------|
| **Siamese ResNet-50 + CMA** | Processes paired close-up + dermoscopy images with a cross-modal attention gate |
| **OHL-CE loss** | Encodes the ISIC-DX ontology into training via DWSL soft labels (τ = 2.0) and APCL ancestor-path penalty (λ = 0.1) |
| **MRC-Eval** | Assesses confidence calibration at all four ISIC-DX hierarchy levels simultaneously |

---

## Key Results

Evaluated on **MILK10k** (10,480 images · 5,240 cases · 95.7% histopathologically confirmed · 48 ISIC-DX classes) via stratified 5-fold cross-validation:

| Condition | Macro-Recall | Top-1 | HDM ↓ | MBER ↓ | HCM ↓ |
|-----------|:---:|:---:|:---:|:---:|:---:|
| M1 Flat CE (baseline) | 0.297 ± 0.027 | 0.586 ± 0.016 | 1.687 | 0.217 | 0.083 |
| **M5 OHL-CE (proposed)** | **0.305 ± 0.017** | **0.595 ± 0.017** | **1.631** | **0.207** | 0.260 |
| **M6 OHL-CE + TS (proposed)** | **0.318 ± 0.026** | **0.599 ± 0.010** | **1.617** | 0.215 | **0.101** |

- **3.3%** relative reduction in Hierarchical Distance Metric (M5 vs M1)
- **4.6%** relative reduction in Malignancy Boundary Error Rate (M5 vs M1)
- **HCM recovered** from 0.260 → 0.101 via post-hoc temperature scaling (M6)
- **Novel finding:** fitted temperatures T̄ = 0.428 < 1 across all folds — OHL-CE induces *overconfidence sharpening*, the inverse of the standard T > 1 correction

> HDM = Hierarchical Distance Metric · MBER = Malignancy Boundary Error Rate · HCM = Hierarchical Calibration Metric · TS = temperature scaling

---

## Repository Structure

```
ohl-mrc/
├── src/ohl_mrc/
│   ├── model.py              # Siamese ResNet-50 + CMA gate (42.4M params)
│   ├── losses.py             # DWSL, APCL, OHLCELoss, FlatCELoss
│   ├── data.py               # MILK10k dataset loading, pairing, encoding
│   ├── taxonomy.py           # ISIC-DX graph, distance matrix, ancestor helpers
│   ├── training.py           # train_one_epoch, evaluate, temperature scaling
│   ├── metrics.py            # HDM, MBER, ECE_L1–L4, HCM, macro-recall
│   ├── hparam.py             # Hyperparameter grid search (fold-1 proxy)
│   ├── cv.py                 # 5-fold cross-validation orchestrator
│   ├── stats.py              # Wilcoxon tests, Bonferroni correction
│   ├── figures.py            # Publication figures
│   └── figures_supplementary.py
├── notebooks/
│   └── OHL_MRC_Colab.ipynb   # Self-contained Google Colab notebook
├── configs/
│   └── default.yaml          # All hyperparameters
├── scripts/
│   └── run_experiment.py     # CLI entry point
├── requirements.txt
└── README.md
```

---

## Dataset

This repository requires the **MILK10k** dataset, publicly available at:

- **ISIC Archive:** https://doi.org/10.34970/648456
- **Harvard Dataverse:** https://doi.org/10.7910/DVN/FSXRAQ

> Tschandl P, Akay BN, Rosendahl C, et al. MILK10k: A hierarchical multimodal imaging-learning toolkit for diagnosing pigmented and nonpigmented skin cancer and its simulators. *J Invest Dermatol.* 2026;146(2):357–364. https://doi.org/10.1016/j.jid.2025.06.1594

Download and arrange files as:

```
/path/to/milk10k/data/
├── MILK10k_Training_GroundTruth.csv
├── MILK10k_Training_Metadata.csv
├── MILK10k_Training_Supplement.csv
├── MILK10k_Test_Metadata.csv
├── images_train/
└── images_test/
```

---

## Quickstart

### Option A — Google Colab (recommended)

Open `notebooks/OHL_MRC_Colab.ipynb` in Google Colab with an **A100 runtime**. Mount your Google Drive, point `DATA_ROOT` to your MILK10k folder, and run cells top to bottom.

Estimated runtime: ~8 hours for full 5-fold CV across all 6 conditions.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mynoddin/ohl-mrc/blob/main/notebooks/OHL_MRC_Colab.ipynb)

### Option B — Local / HPC

```bash
# 1. Clone
git clone https://github.com/[YOUR_USERNAME]/ohl-mrc.git
cd ohl-mrc

# 2. Install
pip install -r requirements.txt

# 3. Set data path
nano configs/default.yaml     # set data_root: /path/to/milk10k/data

# 4. Run hyperparameter search (fold-1 only, ~30 min on A100)
python scripts/run_experiment.py --stage hparam

# 5. Run full 5-fold CV (~8 hours on A100)
python scripts/run_experiment.py --stage cv

# 6. Generate figures
python scripts/run_experiment.py --stage figures
```

---

## Experimental Conditions

Six ablation conditions isolate each framework component:

| ID | Loss | CMA gate | Temperature scaling | Description |
|----|------|:---:|:---:|-------------|
| M1 | Flat CE | ✓ | — | Baseline |
| M2 | Flat CE | ✓ | ✓ | Baseline + TS |
| M3 | Flat CE | ✓ | — | Baseline + CMA |
| M4 | DWSL only | ✓ | — | Without APCL |
| M5 | **OHL-CE** | ✓ | — | **Proposed** |
| M6 | **OHL-CE** | ✓ | ✓ | **Proposed + TS** |

To run a single condition:

```bash
python scripts/run_experiment.py --stage cv --condition M5
```

The CV stage is **checkpoint-aware** — if interrupted, re-running skips completed fold/condition pairs automatically.

---

## Hyperparameters

All hyperparameters are in `configs/default.yaml`. The paper uses:

```yaml
loss:
  tau: 2.0      # DWSL temperature (grid-searched: τ ∈ {0.5, 1.0, 2.0, 4.0, 8.0})
  lambda: 0.1   # APCL coefficient (grid-searched: λ ∈ {0.1, 0.25, 0.5, 1.0})

training:
  epochs: 100
  lr: 1.0e-4
  weight_decay: 1.0e-4
  batch_size: 32
  seed: 42

cv:
  n_folds: 5
  img_size: 224
```

---

## Metrics

| Metric | Description |
|--------|-------------|
| **HDM** | Hierarchical Distance Metric — average shortest-path distance between true and predicted ISIC-DX classes (lower = better) |
| **MBER** | Malignancy Boundary Error Rate — fraction of predictions crossing the malignant/benign boundary *(novel metric, this work)* |
| **HCM** | Hierarchical Calibration Metric — weighted ECE across 4 hierarchy levels *(novel metric, this work)* |
| **ECE_L1–L4** | Expected Calibration Error at each ISIC-DX hierarchy level |
| Macro-Recall | Unweighted mean per-class recall across 48 ISIC-DX classes |
| Top-1 / Top-3 | Standard accuracy metrics |

---

## Requirements

```
torch>=2.0
torchvision>=0.15
networkx>=3.0
scikit-learn>=1.3
statsmodels>=0.14
seaborn>=0.12
matplotlib>=3.7
pandas>=2.0
Pillow>=9.0
tqdm
scipy
pyyaml
numpy>=1.24
```

Install all at once:

```bash
pip install -r requirements.txt
```

Hardware used: NVIDIA A100-SXM4-40 GB (Google Colab Pro). Experiments also run on any CUDA GPU; reduce batch size if memory is limited.

---

## Availability

> **This repository will be made fully public upon acceptance of the associated manuscript, or earlier upon request from reviewers.**
>
> If you are a reviewer and wish to access the code during the review process, please contact the corresponding author at mmynoddin@gmail.com with your affiliation.

Trained model weights for all 6 conditions × 5 folds will be released alongside the code.

---

## Citation

If you use this code, the OHL-MRC framework, or the MBER/MRC-Eval metrics, please cite:

```bibtex
@article{mynoddin2025ohlmrc,
  title   = {Ontology-guided hierarchical loss with multi-resolution
             calibration for fine-grained skin lesion classification
             on {MILK10k}},
  author  = {Mynoddin, Md and Chakraborty, Dulal and Dev, Troyee},
  journal = {Journal of Investigative Dermatology},
  year    = {2025},
  note    = {Under review}
}
```

Please also cite the MILK10k dataset and the ISIC-DX taxonomy:

```bibtex
@article{tschandl2026milk10k,
  title   = {{MILK10k}: {A} hierarchical multimodal imaging-learning toolkit
             for diagnosing pigmented and nonpigmented skin cancer
             and its simulators},
  author  = {Tschandl, Philipp and Akay, Bengu Nisa and Rosendahl, Cliff
             and Rotemberg, Veronica and others},
  journal = {J Invest Dermatol},
  volume  = {146},
  number  = {2},
  pages   = {357--364.e7},
  year    = {2026},
  doi     = {10.1016/j.jid.2025.06.1594}
}

@article{scope2025isicdx,
  title   = {International {Skin Imaging Collaboration}-{Designated Diagnoses}
             ({ISIC-DX}): consensus terminology for lesion diagnostic labeling},
  author  = {Scope, Alon and Liopyris, Konstantinos and Weber, Jochen
             and others},
  journal = {J Eur Acad Dermatol Venereol},
  volume  = {39},
  number  = {1},
  pages   = {117--125},
  year    = {2025},
  doi     = {10.1111/jdv.20055}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

**Md Mynoddin**  
Department of Computer Science and Engineering  
Rangamati Science and Technology University, Bangladesh  
✉ mmynoddin@gmail.com
