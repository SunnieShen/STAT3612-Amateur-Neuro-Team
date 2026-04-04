# STAT3612 Group Project: Multimodal Presurgical Brain Tumor Classification

> **Course**: STAT3612 Statistical Machine Learning, Spring 2026, HKU  
> **Team**: Amateur Neuro Team  
> **Task**: 5-class brain tumor classification using multimodal data  
> **Evaluation**: F1 score (Kaggle competition)

## Repository Structure

```
STAT3612-Amateur-Neuro-Team/
│
│── README.md                            ← You are here
│
│── 0. proposal_workflow.md              ← Internal team workflow & division of labor
│── 1. proposal_draft.md                 ← Proposal source (Markdown)
│── 1. proposal_draft.docx               ← Proposal export (Word, for submission)
│── STAT3612_project-2026_Updated.pdf    ← Course project description
│
├── notebooks/
│   └── full_pipeline.ipynb              ← Main notebook (end-to-end pipeline)
│
├── kaggle-dataset/                      ← Competition data (not committed to Git)
│   ├── train.json / val.json / test.json    ← Metadata + labels (train/val only)
│   ├── sample_submission.csv                ← Kaggle submission template
│   ├── image_features/                      ← ResNet deep features (.npy, 2048-d)
│   │   └── image_features/<case_id>/<modality>/image.npy
│   ├── radiomics_info/                      ← PyRadiomics features (CSV, 5 per modality)
│   │   ├── train/ val/ test/
│   │   └── README.md                        ← Feature descriptions
│   ├── clinical_information/                ← Demographics + report-derived fields (CSV)
│   │   └── {train,val,test}_patient_info.csv
│   └── original_raw_report/                 ← Raw radiology report text (CSV)
│       └── {train,val,test}_patient_info.csv
│
└── baseline_backup/                     ← Historical backups (not for submission)
    ├── brain_tumor_pipeline.ipynb
    ├── project_pipeline_backup.ipynb
    └── ...
```

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/SunnieShen/STAT3612-Amateur-Neuro-Team.git
cd STAT3612-Amateur-Neuro-Team

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm jupyter

# 3. Run the pipeline
jupyter notebook notebooks/full_pipeline.ipynb
```

The notebook auto-detects the `kaggle-dataset/` folder regardless of working directory.

## Pipeline Overview

The notebook (`full_pipeline.ipynb`) follows a three-stage pipeline aligned with the proposal:

| Section | Purpose |
|---|---|
| **0. EDA** | Class imbalance (40:1), multimodal incompleteness (~20%), missing demographics |
| **1. Feature Engineering** | Load 4 modalities → PCA → standardize → 800-d fused vector |
| **2. Stage 1: Single-Modality Baselines** | Per-modality classifiers to measure individual contributions |
| **3. Stage 2: Multimodal Fusion** | Early fusion (concatenated features) vs Late fusion (voting/stacking) |
| **4. Stage 3: Optimization** | Feature selection, class imbalance (SMOTE), hyperparameter tuning *(TODO)* |
| **5. Model Analysis** | Model comparison, modality ablation, interpretability *(SHAP TODO)* |
| **6. Submission** | Generate `submission.csv` for Kaggle |

## Data Modalities

| Modality | Source | Dimensions | Notes |
|---|---|---|---|
| Deep image features | `image_features/` | 2048 × 4 sequences | ResNet; zero-padded if modality missing |
| Radiomics | `radiomics_info/` | 5 × 4 sequences = 20 | PyRadiomics; fillna(0) for missing |
| Clinical | `clinical_information/` | 24 | Sex, Age, signal intensities, location keywords |
| Text | `original_raw_report/` + JSON `report` | TF-IDF 500 | Radiology findings (impression excluded) |

## Key Files

| File | Purpose |
|---|---|
| `1. proposal_draft.md` | Project proposal (submit as PDF) |
| `notebooks/full_pipeline.ipynb` | Reproducible pipeline (submit as final notebook) |
| `kaggle-dataset/sample_submission.csv` | Submission format template |

## References

1. Price M, et al. CBTRUS statistical report. *Neuro-oncology*, 2024.
2. He K, et al. Deep residual learning for image recognition. *CVPR*, 2016.
3. Lundberg S, Lee S. SHAP: A unified approach to interpreting model predictions. *NeurIPS*, 2017.
4. Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. Springer, 2009.
