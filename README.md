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
├── dataset/                             ← Competition data layout（说明见 dataset/README.md）
│   ├── README.md                        ← 如何放置 train/val 与 test、目录约定
│   ├── kaggle-dataset/                  ← train.json, val.json 与训练/验证侧多模态文件
│   └── new_test/                        ← test.json、sample_submission 与测试侧文件（可与上者分开发布）
│
├── notebooks/
│   ├── pipeline_macro.ipynb             ← 主流程（Macro-F1 对齐排行榜）
│   └── pipeline_micro.ipynb             ← 同上，指标为 Micro-F1
│
└── baseline_backup/                     ← Historical backups (not for submission)
    ├── brain_tumor_pipeline.ipynb
    ├── project_pipeline_backup.ipynb
    └── ...
```

**可选（旧版单目录）**：若把全部 split 解压到同一文件夹，也可使用仓库根目录下的 `kaggle-dataset/`（内含 `train.json` / `val.json` / `test.json` 并列）；notebook 会自动识别 `dataset/kaggle-dataset` 或 `kaggle-dataset`。

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/SunnieShen/STAT3612-Amateur-Neuro-Team.git
cd STAT3612-Amateur-Neuro-Team

# 2. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm jupyter

# 3. 按 dataset/README.md 放置 Kaggle/课程提供的解压数据

# 4. Run the pipeline
jupyter notebook notebooks/pipeline_macro.ipynb
```

Pipeline notebook 会从当前工作目录向上查找 **`dataset/kaggle-dataset/train.json`**，并在同级 **`dataset/new_test/test.json`** 存在时自动作为测试数据根目录（无需手动改路径）。

## Pipeline Overview

主 notebook（`pipeline_macro.ipynb` / `pipeline_micro.ipynb`）对应提案中的阶段划分如下：

| Section | Purpose |
|---|---|
| **0. EDA** | Class imbalance, multimodal incompleteness, missing demographics |
| **1. Feature Engineering** | Load 4 modalities → PCA → standardize → fused vector |
| **2. Stage 1: Single-Modality Baselines** | Per-modality classifiers |
| **3. Stage 2: Multimodal Fusion** | Early vs late fusion |
| **4. Stage 3: Optimization** | Feature selection, imbalance *(optional / TODO)* |
| **5. Model Analysis** | Comparison, ablation *(SHAP TODO)* |
| **6. Submission** | Generate `submission.csv` for Kaggle |

## Data Modalities

| Modality | Source | Dimensions | Notes |
|---|---|---|---|
| Deep image features | `image_features/` | 2048 × 4 sequences | Zero-padded if modality missing |
| Radiomics | `radiomics_info/` | 5 × 4 sequences = 20 | PyRadiomics; fillna(0) for missing |
| Clinical | `clinical_information/` | 24 | Sex, Age, signal intensities, location keywords |
| Text | JSON `report` | TF-IDF 500 | Radiology findings |

## Key Files

| File | Purpose |
|---|---|
| `dataset/README.md` | 数据目录结构与放置步骤 |
| `1. proposal_draft.md` | Project proposal (submit as PDF) |
| `notebooks/pipeline_macro.ipynb` | Reproducible pipeline |
| `dataset/new_test/sample_submission.csv` 或 `dataset/kaggle-dataset/sample_submission.csv` | Submission template |

## References

1. Price M, et al. CBTRUS statistical report. *Neuro-oncology*, 2024.
2. He K, et al. Deep residual learning for image recognition. *CVPR*, 2016.
3. Lundberg S, Lee S. SHAP: A unified approach to interpreting model predictions. *NeurIPS*, 2017.
4. Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. Springer, 2009.
