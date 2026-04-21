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
│── report_draft.md                      ← Draft report notes
│── report_final.md                      ← Final report markdown
│── requirements-notebook.txt            ← Notebook dependencies
│
├── dataset/                             ← Competition data layout（说明见 dataset/README.md）
│   ├── README.md                        ← 如何放置 train/val 与 test、目录约定
│   ├── kaggle-dataset/                  ← train.json, val.json 与训练/验证侧多模态文件
│   └── new_test/                        ← test.json、sample_submission 与测试侧文件（可与上者分开发布）
│
├── notebooks/
│   ├── pipeline_0420.ipynb              ← 当前主流程（推荐运行）
│   ├── pipeline_macro_0418.ipynb        ← 旧版主流程快照
│   ├── pipeline_macro.ipynb             ← 旧版 Macro-F1 流程
│   ├── pipeline_micro.ipynb             ← 旧版 Micro-F1 流程
│   └── submission.csv                   ← 当前导出的提交文件
```

**可选（旧版单目录）**：若把全部 split 解压到同一文件夹，也可使用仓库根目录下的 `kaggle-dataset/`（内含 `train.json` / `val.json` / `test.json` 并列）；notebook 会自动识别 `dataset/kaggle-dataset` 或 `kaggle-dataset`。

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/SunnieShen/STAT3612-Amateur-Neuro-Team.git
cd STAT3612-Amateur-Neuro-Team

# 2. Install dependencies
pip install -r requirements-notebook.txt
pip install pandas matplotlib seaborn imbalanced-learn jupyter

# 3. 按 dataset/README.md 放置 Kaggle/课程提供的解压数据

# 4. Run the pipeline (current)
jupyter notebook notebooks/pipeline_0420.ipynb
```

`pipeline_0420.ipynb` 会自动查找数据目录：
- 优先在当前目录或上级目录定位 `dataset/kaggle-dataset/train.json` + `val.json`（也兼容单目录 `kaggle-dataset/`）。
- 测试集会自动识别 `train/val` 同目录下的 `test.json`，或兄弟目录 `dataset/new_test/test.json`。
- 也可通过环境变量显式指定（如 `STAT3612_TRAIN_VAL_DIR`）。

## Pipeline Overview

当前主 notebook（`pipeline_0420.ipynb`）对应提案中的阶段划分如下：

| Section | Purpose |
|---|---|
| **0. Setup & EDA** | Environment check, split loading, class imbalance, multimodal incompleteness |
| **1. Feature Engineering** | Load 4 modalities → image PCA (256) → standardize → fused vector |
| **2. Stage 1: Single-Modality Baselines** | Per-modality classifiers |
| **3. Stage 2: Multimodal Fusion** | Early vs late fusion |
| **4. Stage 3: Optimization** | Feature selection / PCA sweep / SVM & XGB tuning / imbalance strategies |
| **5. Model Analysis** | Model comparison, confusion matrix, modality ablation, feature importance |
| **6. Submission** | Generate `submission.csv` for Kaggle |

> 评估指标：主指标为 **Macro-F1**（与 Kaggle 多分类 F1 排行一致），辅指标包括 Weighted-F1 / Accuracy / per-class metrics。

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
| `notebooks/pipeline_0420.ipynb` | 当前可复现实验主流程 |
| `notebooks/pipeline_macro.ipynb` / `notebooks/pipeline_micro.ipynb` | 旧版流程备份 |
| `dataset/new_test/sample_submission.csv` 或 `dataset/kaggle-dataset/sample_submission.csv` | Submission template |
| `notebooks/submission.csv` | `pipeline_0420.ipynb` 默认输出的提交文件 |

## References

1. Price M, et al. CBTRUS statistical report. *Neuro-oncology*, 2024.
2. He K, et al. Deep residual learning for image recognition. *CVPR*, 2016.
3. Lundberg S, Lee S. SHAP: A unified approach to interpreting model predictions. *NeurIPS*, 2017.
4. Hastie T, Tibshirani R, Friedman J. *The Elements of Statistical Learning*. Springer, 2009.
