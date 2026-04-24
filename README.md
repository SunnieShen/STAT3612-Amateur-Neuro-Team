# STAT3612 Group Project: Multimodal Presurgical Brain Tumor Classification

> **Course**: STAT3612 Statistical Machine Learning, Spring 2026, HKU  
> **Group**: Group 7（Amateur Neuro Team）  
> **Task**: 五分类脑肿瘤术前分型（多模态）  
> **Evaluation**: Macro-F1（与 Kaggle 课程竞赛排行一致）

## 仓库结构（与 `report/`、`submission/` 终版一致）

```
STAT3612-Amateur-Neuro-Team/
│
├── README.md
├── requirements-notebook.txt          ← Notebook 依赖（torch / sklearn / xgboost 等）
│
├── proposal/                          ← 开题与队内流程
│   ├── 0. proposal_workflow.md
│   ├── 1. proposal_draft.md
│   └── proposal-Amateur-Neuro-Team.pdf
│
├── report/                            ← 书面报告与图表（终稿 Markdown + 插图）
│   ├── final_report.md                ← 与提交 PDF 对应的主报告（含 Executive Summary、全文与参考文献）
│   ├── report_final.md                ← 与 notebook 章节锚点对齐的摘要版（便于对照代码）
│   ├── report_draft.md                ← 早期草稿
│   ├── report.md                      ← 报告片段/导出用
│   └── fig*.png, figure_*.png, eda_flowchart.png …
│
├── submission/                        ← 课程提交终版（文件名以 Group7 为前缀）
│   ├── Group7_report.pdf
│   ├── Group7_proposal.pdf
│   ├── Group7_slide.pdf
│   └── Group7_notebook.ipynb          ← 与终版报告一致的完整复现 notebook
│
├── dataset/                           ← 竞赛数据布局（说明见 dataset/README.md）
│   ├── README.md
│   ├── kaggle-dataset/                ← train.json, val.json 与训练/验证侧多模态文件
│   └── new_test/                      ← test.json、sample_submission 与测试侧文件
│
└── notebooks/                         ← 开发过程中的 notebook 副本与快照
    ├── pipeline_0420.ipynb          ← 与 submission 主线结构相同的开发版（便于本地迭代）
    ├── pipeline_0422.ipynb, pipeline_0422(1).ipynb
    ├── pipeline_macro_0418.ipynb, pipeline_macro.ipynb, pipeline_micro.ipynb
    └── submission.csv               ← 由 notebook 写出的默认提交 CSV（路径依运行目录而定）
```

**数据目录（与 `dataset/README.md` 一致）**：训练/验证放在 `dataset/kaggle-dataset/`，测试放在 `dataset/new_test/`；若将全部 split 解压到同一目录，也可使用根目录下的 `kaggle-dataset/`（`train.json` / `val.json` / `test.json` 并列）。Notebook 会向上查找 `dataset/kaggle-dataset` 或 `kaggle-dataset`。

## Quick Start

```bash
git clone https://github.com/SunnieShen/STAT3612-Amateur-Neuro-Team.git
cd STAT3612-Amateur-Neuro-Team

pip install -r requirements-notebook.txt
pip install jupyter imbalanced-learn

# 按 dataset/README.md 放置 Kaggle/课程提供的解压数据

# 推荐：打开提交终版 notebook（与 Group7_report.pdf / final_report.md 一致）
jupyter notebook submission/Group7_notebook.ipynb
```

本地开发也可运行 `notebooks/pipeline_0420.ipynb`（与终版 notebook 为同一套「STAT3612: Multimodal Brain Tumor Classification — Full Pipeline」结构；二者体积与执行缓存可能不同）。

**数据路径**：notebook 会优先解析 `dataset/kaggle-dataset/train.json` 与 `val.json`，测试集为同级 `test.json` 或 `dataset/new_test/test.json`。可通过环境变量显式指定，例如 `STAT3612_TRAIN_VAL_DIR`（详见 notebook 中数据加载单元格）。

## 数据划分（与 `report/final_report.md` 一致）

| Split | Cases | 比例（约） |
|-------|------:|-----------|
| Train | 1,983 | 75.0% |
| Validation | 283 | 10.7% |
| Test | 378 | 14.3% |

可分析样本量以报告为准：共 **2,644** 例可访问数据（总队列 **2,838**）。约 **19.7%** 训练病例缺失至少一种影像模态；JSON 中人口学字段缺失约 **69.6%**，临床 CSV 用于补全 Sex 等（详见终稿 EDA 小节）。

## Pipeline 概览（与 `submission/Group7_notebook.ipynb` 首页表格一致）

| Section | Proposal 阶段 | 内容 |
|---------|---------------|------|
| **0. Setup & EDA** | Preliminary Findings | 环境、加载、类别不平衡、缺失与多模态不完整 |
| **1. Feature Engineering** | Data Processing | 四模态加载与编码、PCA、标准化 |
| **2. Single-Modality Baselines** | Stage 1 | 各模态独立基线 |
| **3. Multimodal Fusion** | Stage 2 | 早期拼接融合与晚期 voting/stacking |
| **4. Optimization** | Stage 3 | 特征选择、PCA 扫描、SVM/XGB 等调参 |
| **5. Model Analysis** | Analysis Plan | 模型对比、模态消融、可解释性 |
| **6. Kaggle Submission** | — | 生成提交文件 |

**主指标**：**Macro-F1**（与 Kaggle 多分类 F1 一致）。**辅指标**：Weighted-F1、Accuracy、各类别 Precision/Recall/F1 等。定量结论、调参后验证分数与 Scheme A 严格验证协议见 **`report/final_report.md`** 与 **`submission/Group7_report.pdf`**。

## 数据模态（与终稿 Feature Engineering / EDA 描述一致）

| Modality | 来源 | 维度要点 | 说明 |
|----------|------|------------|------|
| Deep image features | `image_features/` | 2048 × 4 序列 | 缺失模态零填充 |
| Radiomics | `radiomics_info/` | 5 × 4 = 20 | PyRadiomics；缺失 NaN→0 |
| Clinical | `clinical_information/` | 24 | 性别、年龄、信号强度、部位等 |
| Text | JSON `report` | TF-IDF 500（含 uni/bi-gram 等设定） | 影像学报告文本 |

## 关键文件索引

| 路径 | 用途 |
|------|------|
| `dataset/README.md` | 数据目录结构与放置步骤 |
| `proposal/1. proposal_draft.md` | 开题 Markdown 源稿 |
| `report/final_report.md` | 最终报告正文（与 `submission/Group7_report.pdf` 对应） |
| `report/report_final.md` | 与 notebook 章节标题对照的精简版 |
| `submission/Group7_notebook.ipynb` | 课程提交用完整 pipeline |
| `notebooks/pipeline_0420.ipynb` | 同结构开发副本 |
| `dataset/new_test/sample_submission.csv` 或 `dataset/kaggle-dataset/sample_submission.csv` | Kaggle 提交模板 |
| `notebooks/submission.csv` | 本地运行 notebook 时常见默认输出路径 |

## 参考文献（与 `report/final_report.md` 第 13 节一致）

1. Chicco, D., & Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. *BMC Genomics*, 21(1), 1–13.
2. He, K., et al. (2016). Deep residual learning for image recognition. *CVPR*.
3. Price, M., et al. (2024). CBTRUS statistical report. *Neuro-Oncology*, 26(Suppl 6), vi1–vi85.
4. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513–523.
5. Soenksen, L. R., et al. (2022). Integrated multimodal artificial intelligence framework for healthcare applications. *npj Digital Medicine*, 5, 149.

正文另引 **Wang et al. (2024)** 等；完整参考文献以 **`report/final_report.md`** 第 13 节与 **`submission/Group7_report.pdf`** 为准。
