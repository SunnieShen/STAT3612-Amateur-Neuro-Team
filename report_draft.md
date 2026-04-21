# Draft Report: Multimodal Brain Tumor Classification

## 1. Project Overview

This project builds an end-to-end multimodal classification pipeline for brain tumor subtype prediction, integrating image embeddings, radiomics, clinical variables, and radiology text features. The workflow is organized as a staged modeling process: exploratory analysis, feature engineering, single-modality baselines, multimodal fusion, optimization, and final submission generation.  
**Notebook section:** `# STAT3612: Multimodal Brain Tumor Classification — Full Pipeline`

The core objective is to maximize generalization on minority-sensitive metrics under severe class imbalance, while maintaining a transparent and reproducible experimental setup. Across the notebook, model selection primarily follows stratified 5-fold CV Macro-F1, with held-out validation used as an external consistency check.  
**Notebook section:** `## 0.3 EDA: Class Imbalance (~40:1)`, `## 2.1 Stage 1 — Summary & Visualization`

## 2. Data Understanding and Preprocessing

Dataset splits are fixed at 1983 training cases, 283 validation cases, and 378 test cases. EDA confirms substantial data heterogeneity: strong class imbalance, partial modality incompleteness, and non-trivial missingness in selected demographic fields when extracted from JSON metadata.  
**Notebook section:** `## 0.2 Load Metadata`, `## 0.3 EDA: Class Imbalance (~40:1)`, `## 0.4 EDA: Multimodal Incompleteness (~20% missing 1 modality)`, `## 0.5 EDA: Missing Demographics (JSON ~70% vs CSV ~0%)`

Feature construction follows modality-specific logic. Image features are high-dimensional embeddings aggregated across four MRI modalities with zero padding for missing channels; radiomics are compact handcrafted descriptors; clinical features come from structured records; and text is represented via TF-IDF vectors from reports. PCA is then applied to image embeddings (256 components, explained variance around 0.92), followed by scaling and label encoding.  
**Notebook section:** `## 1.1 Image Features (2048-d × 4 modalities, zero-pad if missing)`, `## 1.2 Radiomics Features (5 × 4 modalities, NaN → 0)`, `## 1.3 Clinical Features (from CSV; demographics sourced from clinical_information)`, `## 1.4 Text Features (TF-IDF from radiology reports)`, `## 1.5 PCA, Standardize, Encode Labels`

## 3. Stage 1: Single-Modality Baselines

Single-modality experiments show that text and clinical signals are substantially stronger than image-PCA-only or radiomics-only settings in this pipeline configuration. On held-out validation, Text achieves Macro-F1 = 0.7027 and Clinical reaches 0.6688, compared with 0.3029 for Image (PCA-256) and 0.2899 for Radiomics.  
**Notebook section:** `## 2.1 Stage 1 — Summary & Visualization`

These baseline outcomes establish two important references: (1) text-derived representations carry strong discriminative content in the current data split, and (2) naive image/radiomics usage remains insufficient without stronger feature selection, fusion design, or model calibration.  
**Notebook section:** `## 2.1 Stage 1 — Summary & Visualization`

## 4. Stage 2: Multimodal Fusion and Model Comparison

The multimodal stage evaluates early-fusion learners (LR, RF, SVM-RBF, XGB, LGB, MLP, and a PyTorch multi-branch MLP) as well as late-fusion ensembles (soft voting and stacking). Comparative reporting includes CV Macro-F1, validation Macro-F1, accuracy, weighted F1, and macro AUROC.  
**Notebook section:** `# 3. Stage 2 — Multimodal Fusion`, `## 3.2 Late Fusion — Voting & Stacking`, `## 3.3 Stage 2 — Comprehensive Result Comparison (Tutorial 7 Style)`

At this stage, SVM-RBF is the strongest held-out model among the main baselines (Val Macro-F1 = 0.7834), with soft voting very close (0.7816). Interestingly, PyTorch multi-branch MLP has the highest CV Macro-F1 (0.7423) but lower validation Macro-F1 (0.7126), indicating a noticeable CV-to-validation performance gap relative to SVM-based solutions.  
**Notebook section:** `## 3.3 Stage 2 — Comprehensive Result Comparison (Tutorial 7 Style)`

## 5. Stage 3: Optimization

Feature selection optimization improves SVM-RBF further: the `L1 top-128` configuration reaches Val Macro-F1 = 0.7853, slightly outperforming the original SVM baseline. This optimized variant becomes the best model in downstream global comparison.  
**Notebook section:** `## 4.1.4 Optimization Schemes to SVM-RBF`, `## 5.1 Model Comparison Summary (Early + Late Fusion)`

XGBoost optimization also shows meaningful gains over its untuned baseline, with the optimized variant (`MI top-256`) reported at Val Macro-F1 = 0.7457. While still below the top SVM models, this demonstrates that feature-space refinement is beneficial for tree-based methods in this task.  
**Notebook section:** `## 4.1.5 Optimization Schemes to XGBoost`, `## 5.1 Model Comparison Summary (Early + Late Fusion)`

Class-imbalance mitigation (Section 4.2) and hyperparameter tuning (Section 4.3) have been implemented in code structure, but their new outputs are currently not persisted in notebook cell outputs. These sections should be re-executed before final report locking, so that final numeric comparisons can be inserted.  
**Notebook section:** `### 4.2 Class Imbalance Mitigation`, `### 4.3 Hyperparameter Tuning`

## 6. Model Analysis

The final ranked summary reports `SVM-RBF (Optimized 4.1: L1 top-128)` as the best validation model (Val Macro-F1 = 0.7853), followed by baseline SVM-RBF and soft voting. This indicates that careful feature compression/selection with a margin-based classifier is currently the most reliable path in this notebook.  
**Notebook section:** `## 5.1 Model Comparison Summary (Early + Late Fusion)`

Modality ablation with LightGBM highlights that, in isolation, text performs best (0.7027), followed by clinical (0.6688), whereas image-only and radiomics-only settings are much weaker. In this specific ablation setup, simply combining all modalities does not automatically yield the best Macro-F1, suggesting that fusion quality depends strongly on representation alignment and model choice rather than raw feature concatenation.  
**Notebook section:** `## 5.2 Modality Ablation Study`

Feature-importance source breakdown provides complementary interpretability: RF Top-30 is dominated by text features (22/30), while LGB Top-30 allocates substantial weight to image features (19/30). This discrepancy suggests different inductive biases across tree models and supports a multi-model analysis perspective instead of relying on a single interpretation view.  
**Notebook section:** `## 5.3 Feature Importance & Interpretability (Tutorial 7/8 Style)`

## 7. Current Conclusion and Next Actions

The current evidence supports optimized SVM-RBF as the primary candidate for final submission, with soft voting as a competitive backup. The strongest immediate improvement opportunity is to re-run and document Sections 4.2 and 4.3 so imbalance and tuning impacts are quantitatively integrated into the final leaderboard and discussion.  
**Notebook section:** `### 4.2 Class Imbalance Mitigation`, `### 4.3 Hyperparameter Tuning`, `# 6. Generate Kaggle Submission`

After re-execution, the final report should add: (1) a compact table for imbalance strategy comparison, (2) tuned-parameter tables for SVM/LGB with CV and validation metrics, and (3) updated final-model justification tied to both performance and robustness under class imbalance.  
**Notebook section:** `### 4.2 Class Imbalance Mitigation`, `### 4.3 Hyperparameter Tuning`, `## 5.1 Model Comparison Summary (Early + Late Fusion)`
