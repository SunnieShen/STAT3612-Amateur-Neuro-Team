# Final Report (Draft-Ready): Multimodal Brain Tumor Classification

## Abstract

This report presents an end-to-end multimodal pipeline for brain tumor subtype classification using image embeddings, radiomics, clinical variables, and radiology text features. We adopt a staged experimental design: data auditing, modality-specific feature engineering, single-modality baselines, multimodal fusion, and optimization. Model selection is primarily based on stratified 5-fold cross-validation (CV) Macro-F1, with held-out validation as an external reference. The strongest overall model is an optimized SVM-RBF with L1-based feature selection (`top-128`), achieving validation Macro-F1 of **0.7853**. Results also show strong text and clinical predictive signals, while image/radiomics features require careful integration to contribute consistently.  
**Notebook section:** `# STAT3612: Multimodal Brain Tumor Classification — Full Pipeline`

## 1. Problem Statement and Evaluation Philosophy

The objective is multiclass brain tumor classification under severe class imbalance and partial modality incompleteness. Because imbalance can inflate naive metrics, we prioritize **Macro-F1** as the principal criterion and use stratified CV for robust model ranking. Held-out validation is used to confirm transferability of CV trends.  
**Notebook section:** `## 0.3 EDA: Class Imbalance (~40:1)`, `## 2.1 Stage 1 — Summary & Visualization`

## 2. Data and Preprocessing

The experiment uses fixed train/validation/test splits (1983 / 283 / 378). EDA highlights three practical challenges: skewed class frequency, non-negligible missing modalities, and missing demographic fields in raw JSON sources. These findings motivate robust preprocessing and metric choices.  
**Notebook section:** `## 0.2 Load Metadata`, `## 0.3 EDA: Class Imbalance (~40:1)`, `## 0.4 EDA: Multimodal Incompleteness (~20% missing 1 modality)`, `## 0.5 EDA: Missing Demographics (JSON ~70% vs CSV ~0%)`

Feature pipelines are modality-aware: image embeddings (4 modalities, zero-padded if missing), radiomics descriptors, structured clinical variables, and TF-IDF report text. Image dimensionality is reduced via PCA (256 components, explained variance ~0.92), followed by scaling and label encoding.  
**Notebook section:** `## 1.1 Image Features (2048-d × 4 modalities, zero-pad if missing)`, `## 1.2 Radiomics Features (5 × 4 modalities, NaN → 0)`, `## 1.3 Clinical Features (from CSV; demographics sourced from clinical_information)`, `## 1.4 Text Features (TF-IDF from radiology reports)`, `## 1.5 PCA, Standardize, Encode Labels`

## 3. Baseline Findings (Single Modality)

Single-modality results indicate that text and clinical information dominate in the current setup. On validation Macro-F1, Text reaches **0.7027** and Clinical **0.6688**, while Image (PCA-256) and Radiomics remain much lower (**0.3029** and **0.2899**, respectively).  
**Notebook section:** `## 2.1 Stage 1 — Summary & Visualization`

This baseline pattern suggests that imaging/radiomics features are not intrinsically uninformative, but are harder to exploit without stronger representation alignment or optimization.  
**Notebook section:** `## 2.1 Stage 1 — Summary & Visualization`

## 4. Multimodal Modeling

We evaluate both early fusion (LR, RF, SVM-RBF, XGB, LGB, MLP, PyTorch multi-branch MLP) and late fusion (soft voting, stacking). Comparative reporting includes CV Macro-F1, validation Macro-F1, accuracy, weighted F1, and macro AUROC.  
**Notebook section:** `# 3. Stage 2 — Multimodal Fusion`, `## 3.2 Late Fusion — Voting & Stacking`, `## 3.3 Stage 2 — Comprehensive Result Comparison (Tutorial 7 Style)`

Among base multimodal models, SVM-RBF provides the strongest held-out Macro-F1 (**0.7834**), closely followed by soft voting (**0.7816**). Although the PyTorch multi-branch MLP has high CV Macro-F1, its validation Macro-F1 is lower, indicating a larger CV-to-validation gap than SVM-based configurations.  
**Notebook section:** `## 3.3 Stage 2 — Comprehensive Result Comparison (Tutorial 7 Style)`

## 5. Optimization Results

Feature-selection optimization further improves SVM-RBF. The `L1 top-128` setting reaches validation Macro-F1 **0.7853**, becoming the top model in final comparison.  
**Notebook section:** `## 4.1.4 Optimization Schemes to SVM-RBF`, `## 5.1 Model Comparison Summary (Early + Late Fusion)`

Optimized XGBoost (`MI top-256`) also improves over its baseline and reaches validation Macro-F1 **0.7457**, confirming the value of feature-space refinement for tree-based methods.  
**Notebook section:** `## 4.1.5 Optimization Schemes to XGBoost`, `## 5.1 Model Comparison Summary (Early + Late Fusion)`

Sections `4.2` (class imbalance mitigation) and `4.3` (hyperparameter tuning) are implemented but currently lack persisted notebook outputs after refactoring. They should be re-executed and numerically integrated before final submission.  
**Notebook section:** `### 4.2 Class Imbalance Mitigation`, `### 4.3 Hyperparameter Tuning`

## 6. Analysis and Interpretation

Global model ranking supports optimized SVM-RBF as the primary submission candidate, with baseline SVM-RBF and soft voting as strong alternatives.  
**Notebook section:** `## 5.1 Model Comparison Summary (Early + Late Fusion)`, `# 6. Generate Kaggle Submission`

Modality ablation (LightGBM) shows `Text Only` > `Clinical Only` > `All (Early Fusion)` in this specific configuration, indicating that straightforward concatenation is not guaranteed to maximize Macro-F1. Fusion quality depends on representation compatibility and classifier bias.  
**Notebook section:** `## 5.2 Modality Ablation Study`

Feature-importance source analysis differs by model family: RF top features are largely text-driven, while LGB allocates more top positions to image-derived features. This divergence suggests complementary inductive biases and motivates multi-model interpretability rather than single-model narratives.  
**Notebook section:** `## 5.3 Feature Importance & Interpretability (Tutorial 7/8 Style)`

## 7. Limitations and Immediate Next Steps

Current limitations are primarily procedural rather than conceptual: newly structured Sections `4.2` and `4.3` need re-execution so final numerical evidence can be formally reported.  
**Notebook section:** `### 4.2 Class Imbalance Mitigation`, `### 4.3 Hyperparameter Tuning`

Before report lock, three updates are recommended:  
- add a concise imbalance-strategy comparison table (4.2),  
- add best-parameter/CV/validation tables for tuned SVM and LGB (4.3),  
- refresh final model justification with these updated results.  
**Notebook section:** `### 4.2 Class Imbalance Mitigation`, `### 4.3 Hyperparameter Tuning`, `## 5.1 Model Comparison Summary (Early + Late Fusion)`

## 8. Conclusion

The pipeline demonstrates that robust multimodal classification is achievable when evaluation is aligned with class-imbalance reality and model selection is CV-driven. In the current state, **SVM-RBF with L1 top-128 features** is the most reliable performer and a justified primary submission model. Final confidence in this conclusion will be strengthened after executing and incorporating the updated 4.2/4.3 outputs.  
**Notebook section:** `## 4.1.4 Optimization Schemes to SVM-RBF`, `### 4.2 Class Imbalance Mitigation`, `### 4.3 Hyperparameter Tuning`, `## 5.1 Model Comparison Summary (Early + Late Fusion)`
