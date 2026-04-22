### 1. Executive Summary
This project develops a statistical machine learning framework for the presurgical classification of five brain tumor subtypes, leveraging a diverse multimodal dataset of 2,838 cases. Confronted with a severe 40:1 class imbalance and ~20% missing modality rates, we implemented a three-stage pipeline integrating MRI-derived image embeddings, radiomics, clinical variables, and radiology report text (TF-IDF). Cross-validated benchmarking demonstrates that text and clinical features consistently outperform image-based signals in isolation, with SVM-RBF under L1-based feature selection (top-128) achieving the highest validation Macro-F1 of 0.7853. Feature importance analysis pinpoints clinical location variables and TF-IDF tokens as the primary discriminative predictors,confusion analysis further reveals that Brain Metastase Tumour and Glioma present the most significant classification overlap due to similar radiological patterns. These results validate the clinical necessity of fusing unstructured radiology reports with structured demographic and clinical data, providing radiologists with an interpretable, data-driven reference to reduce diagnostic uncertainty in presurgical planning.
### **2. Introduction**

#### **2.1 Background and Clinical Significance**
Accurate presurgical classification of brain tumor subtypes is a critical determinant of neurosurgical strategy and patient prognosis. The five subtypes addressed in this study—Glioma, Meningioma, Brain Metastase Tumour, Tumors of the sellar region, and Pineal tumour and Choroid plexus tumour—vary substantially in their biological behavior and recommended treatment pathways (Price et al., 2024). For instance, while gliomas typically require maximal safe surgical resection combined with adjuvant chemoradiotherapy, meningiomas may be managed conservatively, and brain metastases often necessitate systemic treatment targeting the primary malignancy. Therefore, reliable preoperative classification is pivotal for guiding clinical decisions and avoiding unnecessary interventions.

In routine practice, radiologists synthesize multi-sequence MRI findings with patient demographics. However, manual interpretation remains challenging due to overlapping imaging characteristics and inter-observer variability (Wang et al., 2024). Automating this process via statistical machine learning offers a data-driven reference to reduce diagnostic uncertainty in complex presurgical planning.
#### **2.2 Dataset Overview**  
We used a curated multimodal dataset from the course Kaggle competition, containing 2,644 accessible cases from a total cohort of 2,838 patients. The data is split into training (1,983 cases, 75.0%), validation (283 cases, 10.7%), and test sets (378 cases, 14.3%). The label distribution is consistent across splits, dominated by Glioma (46.6%) and Meningioma (36.7%), followed by Brain Metastase (12.7%), Sellar region (2.8%), and Pineal/Choroid plexus tumours (1.2%). This confirms that stratified partitioning was effective. Notably, the dataset reflects real-world incompleteness: about 19.7% of training cases lack one imaging modality, and demographic data is missing in 69.6% of JSON records, though we recovered the latter using accompanying clinical CSV files.

#### **2.3 Core Challenges**  
The project addresses three main technical challenges. First, Extreme Class Imbalance is a major hurdle; the largest class (Glioma) outnumbers the smallest (Pineal/Choroid plexus) by a 40:1 ratio. This makes overall accuracy misleading, as a naive classifier could score well while failing on rare tumors. We therefore use Macro-F1 as the primary metric to ensure equal weighting for all classes (Chicco & Jurman, 2020).

Second, Multimodal Incompleteness mirrors clinical reality, with nearly 20% of cases missing a modality. Instead of discarding these samples—which would bias the model—we use zero-padding to maintain sample size and consistency (Soenksen et al., 2022).

Third, Cross-modality Heterogeneity presents a fusion challenge. The data comes from vastly different feature spaces: high-dimensional visual embeddings (ResNet; He et al., 2016), radiomics descriptors, categorical clinical variables, and sparse TF-IDF text vectors. Our three-stage pipeline is designed to fuse these heterogeneous inputs without losing discriminative signals.
#### **2.4 Report Roadmap**
The report is organized to follow our development workflow. Section 3 performs exploratory data analysis on class and modality distributions. Section 4 details feature engineering, including PCA-based dimensionality reduction for image embeddings. Section 5 establishes single-modality baselines to quantify independent predictive power. Section 6 presents multimodal fusion strategies, comparing early concatenation against late fusion via stacking and voting. Section 7 describes the optimization process, covering L1-based feature selection and imbalance mitigation. Section 8 provides model analysis, including feature importance and error analysis, followed by a discussion of limitations in Section 9.

### 3. Exploratory Data Analysis

In this section, we examine the multimodal dataset to identify patterns and anomalies that directly inform our preprocessing and modeling decisions. Figure 1 illustrates how each finding connects to the subsequent pipeline stages.

![[eda_flowchart.png]]
#### 3.1 Class Distribution

![[fig2_class_dist.png]]
The training set contains 1,983 cases distributed across five classes. Glioma dominates with 924 cases (46.6%), followed by Meningioma (728, 36.7%), Brain Metastase Tumour (252, 12.7%), Tumors of the sellar region (56, 2.8%), and Pineal/Choroid plexus tumour (23, 1.2%). The largest-to-smallest ratio is 40:1.

A model trained naively on this data would learn to predict Glioma by default. It could achieve over 46% accuracy while failing entirely on the rarest classes — which is the worst possible outcome when rare tumor misdiagnosis carries the highest clinical cost. We therefore adopted Macro-F1 as our primary evaluation metric, which treats all five classes equally regardless of size (Chicco & Jurman, 2020). This finding also directly motivated our Stage 3 experiments on imbalance mitigation, where we compared class-weighted training, random oversampling, and SMOTE.
#### 3.2 Modality Completeness

![[fig3_modality 1.png]]

About one in five cases across all splits is missing at least one MRI sequence. The test set is more variable — beyond the expected ~21% with one missing modality, 19 cases have two or three sequences absent. This reflects clinical reality: scans get skipped due to patient contraindications, scanner errors, or time pressure.

Discarding these cases was not an option. For minority classes with fewer than 60 training samples, losing even a handful of cases would meaningfully distort the class distribution further. We applied zero-padding to replace missing modality features with zeros, keeping all cases in the pipeline while maintaining consistent input dimensions (Soenksen et al., 2022).
#### 3.3 Patient Demographics by Tumor Subtype

Despite high overall missingness in demographic fields — 69.6% of cases lack recorded Sex and Age in the JSON source — the clinical CSV files provided complete Sex records for all 1,983 training cases, with Age available for a meaningful subset. We analysed the known demographic distributions to understand whether different tumor types present in distinguishable patient populations.

![[fig4_demographics 1.png]]

Age patterns show clear separation across subtypes. Pineal/Choroid plexus tumours present in the youngest patients (mean age 33.8, range 5–66), consistent with their known occurrence in children and young adults. Gliomas span the widest age range (mean 45.3, range 3–76), reflecting the heterogeneous nature of this group. Meningiomas and Brain Metastases tend to appear in older patients, with means of 53.8 and 55.0 respectively. These differences suggest that age carries genuine discriminative signal, particularly for separating the rare pediatric-skewed classes from the adult-dominant majority.

Sex patterns among cases with known values show that Meningioma is disproportionately female (63% female among known cases), a pattern well-established in clinical literature. Glioma and Brain Metastase lean slightly male (56% and 59%). These patterns, while based on a subset of known cases, are consistent with epidemiological expectations and confirm that Sex is a meaningful clinical variable worth retaining in our feature set.

Given the high missingness in both fields, we filled missing Age values with the training-set median and used the CSV as the authoritative source for Sex. Both features were retained as part of the 24-dimensional clinical feature vector.
#### 3.4 Radiomics Feature Redundancy

PyRadiomics extracts five features per MRI sequence — first-order Mean, Entropy, 90th Percentile, GLCM Contrast, and GLCM JointEntropy — giving 20 features across four modalities. We found four highly correlated pairs, all involving Entropy and JointEntropy:

![[fig5_radiomics_corr 2.png]]

These pairs are effectively measuring the same signal. Including both inflates the feature space without adding information, and distance-based models like SVM are particularly sensitive to this redundancy. We dropped one feature from each correlated pair, reducing the radiomics set from 20 to 16. This de-correlation step became one of the tested optimization strategies in Stage 3.

#### 3.5 Summary

Four findings, four decisions. The 40:1 class imbalance set Macro-F1 as our evaluation metric and put imbalance mitigation on the Stage 3 agenda. The 20% modality missing rate led to zero-padding rather than sample removal. The demographic analysis confirmed that age and sex carry genuine discriminative signal and are worth recovering from the CSV source despite high missingness. The radiomics redundancy flagged four feature pairs for removal before fusion. Each modeling choice in the sections that follow traces back to something we found here first.
