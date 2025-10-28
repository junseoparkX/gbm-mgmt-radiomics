# MGMT Methylation Prediction from GBM MRI Radiomics

This repo is about predicting MGMT promoter methylation status in glioblastoma (GBM) using MRI-derived radiomic features.

## What we are doing

- We start from TCGA-GBM subjects with multi-parametric MRI (T1, T1-Gd, T2, FLAIR).
- For each subject we already have a large radiomics feature vector:
  - volumetric features (tumor subregion volumes and ratios),
  - intensity / histogram statistics,
  - spatial location features,
  - texture features (GLCM, GLRLM, GLSZM, NGTDM),
  - morphology (eccentricity, solidity),
  - tumor growth model features (`TGM_*`).
- We also have the ground truth MGMT promoter methylation status for each subject.

Future notebooks (e.g. `model_baseline.ipynb`) will train classifiers (Random Forest, XGBoost, SVM, etc.) and report accuracy, sensitivity, specificity, and ROC AUC.

## Why we are doing it

MGMT promoter methylation is clinically important in GBM because it is linked to treatment response.  
The goal here is to build a noninvasive predictor directly from MRI features.

We are specifically trying to match or beat the published radiogenomic MGMT predictors based on genetic-algorithm feature selection + classical ML.

## Reported reference performance (GBM cohort)

These numbers are from prior work that used a genetic algorithm (GA) to select a subset of radiomic features, then trained a model on that subset.  
Our goal is to meet or outperform this level.

| Model     | Sensitivity | Specificity | Accuracy | AUC   |
|-----------|-------------|-------------|----------|-------|
| GA-RF     | 0.894       | 0.966       | 0.925    | 0.93  |
| GA-XGB    | 0.889       | 0.880       | 0.889    | (–)   |
| GA-SVM    | 0.720       | 0.454       | 0.678    | (–)   |
| XGB-Fscore baseline | – | – | 0.887 | 0.896 |

Notes:
- GA-RF = Random Forest trained on GA-selected features. This is the best reported model (Accuracy 0.925, AUC 0.93).
- XGB-Fscore is a published baseline using XGBoost with F-score feature ranking (Accuracy 0.887, AUC 0.896).
- (–) = not explicitly reported.

## What success means for us

- Higher external test accuracy / AUC than GA-RF.
- More balanced sensitivity & specificity (not just over-calling methylated).
- A clean, reproducible pipeline: data prep notebook + training notebook + saved splits.

In short: same clinical question, same task (MGMT methylation prediction), but we aim to make it simpler to reproduce and ideally push the performance beyond the GA-RF benchmark.
