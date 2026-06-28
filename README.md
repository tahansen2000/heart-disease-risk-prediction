# Heart Disease Risk Prediction & Patient Clustering

Predicting heart disease diagnosis and identifying patient risk clusters using Neural Networks and K-Means on the UCI Heart Disease dataset.

## Overview

This project applies both supervised and unsupervised machine learning to the **UCI Heart Disease dataset** (920 patient records) to answer two questions:

1. **Can patient health metrics predict a positive heart disease diagnosis?**
2. **Are there natural patient groupings based on health metrics, and do these clusters align with heart disease risk?**

A Neural Network was used to address the first question, and K-Means Clustering was used to address the second — giving both a predictive and a descriptive view of heart disease risk in the data.

## Key Results

- **Neural Network Accuracy: 78%**, with precision/recall balanced to favor catching true positives (fewer false negatives than false positives — clinically the safer error to make)
- **Top predictors** (via permutation importance): ST depression (`oldpeak`), atypical chest pain, maximum heart rate (`thalch`), and exercise-induced angina
- **K-Means identified 3 natural patient clusters**, corresponding to **low (38%)**, **medium (55%)**, and **high (63%)** heart disease risk groups — showing patients can be meaningfully segmented by risk using just their health metrics

## Methods

**Data Cleaning & EDA**
- Feature-by-feature review of 920 observations across 13 original variables
- Missing/invalid values (e.g., resting blood pressure recorded as 0, cholesterol recorded as 0) handled with mean, median, or mode imputation depending on each feature's distribution
- Three features dropped due to excessive missingness (33–66% NaN)
- Final dataset: 10 features, 1 target variable

**Feature Engineering**
- Categorical variables (`sex`, `cp`, `restecg`) converted to dummy variables
- Target variable binarized (any heart disease severity level → positive diagnosis)
- 75/25 train/test split
- **SMOTE** applied to training data to correct class imbalance

**Modeling**
- **Supervised:** Neural Network (chosen for its ability to capture non-linear relationships; interpretability addressed via permutation importance)
- **Unsupervised:** K-Means Clustering (optimal cluster count of 3 selected via the elbow method)

## Tools

- Python
- pandas, NumPy
- scikit-learn
- imbalanced-learn (SMOTE)
- Matplotlib / Seaborn

## Repository Contents

- `[notebook_name].ipynb` — full analysis, code, and visualizations
- `Final Project Report.pdf` — written project summary and methodology

## Author

**Taner Hansen**
Graduate student | Aspiring Data Analyst/Scientist with a background in higher education data analytics
