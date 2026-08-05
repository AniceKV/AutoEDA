# Executive Summary Report: Automated Exploratory Data Analysis (EDA) Pipeline

**Dataset:** Titanic-Dataset.csv  
**Pipeline Output Directory:** Working Directory  
**Target Variable:** Survived (Binary Classification)  
**Report Date:** Current Execution  

---

## 1. Dataset Overview

The automated EDA pipeline processed the Titanic dataset, yielding a structured profile of 891 passenger records across 12 distinct attributes. The dataset contains a mix of numerical, categorical, and high-cardinality text features.

| Attribute | Details |
|---|---|
| Total Rows | 891 |
| Total Columns | 12 |
| Target Column | Survived |
| Problem Type | Classification |
| Source File | Titanic-Dataset.csv |

### 1.1 Schema Summary

| Column | Dtype | Cardinality | Missing Count | Missing % |
|---|---|---|---|---|
| PassengerId | int64 | 891 | 0 | 0.0% |
| Survived | int64 | 2 | 0 | 0.0% |
| Pclass | int64 | 3 | 0 | 0.0% |
| Name | object | 891 | 0 | 0.0% |
| Sex | object | 2 | 0 | 0.0% |
| Age | float64 | 88 | 177 | 19.87% |
| SibSp | int64 | 7 | 0 | 0.0% |
| Parch | int64 | 7 | 0 | 0.0% |
| Ticket | object / float64 | 681 / 514 | 230 (pre-imputation) | 25.8% (pre-imputation) |
| Fare | float64 | 248 | 0 | 0.0% |
| Cabin | object | 147 | 687 | 77.1% |
| Embarked | object | 3 | 2 | 0.22% |

---

## 2. Data Quality & Missing Value Imputation

The pipeline identified three columns with missing values and applied a rule-based imputation strategy to ensure model readiness.

### 2.1 Missing Value Profile (Pre-Imputation)

| Column | Missing Values | Percentage |
|---|---|---|
| Age | 177 | 19.9% |
| Cabin | 687 | 77.1% |
| Embarked | 2 | 0.2% |
| Ticket | 230 | 25.8% |

### 2.2 Imputation Rules Applied

1. Standardized missing string placeholders ('?', 'NA', 'N/A', 'null') to NaN.
2. Numeric columns with skewness > 1.0 or < -1.0 use median imputation.
3. Numeric columns with skewness between -1.0 and 1.0 use mean imputation.
4. Categorical/String columns use mode imputation with 'Unknown' fallback.

### 2.3 Imputation Results

| Column | Method | Skewness | Fill Value | Missing After |
|---|---|---|---|---|
| Age | Median | 0.39 | 28.0 | 0 |
| Ticket | Median | 5.27 | 236171.0 | 0 |
| Cabin | Mode | N/A | B96 B98 | 0 |
| Embarked | Mode | N/A | S | 0 |

---

## 3. Statistical Distributions & Outlier Analysis

Key statistical metrics were computed for numerical features, alongside outlier detection using the Interquartile Range (IQR) method.

### 3.1 Key Statistical Metrics

| Feature | Mean | Median | Range | Skewness |
|---|---|---|---|---|
| Age | 29.70 | 28.00 | [0.42, 80.00] | 0.39 |
| Fare | 32.20 | 14.45 | [0.00, 512.33] | 4.79 |
| SibSp | 0.52 | 0.00 | [0.00, 8.00] | 3.70 |
| Parch | 0.38 | 0.00 | [0.00, 6.00] | 2.75 |
| Pclass | 2.31 | 3.00 | [1.00, 3.00] | N/A |

### 3.2 Outlier Analysis (IQR Method)

| Feature | Q1 | Q3 | IQR | Lower Bound | Upper Bound | Outlier Count | Outlier % | Action |
|---|---|---|---|---|---|---|---|---|
| Age | 22.0 | 35.0 | 13.0 | 2.5 | 54.5 | 66 | 7.41% | Profile |
| Fare | 7.91 | 31.0 | 23.09 | -26.72 | 65.63 | 116 | 13.02% | Profile |
| SibSp | 0.0 | 1.0 | 1.0 | -1.5 | 2.5 | 46 | 5.16% | Profile |
| Parch | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 213 | 23.91% | Profile |

---

## 4. Correlation & Hypothesis Testing

The pipeline computed pairwise correlations and performed statistical hypothesis tests to identify significant predictors for the target variable (`Survived`).

### 4.1 Top Correlations with Target and Features

| Feature 1 | Feature 2 | Correlation |
|---|---|---|
| Pclass | Fare | -0.5495 |
| SibSp | Parch | 0.4148 |
| Pclass | Age | -0.3399 |
| Survived | Pclass | -0.3385 |
| Survived | Fare | 0.2573 |
| Pclass | Ticket | 0.2370 |
| Age | SibSp | -0.2333 |
| Parch | Fare | 0.2162 |
| SibSp | Ticket | 0.1836 |
| Age | Parch | -0.1725 |

### 4.2 Hypothesis Testing Results (Significant Predictors)

| Feature | Test Name | Statistic | P-Value | Significant? |
|---|---|---|---|---|
| Pclass | Pearson Correlation | -0.3385 | 2.54e-25 | Yes |
| Sex | Two-Sample Welch T-Test | 18.6718 | 2.28e-61 | Yes |
| Parch | Pearson Correlation | 0.0816 | 1.48e-02 | Yes |
| Ticket | Pearson Correlation | -0.1054 | 1.63e-03 | Yes |
| Fare | Pearson Correlation | 0.2573 | 6.12e-15 | Yes |
| Cabin | One-Way ANOVA | 1.8019 | 4.20e-07 | Yes |
| Embarked | One-Way ANOVA | 13.3269 | 1.98e-06 | Yes |
| Age | Pearson Correlation | -0.0649 | 5.28e-02 | No |
| SibSp | Pearson Correlation | -0.0353 | 2.92e-01 | No |

---

## 5. Feature Engineering & State Transitions

The pipeline tracked the evolution of the dataset across four state snapshots (`df_state_v0.csv` through `df_state_v3.csv`), documenting the transformation of raw data into a model-ready format.

### 5.1 State Transition Summary

| State File | Key Transformations Observed |
|---|---|
| df_state_v0.csv | Raw initial state. Ticket is a string; Age, Cabin, and Embarked contain missing values. |
| df_state_v1.csv | Ticket column converted to numeric (float64). Missing values in Age, Cabin, and Embarked begin to be populated. |
| df_state_v2.csv | Consistent state with v1. All missing values in Age, Cabin, Embarked, and Ticket are filled. |
| df_state_v3.csv | Final processed state. Identical to v2, confirming stable imputation and type casting. |

### 5.2 Engineered Features

No new engineered features were generated during this pipeline execution. The `engineered_features` list in the metrics output is empty.

---

## 6. Artifact Inventory

The following visualization artifacts were generated by the EDA pipeline and are available in the working directory.

### 6.1 Distribution Plots

| Artifact | Type | Size (KB) |
|---|---|---|
| dist_Age.png | Image Visualization | 41.31 |
| dist_Embarked.png | Image Visualization | 23.47 |
| dist_Fare.png | Image Visualization | 32.79 |
| dist_Parch.png | Image Visualization | 37.55 |
| dist_Pclass.png | Image Visualization | 41.48 |
| dist_Sex.png | Image Visualization | 24.80 |
| dist_SibSp.png | Image Visualization | 32.97 |
| dist_Survived.png | Image Visualization | 36.18 |
| feature_distributions.png | Image Visualization | 187.42 |

### 6.2 Interaction & Correlation Visuals

| Artifact | Type | Size (KB) |
|---|---|---|
| correlation_matrix.png | Image Visualization | 120.69 |
| target_interactions.png | Image Visualization | 52.84 |

---

## 7. Predictive Modeling Blueprint

Based on the EDA findings, the pipeline generated a structured blueprint for predictive modeling.

### 7.1 Model Configuration

| Parameter | Specification |
|---|---|
| Target | Survived |
| Problem Type | Classification |
| Dataset Dimensions | 891 rows x 12 columns |

### 7.2 Recommended Algorithms

1. Regularized Logistic Regression (baseline)
2. Random Forest Classifier
3. Gradient Boosting Classifier (XGBoost / LightGBM)
4. Support Vector Classifier (SVM)

### 7.3 Feature Selection Strategy

- Exclude high-cardinality ID or text name columns.
- Rank features using cross-validated permutation importance and mutual information.
- Remove collinear features exceeding correlation threshold > 0.85.

### 7.4 Validation Strategy

- Stratified K-Fold Cross-Validation (5 folds)
- Evaluation Metrics: Balanced Accuracy, Macro F1, Precision-Recall AUC, and Confusion Matrix.

### 7.5 Overfitting Risk Mitigation

- Apply regularization penalties (L1/L2).
- Limit tree depth and enforce minimum samples per leaf.
- Perform hyperparameter tuning strictly within cross-validation folds.

---

## 8. Key Findings & Executive Recommendations

1. **Survival is heavily influenced by Socio-economic status and Gender.** `Pclass`, `Sex`, and `Fare` are the strongest statistically significant predictors, with `Sex` yielding the highest test statistic (T = 18.67, p < 0.001).
2. **Cabin data is highly sparse.** With 77.1% missing values, `Cabin` requires careful handling (currently mode-imputed as 'B96 B98') or feature extraction (e.g., deck letter) before modeling.
3. **Fare distribution is highly skewed.** The skewness of 4.79 and the presence of 13% outliers suggest that log-transformation or robust scaling should be applied prior to model training.
4. **Family size proxies (SibSp, Parch) show weak direct correlation with survival.** While `SibSp` and `Parch` are correlated with each other (r = 0.41), their individual correlation with `Survived` is weak and not statistically significant in the Pearson test. A combined "FamilySize" feature may yield better predictive power.
5. **Modeling readiness:** The dataset is now clean, imputed, and ready for the predictive modeling phase as outlined in the blueprint. The recommended approach is to start with Regularized Logistic Regression as a baseline, then iterate with tree-based ensemble methods.