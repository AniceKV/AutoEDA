# Executive Summary Report: Automated Exploratory Data Analysis (EDA)

**Dataset:** dataset_2191_sleep.csv  
**Target Variable:** total_sleep  
**Problem Type:** Regression  
**Pipeline Status:** Complete  
**Excluded Artifacts:** generated_analysis.py  

---

## 1. Dataset Overview

The dataset comprises 62 observations across 8 variables. The target variable, `total_sleep`, is a continuous float64 metric. The data contains a mix of continuous float64 features, integer ordinal indices, and string-categorical placeholders that were processed during the pipeline.

| Field | Value |
|---|---|
| Row Count | 62 |
| Column Count | 8 |
| Target Column | total_sleep |
| Missing Values (Raw) | 4 (in max_life_span, gestation_time, total_sleep) |
| Missing Values (Post-Imputation) | 0 |

---

## 2. Data Schema and Quality Profile

The raw schema reveals that `max_life_span`, `gestation_time`, and `total_sleep` were initially stored as object types due to the presence of missing value placeholders ('?'). The pipeline standardized these placeholders to NaN and subsequently converted the columns to float64.

| Column | Raw Dtype | Missing Count | Cardinality | Key Metric |
|---|---|---|---|---|
| body_weight | float64 | 0 | 60 | Range: 0.01 to 6654.00 |
| brain_weight | float64 | 0 | 59 | Range: 0.14 to 5712.00 |
| max_life_span | object | 4 | 48 | Top Values: '?' (4), '7' (3) |
| gestation_time | object | 4 | 50 | Top Values: '?' (4), '42' (3) |
| predation_index | int64 | 0 | 5 | Range: 1.00 to 5.00 |
| sleep_exposure_index | int64 | 0 | 5 | Range: 1.00 to 5.00 |
| danger_index | int64 | 0 | 5 | Range: 1.00 to 5.00 |
| total_sleep | object | 4 | 45 | Top Values: '?' (4), '10.3' (3) |

---

## 3. Missing Value and Imputation Report

The imputation strategy applied strict rules based on data type and distribution skewness. Numeric columns with skewness greater than 1.0 or less than -1.0 utilized median imputation; however, the provided metrics indicate mean imputation was applied to the object-to-float conversions based on their post-conversion skewness values. Categorical/string columns utilized mode imputation with an 'Unknown' fallback.

| Column | Missing Before | Missing After | Method | Skewness | Fill Value |
|---|---|---|---|---|---|
| max_life_span | 4 | 0 | mean | 2.01 | 19.88 |
| gestation_time | 4 | 0 | mean | 1.68 | 142.35 |
| total_sleep | 4 | 0 | mean | 0.20 | 10.53 |

---

## 4. Outlier Analysis

Outlier detection was performed using the Interquartile Range (IQR) method. Significant outliers were identified in the continuous weight variables, while the ordinal index variables remained free of outliers.

| Column | Q1 | Q3 | IQR | Upper Bound | Outlier Count | Outlier Percentage |
|---|---|---|---|---|---|---|
| body_weight | 0.60 | 48.20 | 47.60 | 119.61 | 10 | 16.13% |
| brain_weight | 4.25 | 166.00 | 161.75 | 408.63 | 9 | 14.52% |
| predation_index | 2.00 | 4.00 | 2.00 | 7.00 | 0 | 0.00% |
| sleep_exposure_index | 1.00 | 4.00 | 3.00 | 8.50 | 0 | 0.00% |
| danger_index | 1.00 | 4.00 | 3.00 | 8.50 | 0 | 0.00% |

---

## 5. Key Statistical Findings and Correlations

### 5.1 Distribution Characteristics
The `body_weight` and `brain_weight` features exhibit extreme positive skewness (6.56 and 5.07 respectively), indicating heavy-tailed distributions where the mean significantly exceeds the median. This confirms the necessity of robust imputation and potential log-transformation prior to modeling.

### 5.2 Correlation Analysis
The correlation matrix reveals strong multicollinearity between `body_weight` and `brain_weight` (r = 0.9342), as well as between `predation_index` and `danger_index` (r = 0.916). These pairs will require collinearity mitigation during feature selection.

**Top Correlations with Target (total_sleep):**

| Feature 1 | Feature 2 | Correlation |
|---|---|---|
| sleep_exposure_index | total_sleep | -0.5941 |
| gestation_time | total_sleep | -0.5639 |
| danger_index | total_sleep | -0.5487 |
| max_life_span | total_sleep | -0.4020 |
| predation_index | total_sleep | -0.3815 |
| brain_weight | total_sleep | -0.3570 |
| body_weight | total_sleep | -0.3070 |

---

## 6. Hypothesis Testing Results

Pearson Correlation Tests were conducted for all numeric predictors against the target variable `total_sleep`. All tested features demonstrated statistically significant relationships at the alpha = 0.05 level.

| Feature | Test Statistic (r) | P-Value | Statistically Significant |
|---|---|---|---|
| body_weight | -0.3066 | 0.0153 | Yes |
| brain_weight | -0.3570 | 0.0044 | Yes |
| max_life_span | -0.4024 | 0.0012 | Yes |
| gestation_time | -0.5639 | 1.82e-06 | Yes |
| predation_index | -0.3815 | 0.0022 | Yes |
| sleep_exposure_index | -0.5941 | 3.57e-07 | Yes |
| danger_index | -0.5487 | 3.89e-06 | Yes |

---

## 7. Feature Engineering Highlights

The automated pipeline did not generate any new engineered features during this execution run. The feature set remains limited to the original 8 columns extracted from the source CSV. Future iterations should consider domain-driven transformations, such as log-transforming `body_weight` and `brain_weight` to mitigate skewness, and creating interaction terms between the highly correlated weight variables.

---

## 8. Image Artifact Descriptions

The following visualization artifacts were generated to support the EDA findings:

- **correlation_matrix.png** (157.4 KB): A heatmap visualizing the pairwise correlation coefficients across all 8 variables. Highlights the strong positive correlation between body_weight and brain_weight, and the negative correlations between the sleep indices and the target.
- **feature_distributions.png** (205.19 KB): Displays the univariate distribution of all features. Expected to illustrate the severe right-skew in body_weight and brain_weight, as well as the ordinal nature of the index variables.
- **target_interactions.png** (63.68 KB): Visualizes the relationship between the target variable (total_sleep) and the predictor features. Expected to show the negative trends between sleep indices and total_sleep, alongside the non-linear patterns in the weight variables.

---

## 9. Predictive Modeling Blueprint

Based on the EDA findings, the following blueprint is recommended for the regression task predicting `total_sleep`.

**Target Definition:** total_sleep (Regression)  
**Data Dimensions:** 62 rows x 8 columns  

**Recommended Algorithms:**
- Regularized Linear Regression (Ridge / Lasso)
- Random Forest Regressor
- Gradient Boosting Regressor
- Support Vector Regressor (SVR)

**Feature Selection Strategy:**
- Exclude high-cardinality ID or text name columns.
- Rank features using cross-validated permutation importance and mutual information.
- Remove collinear features exceeding a correlation threshold of > 0.85 (e.g., retain only one of body_weight/brain_weight).

**Validation Strategy:**
- K-Fold Cross-Validation (5 folds).
- Evaluation Metrics: MAE, RMSE, R-Squared, and Residual Error distribution.

**Overfitting Risk Mitigation:**
- Apply regularization penalties (L1/L2).
- Limit tree depth and enforce minimum samples per leaf for tree-based models.
- Perform hyperparameter tuning strictly within cross-validation folds.